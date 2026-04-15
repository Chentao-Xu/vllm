# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp_quant import (
    fused_quantize_mx,
    matmul_mxf4_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


logger = init_logger(__name__)
_MXFP4_BLOCK_SIZE = 32
_VALID_HADAMARD_GROUP_SIZES = (32, 64, 128)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _pad_last_dim(tensor: torch.Tensor, target_last_dim: int) -> torch.Tensor:
    current_last_dim = tensor.shape[-1]
    if target_last_dim < current_last_dim:
        raise ValueError(
            f"target_last_dim={target_last_dim} is smaller than current "
            f"last dim={current_last_dim}"
        )
    if target_last_dim == current_last_dim:
        return tensor
    pad_cols = target_last_dim - current_last_dim
    return torch.nn.functional.pad(tensor, (0, pad_cols))


_HADAMARD_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _build_hadamard_matrix(
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not _is_power_of_two(group_size):
        raise ValueError("hadamard_group_size must be a power of 2")

    hadamard = torch.ones((1, 1), device=device, dtype=torch.float32)
    current_size = 1
    while current_size < group_size:
        top = torch.cat([hadamard, hadamard], dim=1)
        bottom = torch.cat([hadamard, -hadamard], dim=1)
        hadamard = torch.cat([top, bottom], dim=0)
        current_size *= 2

    hadamard = hadamard / math.sqrt(group_size)
    return hadamard.to(dtype=torch.bfloat16)


def _get_hadamard_matrix(
    group_size: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    cache_key = (group_size, device_type, device_index)
    if cache_key not in _HADAMARD_CACHE:
        device = torch.device(device_type, device_index)
        _HADAMARD_CACHE[cache_key] = _build_hadamard_matrix(group_size, device)
    return _HADAMARD_CACHE[cache_key]


class QutlassOnlineMxfp4Config(QuantizationConfig):
    def __init__(
        self,
        hadamard_group_size: int = 32,
        forward_method: str = "abs_max",
        ignored_layers: list[str] | None = None,
        skip_with_substr: bool = False,
    ) -> None:
        super().__init__()
        if hadamard_group_size not in _VALID_HADAMARD_GROUP_SIZES:
            raise ValueError(
                "qutlass_online_mxfp4 supports hadamard_group_size in "
                f"{_VALID_HADAMARD_GROUP_SIZES}, got {hadamard_group_size}."
            )
        self.hadamard_group_size = hadamard_group_size
        self.k_alignment = math.lcm(_MXFP4_BLOCK_SIZE, hadamard_group_size)
        self.forward_method = forward_method
        self.ignored_layers = ignored_layers or []
        # Keep backward compatibility with config names used by other quantizers.
        self.modules_to_not_convert = self.ignored_layers
        self.skip_with_substr = skip_with_substr

    def __repr__(self) -> str:
        return (
            "QutlassOnlineMxfp4Config("
            f"hadamard_group_size={self.hadamard_group_size}, "
            f"forward_method={self.forward_method}, "
            f"modules_to_not_convert={self.modules_to_not_convert}, "
            f"skip_with_substr={self.skip_with_substr})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "qutlass_online_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)
            self.modules_to_not_convert = self.ignored_layers

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QutlassOnlineMxfp4Config":
        hadamard_group_size = cls.get_from_keys_or(
            config, ["hadamard_group_size"], 32
        )
        forward_method = cls.get_from_keys_or(config, ["forward_method"], "abs_max")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        skip_with_substr = cls.get_from_keys_or(config, ["skip_with_substr"], False)
        return cls(
            hadamard_group_size=hadamard_group_size,
            forward_method=forward_method,
            ignored_layers=ignored_layers,
            skip_with_substr=skip_with_substr,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            is_skipped = is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                skip_with_substr=self.skip_with_substr,
            )
            is_vision_layer = prefix.startswith("visual.") or ".visual." in prefix
            if is_skipped and is_vision_layer:
                logger.warning(
                    "\033[92m[QuantScope] Skip quantization for vision layer: %s\033[0m",
                    prefix,
                )
            if (
                self.skip_with_substr
                and is_vision_layer
                and any("visual" in name for name in self.ignored_layers)
            ):
                assert is_skipped, (
                    f"TEMP ASSERTION FAILED: visual layer should be skipped in "
                    f"llm_only verification mode, but got quantized: {prefix}"
                )
            if is_skipped:
                return UnquantizedLinearMethod()
            return QutlassOnlineMxfp4LinearMethod(self)
        return None


class QutlassOnlineMxfp4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: QutlassOnlineMxfp4Config):
        self.quant_config = quant_config
        self.group_size = quant_config.hadamard_group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size
        del output_size

        if params_dtype != torch.bfloat16:
            raise ValueError("Only bfloat16 is supported by qutlass_online_mxfp4")

        if self.group_size not in _VALID_HADAMARD_GROUP_SIZES:
            raise ValueError(
                "qutlass_online_mxfp4 supports hadamard_group_size in "
                f"{_VALID_HADAMARD_GROUP_SIZES}, got {self.group_size}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        padded_input_size_per_partition = _round_up(
            input_size_per_partition, self.quant_config.k_alignment
        )
        layer.orig_input_size_per_partition = input_size_per_partition
        layer.padded_input_size_per_partition = padded_input_size_per_partition
        layer.qutlass_padding_cols = (
            padded_input_size_per_partition - input_size_per_partition
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.bfloat16,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        weight_attrs = dict(extra_weight_attrs)
        weight_attrs.pop("weight_loader", None)
        set_weight_attrs(weight, weight_attrs)
        layer.register_parameter("weight", weight)

        rounded_m = _round_up(output_size_per_partition, 128)
        rounded_n = _round_up(
            padded_input_size_per_partition // _MXFP4_BLOCK_SIZE, 4
        )
        weight_scale = Parameter(
            torch.empty(rounded_m, rounded_n, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"ignore_warning": True})
        layer.register_parameter("weight_scale", weight_scale)

        hadamard_matrix = Parameter(
            torch.empty(
                self.group_size,
                self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(hadamard_matrix, {"ignore_warning": True})
        layer.register_parameter("hadamard_matrix", hadamard_matrix)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        hadamard_matrix = _get_hadamard_matrix(
            self.group_size,
            layer.weight.device.type,
            layer.weight.device.index,
        )

        orig_n = int(layer.weight.shape[0])
        orig_k = int(
            getattr(layer, "orig_input_size_per_partition", layer.weight.shape[1])
        )
        padded_n = _round_up(orig_n, 128)
        padded_k = int(getattr(layer, "padded_input_size_per_partition", orig_k))
        weight_bf16 = layer.weight.data.to(torch.bfloat16)
        if padded_n > weight_bf16.shape[0]:
            layer_name = getattr(layer, "prefix", layer.__class__.__name__)
            logger.warning(
                "[qutlass_online_mxfp4] Padding layer %s output dim from %d to %d.",
                layer_name,
                weight_bf16.shape[0],
                padded_n,
            )
            weight_bf16 = torch.nn.functional.pad(
                weight_bf16, (0, 0, 0, padded_n - weight_bf16.shape[0])
            )
        if padded_k > weight_bf16.shape[-1]:
            layer_name = getattr(layer, "prefix", layer.__class__.__name__)
            logger.warning(
                "[qutlass_online_mxfp4] Padding layer %s input dim from %d to %d.",
                layer_name,
                weight_bf16.shape[-1],
                padded_k,
            )
            weight_bf16 = _pad_last_dim(weight_bf16, padded_k)
        elif padded_k < weight_bf16.shape[-1]:
            raise RuntimeError(
                "qutlass_online_mxfp4 got unexpected padded_k "
                f"{padded_k} < weight_k {weight_bf16.shape[-1]}."
            )

        layer.orig_output_size_per_partition = orig_n
        layer.padded_output_size_per_partition = padded_n
        q_weight, weight_scales = fused_quantize_mx(
            weight_bf16,
            hadamard_matrix,
            self.quant_config.forward_method,
        )

        layer.hadamard_matrix.data = hadamard_matrix.contiguous()
        layer.weight.data = q_weight.contiguous()
        layer.weight_scale.data = weight_scales.view(torch.uint8).contiguous()

        # Keep a BF16 fallback tensor for rare qutlass internal kernel errors.
        layer.weight_bf16_fallback = weight_bf16.contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_flat = x.contiguous().flatten(end_dim=-2)
        expected_k = int(
            getattr(layer, "padded_input_size_per_partition", x_flat.shape[-1])
        )
        if x_flat.shape[-1] > expected_k:
            raise RuntimeError(
                "qutlass_online_mxfp4 input K dim exceeds expected padded K: "
                f"got {x_flat.shape[-1]}, expected <= {expected_k}."
            )
        x_flat_padded = _pad_last_dim(x_flat, expected_k)
        x_flat_q, x_flat_scales = fused_quantize_mx(
            x_flat_padded,
            layer.hadamard_matrix,
            self.quant_config.forward_method,
        )

        alpha = torch.ones(1, dtype=torch.float32, device=x.device)
        try:
            y = matmul_mxf4_bf16(
                x_flat_q,
                layer.weight.data,
                x_flat_scales,
                layer.weight_scale.data,
                alpha,
            )
        except RuntimeError as exc:
            if "Error Internal" not in str(exc):
                raise
            y = torch.matmul(
                x_flat_padded.to(torch.bfloat16), layer.weight_bf16_fallback.t()
            )

        orig_n = int(getattr(layer, "orig_output_size_per_partition", y.shape[-1]))
        if y.shape[-1] != orig_n:
            y = y[..., :orig_n]
        y = y.view(*x.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias

        return y


# Backward-compatible aliases.
CutlassOnlineMxfp4Config = QutlassOnlineMxfp4Config
CutlassOnlineMxfp4LinearMethod = QutlassOnlineMxfp4LinearMethod
