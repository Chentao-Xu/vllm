# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from typing import Any

import torch
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp_quant import (
    fused_quantize_mx,
    matmul_mxf4_bf16,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger


logger = init_logger(__name__)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


_ROTATION_CACHE: dict[
    tuple[int, str, int | None], torch.Tensor
] = {}
_MXFP4_SCALE_GROUP_SIZE = 32
_SUPPORTED_MXFP4_ROTATION_GROUP_SIZES = (32,)
_DEFAULT_MXFP4_ABS_MAX_GLOBAL_SCALE = 3.0
_DEFAULT_MXFP4_QUEST_GLOBAL_SCALE = 1.0


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _build_hadamard_matrix(
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not _is_power_of_two(group_size):
        raise ValueError("hadamard_group_size must be a power of 2")

    return (
        deterministic_hadamard_matrix(
            group_size,
            dtype=torch.float32,
            device=device,
        )
        / math.sqrt(group_size)
    ).to(dtype=torch.bfloat16)


def _default_global_scale(forward_method: str) -> float:
    if forward_method == "abs_max":
        # fusedQuantizeMx(abs_max) scales FP4 values by 3.0. Applying this
        # factor on both activation and weight sides requires alpha=1/9.
        return _DEFAULT_MXFP4_ABS_MAX_GLOBAL_SCALE
    if forward_method == "quest":
        return _DEFAULT_MXFP4_QUEST_GLOBAL_SCALE
    raise ValueError(
        "qutlass_online_mxfp4.forward_method must be one of: "
        "['abs_max', 'quest']"
    )


def _replace_or_copy_parameter_data(
    parameter: Parameter,
    value: torch.Tensor,
) -> None:
    value = value.contiguous()
    if parameter.data.shape == value.shape and parameter.data.dtype == value.dtype:
        parameter.data.copy_(value)
    else:
        parameter.data = value


def _build_rotation_matrix(
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Keep the simplest deterministic Sylvester Hadamard construction.
    return _build_hadamard_matrix(group_size, device)


def _get_rotation_matrix(
    group_size: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    cache_key = (
        group_size,
        device_type,
        device_index,
    )
    if cache_key not in _ROTATION_CACHE:
        device = torch.device(device_type, device_index)
        _ROTATION_CACHE[cache_key] = _build_rotation_matrix(
            group_size=group_size,
            device=device,
        )
    return _ROTATION_CACHE[cache_key]


class QutlassOnlineMxfp4Config(QuantizationConfig):
    def __init__(
        self,
        hadamard_group_size: int = 32,
        forward_method: str = "abs_max",
        rotation_mode: str = "hadamard",
        rotation_seed: int = 0,
        use_random_permutation: bool = False,
        global_scale: float | None = None,
        modules_to_not_convert: list[str] | None = None,
        skip_with_substr: bool = False,
    ) -> None:
        super().__init__()
        self.hadamard_group_size = hadamard_group_size
        self.forward_method = forward_method
        # Keep API compatibility with existing configs but force deterministic
        # Sylvester-Hadamard behavior.
        self.rotation_mode = "hadamard"
        self.rotation_seed = 0
        self.use_random_permutation = False
        self.global_scale = (
            _default_global_scale(forward_method)
            if global_scale is None
            else global_scale
        )
        self.modules_to_not_convert = modules_to_not_convert
        self.skip_with_substr = skip_with_substr

        if self.forward_method not in ("abs_max", "quest"):
            raise ValueError(
                "qutlass_online_mxfp4.forward_method must be one of: "
                "['abs_max', 'quest']"
            )
        if rotation_mode != "hadamard":
            raise ValueError(
                "qutlass_online_mxfp4.rotation_mode must be one of: "
                "['hadamard']"
            )
        if self.hadamard_group_size != 32:
            raise ValueError(
                "qutlass_online_mxfp4.hadamard_group_size must be one of: "
                f"{list(_SUPPORTED_MXFP4_ROTATION_GROUP_SIZES)}"
            )
        if self.global_scale <= 0:
            raise ValueError("qutlass_online_mxfp4.global_scale must be > 0")

    def __repr__(self) -> str:
        return (
            "QutlassOnlineMxfp4Config("
            f"hadamard_group_size={self.hadamard_group_size}, "
            f"forward_method={self.forward_method}, "
            f"rotation_mode={self.rotation_mode}, "
            f"rotation_seed={self.rotation_seed}, "
            f"use_random_permutation={self.use_random_permutation}, "
            f"global_scale={self.global_scale}, "
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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QutlassOnlineMxfp4Config":
        hadamard_group_size = cls.get_from_keys_or(
            config, ["hadamard_group_size"], 32
        )
        forward_method = os.getenv(
            "QUTLASS_FORWARD_METHOD",
            cls.get_from_keys_or(config, ["forward_method"], "abs_max"),
        )
        rotation_mode = os.getenv(
            "QUTLASS_ROTATION_MODE",
            cls.get_from_keys_or(config, ["rotation_mode"], "hadamard"),
        )
        rotation_seed = cls.get_from_keys_or(config, ["rotation_seed"], 0)
        use_random_permutation = cls.get_from_keys_or(
            config, ["use_random_permutation"], False
        )
        global_scale_env = os.getenv("QUTLASS_GLOBAL_SCALE")
        global_scale = (
            float(global_scale_env)
            if global_scale_env is not None
            else cls.get_from_keys_or(config, ["global_scale"], None)
        )
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        skip_with_substr = cls.get_from_keys_or(config, ["skip_with_substr"], False)
        return cls(
            hadamard_group_size=hadamard_group_size,
            forward_method=forward_method,
            rotation_mode=rotation_mode,
            rotation_seed=rotation_seed,
            use_random_permutation=use_random_permutation,
            global_scale=global_scale,
            modules_to_not_convert=modules_to_not_convert,
            skip_with_substr=skip_with_substr,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        if isinstance(layer, LinearBase):
            if self.modules_to_not_convert is not None:
                if self.skip_with_substr:
                    is_skipped = any(
                        module in prefix for module in self.modules_to_not_convert
                    )
                else:
                    is_skipped = any(
                        prefix.endswith(module)
                        for module in self.modules_to_not_convert
                    )
                if is_skipped:
                    if prefix.startswith("visual.") or ".visual." in prefix:
                        logger.warning(
                            "\033[92m[QuantScope] Skip quantization for vision layer: %s\033[0m",
                            prefix,
                        )
                    return UnquantizedLinearMethod()
            return QutlassOnlineMxfp4LinearMethod(self, layer_prefix=prefix)
        return None


class QutlassOnlineMxfp4LinearMethod(LinearMethodBase):
    def __init__(
        self,
        quant_config: QutlassOnlineMxfp4Config,
        layer_prefix: str,
    ):
        self.quant_config = quant_config
        self.group_size = quant_config.hadamard_group_size
        self.scale_group_size = _MXFP4_SCALE_GROUP_SIZE
        self.layer_prefix = layer_prefix

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

        if self.group_size not in _SUPPORTED_MXFP4_ROTATION_GROUP_SIZES:
            raise ValueError(
                "qutlass_online_mxfp4 expects hadamard_group_size to be one of "
                f"{list(_SUPPORTED_MXFP4_ROTATION_GROUP_SIZES)}"
            )

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                "The input size must be divisible by hadamard_group_size for qutlass_online_mxfp4"
            )

        output_size_per_partition = sum(output_partition_sizes)
        rounded_output_size_per_partition = _round_up(output_size_per_partition, 128)
        rounded_scale_cols = _round_up(
            input_size_per_partition // self.scale_group_size,
            4,
        )

        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.bfloat16,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        common_attrs = dict(extra_weight_attrs)
        common_attrs.pop("weight_loader", None)
        set_weight_attrs(weight, common_attrs)
        layer.register_parameter("weight", weight)

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 2,
            }
            | common_attrs,
        )
        layer.register_parameter("qweight", qweight)

        weight_scale = Parameter(
            torch.empty(
                rounded_output_size_per_partition,
                rounded_scale_cols,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_scale,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.scale_group_size,
                "ignore_warning": True,
            }
            | common_attrs,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(weight_global_scale, {"ignore_warning": True} | common_attrs)
        layer.register_parameter("weight_global_scale", weight_global_scale)

        act_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(act_global_scale, {"ignore_warning": True} | common_attrs)
        layer.register_parameter("act_global_scale", act_global_scale)

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
        raw_weight = layer.weight.data
        hadamard_matrix = _get_rotation_matrix(
            self.group_size,
            layer.weight.device.type,
            layer.weight.device.index,
        )

        if not torch.isfinite(raw_weight).all():
            raise ValueError("qutlass_online_mxfp4: loaded BF16 weights contain NaN/Inf")

        weight_bf16 = raw_weight.to(torch.bfloat16)
        q_weight, weight_scales = fused_quantize_mx(
            weight_bf16,
            hadamard_matrix,
            self.quant_config.forward_method,
        )

        if not torch.isfinite(weight_scales.view(torch.float8_e8m0fnu).to(torch.float32)).all():
            raise ValueError("qutlass_online_mxfp4: fused MX weight scales contain NaN/Inf")

        _replace_or_copy_parameter_data(layer.hadamard_matrix, hadamard_matrix)
        _replace_or_copy_parameter_data(layer.qweight, q_weight)
        _replace_or_copy_parameter_data(
            layer.weight_scale,
            weight_scales.view(torch.uint8),
        )
        _replace_or_copy_parameter_data(
            layer.weight_global_scale,
            torch.full(
                (1,),
                fill_value=self.quant_config.global_scale,
                dtype=torch.float32,
                device=layer.weight.device,
            ),
        )
        _replace_or_copy_parameter_data(
            layer.act_global_scale,
            torch.full(
                (1,),
                fill_value=self.quant_config.global_scale,
                dtype=torch.float32,
                device=layer.weight.device,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_flat = x.contiguous().flatten(end_dim=-2)
        x_flat_q, x_flat_scales = fused_quantize_mx(
            x_flat,
            layer.hadamard_matrix,
            self.quant_config.forward_method,
        )

        _replace_or_copy_parameter_data(
            layer.act_global_scale,
            torch.full(
                (1,),
                fill_value=self.quant_config.global_scale,
                dtype=torch.float32,
                device=x.device,
            ),
        )

        alpha = 1 / (
            layer.weight_global_scale.to(torch.float32)
            * layer.act_global_scale.to(torch.float32)
        )
        y = matmul_mxf4_bf16(
            x_flat_q,
            layer.qweight.data,
            x_flat_scales,
            layer.weight_scale.data,
            alpha,
        )

        y = y.view(*x.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias

        return y


# Backward-compatible aliases.
CutlassOnlineMxfp4Config = QutlassOnlineMxfp4Config
CutlassOnlineMxfp4LinearMethod = QutlassOnlineMxfp4LinearMethod
