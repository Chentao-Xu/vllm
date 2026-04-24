# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    _mx_inference_linear_transform,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common.kernel_preference import (
    KernelPreference,
)

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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.platforms import current_platform


if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


def pad_mxfp4_weight_for_flashinfer(
    weight: torch.Tensor,
    alignment: int = 128,
    row_alignment: int = 128,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Pad dense BF16/FP16 weight for FlashInfer MXFP4 path."""
    if weight.ndim != 2:
        raise ValueError(
            f"Expected 2D weight for MXFP4 FlashInfer path, got shape {tuple(weight.shape)}"
        )

    orig_n, orig_k = weight.shape
    pad_n = (-orig_n) % row_alignment
    pad_k = (-orig_k) % alignment
    if pad_n > 0 or pad_k > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n)).contiguous()
    return weight, orig_n, orig_k, pad_n, pad_k


def pad_mxfp4_input_for_flashinfer(
    x2d: torch.Tensor,
    expected_k: int,
    row_alignment: int = 128,
) -> tuple[torch.Tensor, int, int]:
    """Pad input rows and K-dimension to FlashInfer MXFP4 expected sizes."""
    orig_m = x2d.shape[0]
    input_k = x2d.shape[-1]
    pad_m = (-orig_m) % row_alignment
    if input_k == expected_k:
        padded = x2d
    elif input_k < expected_k:
        padded = torch.nn.functional.pad(x2d, (0, expected_k - input_k))
    else:
        raise RuntimeError(
            f"Input hidden size {input_k} exceeds expected {expected_k} for MXFP4 matmul"
        )

    if pad_m > 0:
        padded = torch.nn.functional.pad(padded, (0, 0, 0, pad_m))

    return padded.contiguous(), orig_m, pad_m


def slice_mxfp4_flashinfer_rows(
    out2d: torch.Tensor,
    orig_m: int,
) -> torch.Tensor:
    """Slice FlashInfer MXFP4 output rows back to original token count."""
    if out2d.shape[0] < orig_m:
        raise RuntimeError(
            f"Output row count {out2d.shape[0]} is smaller than original {orig_m}"
        )
    if out2d.shape[0] == orig_m:
        return out2d
    return out2d[:orig_m].contiguous()


def slice_mxfp4_flashinfer_output(
    out2d: torch.Tensor,
    orig_n: int,
) -> torch.Tensor:
    """Slice FlashInfer MXFP4 output back to original output size."""
    if out2d.shape[-1] < orig_n:
        raise RuntimeError(
            f"Output hidden size {out2d.shape[-1]} is smaller than original {orig_n}"
        )
    if out2d.shape[-1] == orig_n:
        return out2d
    return out2d[:, :orig_n].contiguous()


class Mxfp4FlashinferConfig(QuantizationConfig):
    """Config class for custom MXFP4 linear quantization."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[list[str]] = None,
        skip_with_substr: bool = False,
    ) -> None:
        super().__init__()

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.skip_with_substr = skip_with_substr

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4_flashinfer"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(
                self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp4FlashinferConfig":
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config,
                                                  ["modules_to_not_convert"],
                                                  None)
        skip_with_substr = cls.get_from_keys_or(config, ["skip_with_substr"],
                                                False)
        return cls(activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers,
                   skip_with_substr=skip_with_substr)


    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if current_platform.is_xpu():
            return self.get_xpu_quant_method(layer, prefix)
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
            if (self.skip_with_substr and is_vision_layer
                    and any("visual" in name for name in self.ignored_layers)):
                assert is_skipped, (
                    f"TEMP ASSERTION FAILED: visual layer should be skipped in "
                    f"llm_only verification mode, but got quantized: {prefix}"
                )
            if is_skipped:
                return UnquantizedLinearMethod()
            return Mxfp4FlashinferLinearMethod(self)
        return None

    def get_cache_scale(self, name: str) -> Optional[str]:
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        return None


class Mxfp4FlashinferLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: Mxfp4FlashinferConfig):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        self.group_size = 32

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

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=params_dtype),
                                        input_dim=1,
                                        output_dim=0,
                                        weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                max(1, input_size_per_partition // self.group_size),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Prefer FlashInfer-native MXFP4 quantization for weights when available,
        # because the GEMM path consumes packed uint8 weights and uint8 scales.
        from flashinfer import mxfp4_quantize

        padded_weight, orig_n, orig_k, pad_n, pad_k = (
            pad_mxfp4_weight_for_flashinfer(
                layer.weight.data,
                alignment=128,
                row_alignment=128,
            )
        )

        layer.orig_N = orig_n
        layer.orig_K = orig_k
        layer.pad_N = pad_n
        layer.pad_K = pad_k
        layer.mxfp4_expected_n = orig_n + pad_n
        layer.mxfp4_expected_k = orig_k + pad_k
        if pad_n > 0 or pad_k > 0:
            logger.warning(
                "[TempPadding][MXFP4][weight] layer=%s orig_shape=(%d, %d) "
                "padded_shape=(%d, %d) pad_n=%d pad_k=%d",
                getattr(layer, "prefix", layer.__class__.__name__),
                orig_n,
                orig_k,
                layer.mxfp4_expected_n,
                layer.mxfp4_expected_k,
                pad_n,
                pad_k,
            )

        weight_q, weight_s = mxfp4_quantize(padded_weight)
        layer.weight = torch.nn.Parameter(weight_q.contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_s.contiguous(), requires_grad=False)
        return


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4, mxfp4_quantize
        from flashinfer.autotuner import AutoTuner

        weight = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight is None or weight_scale is None:
            raise RuntimeError("FlashInfer MXFP4 weights are not initialized")

        input_orig_shape = x.shape
        input_2d = x.reshape(-1, input_orig_shape[-1]).contiguous()

        expected_k = getattr(layer, "mxfp4_expected_k", None)
        if expected_k is None:
            expected_k = int(weight.shape[1]) * 2
        expected_k = int(expected_k)
        if expected_k % self.group_size != 0:
            raise RuntimeError(
                f"MXFP4 expected_k={expected_k} for layer "
                f"{getattr(layer, 'prefix', layer.__class__.__name__)} is not divisible by "
                f"{self.group_size}"
            )

        input_2d, orig_m, _pad_m = pad_mxfp4_input_for_flashinfer(
            input_2d,
            expected_k,
            row_alignment=128,
        )
        if _pad_m > 0 or input_orig_shape[-1] != expected_k:
            logger.warning(
                "[TempPadding][MXFP4][input] layer=%s orig_shape=(%d, %d) "
                "padded_shape=(%d, %d) pad_m=%d pad_k=%d",
                getattr(layer, "prefix", layer.__class__.__name__),
                orig_m,
                input_orig_shape[-1],
                input_2d.shape[0],
                input_2d.shape[-1],
                _pad_m,
                expected_k - input_orig_shape[-1],
            )
        if input_2d.shape[-1] != expected_k:
            raise RuntimeError(
                f"MXFP4 input padding failed for layer "
                f"{getattr(layer, 'prefix', layer.__class__.__name__)}: got "
                f"{input_2d.shape[-1]}, expected {expected_k}"
            )

        x_quant, x_scale = mxfp4_quantize(input_2d)
        out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

        def run_mm_fp4() -> torch.Tensor:
            return mm_fp4(
                x_quant,
                weight.t(),
                x_scale,
                weight_scale.t(),
                out_dtype=out_dtype,
                block_size=self.group_size,
                backend="cudnn",
                use_nvfp4=False,
            )

        try:
            output = run_mm_fp4()
        except RuntimeError as err:
            if "Plan index" not in str(err):
                raise
            logger.warning(
                "FlashInfer cuDNN FP4 tactic cache mismatch for MXFP4; "
                "clearing autotuner cache and retrying once."
            )
            AutoTuner.get().clear_cache()
            output = run_mm_fp4()

        output = slice_mxfp4_flashinfer_rows(output, int(orig_m))
        orig_n = getattr(layer, "orig_N", weight.shape[0])
        output = slice_mxfp4_flashinfer_output(output, int(orig_n))

        if bias is not None:
            if bias.shape[0] != output.shape[-1]:
                raise RuntimeError(
                    f"Bias shape {tuple(bias.shape)} does not match MXFP4 output "
                    f"shape {tuple(output.shape)} after slicing"
                )
            output = output + bias
        return output.reshape(*input_orig_shape[:-1], output.shape[-1])
