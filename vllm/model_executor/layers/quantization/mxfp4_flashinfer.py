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


class MyMxfp4Config(QuantizationConfig):
    """Config class for custom MXFP4 linear quantization."""

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "my_mxfp4"

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
    def from_config(cls, config: dict[str, Any]) -> "MyMxfp4Config":
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config,
                                                  ["modules_to_not_convert"],
                                                  None)
        return cls(activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers)


    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if current_platform.is_xpu():
            return self.get_xpu_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return MyMxfp4LinearMethod(self)
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


class MyMxfp4LinearMethod(LinearMethodBase):

    def __init__(self, quant_config: MyMxfp4Config):
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

        weight_q, weight_s = mxfp4_quantize(layer.weight.data)
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

        weight = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if weight is None or weight_scale is None:
            raise RuntimeError("FlashInfer MXFP4 weights are not initialized")

        input_orig_shape = x.shape
        input_2d = x.reshape(-1, input_orig_shape[-1]).contiguous()
        x_quant, x_scale = mxfp4_quantize(
            input_2d,
        )
        out_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        output = mm_fp4(
            x_quant,
            weight.t(),
            x_scale,
            weight_scale.t(),
            out_dtype=out_dtype,
            block_size=self.group_size,
            backend="cudnn",
            use_nvfp4=False,
        )

        if bias is not None:
            output = output + bias
        return output.reshape(*input_orig_shape[:-1], output.shape[-1])
