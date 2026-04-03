# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    MXLinearConfig,
    ScaleCalculationMode,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    Mxfp8LinearOp, Mxfp8LinearBackend, mxfp8_e4m3_quantize)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform
import time
from typing import Optional


if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Mxfp8Config(QuantizationConfig):
    """Config class for MXFP8."""

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
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(
                self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
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
            # TEMP ASSERTION: used only for rollout quant-scope validation.
            # Remove after confirming llm_only mode truly skips all visual layers.
            if (self.skip_with_substr and is_vision_layer
                    and any("visual" in name for name in self.ignored_layers)):
                assert is_skipped, (
                    f"TEMP ASSERTION FAILED: visual layer should be skipped in "
                    f"llm_only verification mode, but got quantized: {prefix}"
                )
            if is_skipped:
                return UnquantizedLinearMethod()
            return Mxfp8LinearMethod(self, prefix=prefix)
        return None

    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class Mxfp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Mxfp8Config, prefix: str = ""):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.weight_block_size = [1, 32]
        self.prefix = prefix
        self.block_quant = self.weight_block_size is not None
        self.backend: Mxfp8LinearBackend = Mxfp8LinearBackend.FLASHINFER_CUTLASS
        self.mxfp8_linear_op = Mxfp8LinearOp(
            backend=self.backend,
        )

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
                input_size_per_partition // 32,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight.data
        N, K = weight.shape
        
        min_dim = 128
        
        # 1. 计算 Padding
        pad_N = (min_dim - (N % min_dim)) % min_dim
        pad_K = (min_dim - (K % min_dim)) % min_dim
        
        # 2. 执行 Padding
        if pad_N > 0 or pad_K > 0:
            weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))
        
        # 3. 量化 (⚠️ 必须设为 False，PyTorch 只要 contiguous 的二维 scale)
        weight_mxfp8, weight_scale = mxfp8_e4m3_quantize(
            weight,
            is_sf_swizzled_layout=True  
        )
        
        layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        
        # 记录原始大小供截断使用
        layer.orig_N = N
        layer.orig_K = K
        

    def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        return self.mxfp8_linear_op.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias
        )
