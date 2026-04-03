# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

from vllm.logger import init_logger
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class Mxfp8LinearBackend(Enum):
    EMULATION = "emulation"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"


# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    x_q, x_scales = flashinfer_mxfp8_quantize(
        x, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class Mxfp8LinearOp:
    def __init__(self, backend: Mxfp8LinearBackend):
        if backend not in Mxfp8LinearBackend:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend = backend

    def _apply_emulation(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Validate weight_scale dtype and shape (must be 2D for TORCH backend)
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"TORCH backend requires {MXFP8_SCALE_DTYPE} weight_scale dtype, "
                f"got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"TORCH backend requires 2D weight_scale, got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )

        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale)

        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    def _apply_flashinfer_cutlass(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N_padded, K_padded = weight.shape
        input_shape = input.shape
        # 获取真实的 K
        K_orig = input_shape[-1]
        input_2d = input.view(-1, K_orig)
        M_orig = input_2d.shape[0]
        
        min_dim = 128

        # 计算 M 的 Padding
        M_padded = ((M_orig + min_dim - 1) // min_dim) * min_dim
        pad_rows = M_padded - M_orig
        
        # 计算 K 的 Padding（要和权重的 K_padded 对齐）
        pad_cols = K_padded - K_orig

        # 对输入进行 Padding
        if pad_rows > 0 or pad_cols > 0:
            input_2d = torch.nn.functional.pad(input_2d, (0, pad_cols, 0, pad_rows))

        # 动态量化输入 (⚠️ 设为 False)
        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True, 
        )
        if not weight.is_contiguous():
            weight = weight.contiguous()
        scale_a_2d = input_scale.view(M_padded, K_padded // 32).view(torch.float8_e8m0fnu)
        scale_b_2d = weight_scale.view(N_padded, K_padded // 32).view(torch.float8_e8m0fnu)
        # 调用 PyTorch 原生算子
        output = torch._scaled_mm(
            input_mxfp8,
            weight.t(),
            scale_a_2d,
            scale_b_2d,
            bias=None,
            out_dtype=torch.bfloat16,
        )
        # 截断多余的 M 维度
        if pad_rows > 0:
            output = output[:M_orig, :]
            
        # 截断多余的 N 维度 (可以通过 bias 的长度获取真实的 N，或者传 layer.orig_N 进来)
        orig_N = bias.shape[0] if bias is not None else (N_padded - (min_dim - (N_padded % min_dim)) % min_dim) # 这里用偏置长度做 fallback 最安全
        if bias is not None:
            output = output[:, :orig_N] + bias
        else:
            # 如果碰巧遇到没有 bias 的层，通常它本身的 N 已经是标准大小，但也做一个安全切片
            output = output[:, :orig_N] if orig_N != N_padded else output

        output_shape = (*input_shape[:-1], -1)
        return output.view(output_shape)

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backend == Mxfp8LinearBackend.EMULATION:
            return self._apply_emulation(input, weight, weight_scale, out_dtype, bias)

        assert self.backend == Mxfp8LinearBackend.FLASHINFER_CUTLASS
        return self._apply_flashinfer_cutlass(
            input, weight, weight_scale, out_dtype, bias
        )
