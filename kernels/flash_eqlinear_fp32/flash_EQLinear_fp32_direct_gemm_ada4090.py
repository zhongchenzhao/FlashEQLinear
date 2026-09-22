#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct structured-GEMM FP32 EQLinear path for Ada4090.

This path uses the original-input fused CUDA kernels in a standalone module so
it can be benchmarked independently from the legacy `cuda_extension` route.
"""

import math
from importlib import import_module

import torch
from torch import nn

try:
    _direct_backend = import_module(
        ".flash_EQLinear_cuda_direct_gemm_fp32_ada4090", __package__
    ) if __package__ else import_module("flash_EQLinear_cuda_direct_gemm_fp32_ada4090")
except ModuleNotFoundError as error:
    if error.name not in {
        f"{__package__}.flash_EQLinear_cuda_direct_gemm_fp32_ada4090",
        "flash_EQLinear_cuda_direct_gemm_fp32_ada4090",
    }:
        raise
    try:
        _direct_backend = import_module("flash_EQLinear_cuda_direct_gemm_fp32_ada4090")
    except ModuleNotFoundError as fallback_error:
        if fallback_error.name != "flash_EQLinear_cuda_direct_gemm_fp32_ada4090":
            raise
        raise ImportError(
            "The FP32 CUDA extension is not built. Run python kernels/setup.py from the repository root."
        ) from fallback_error

if __package__:
    from .flash_EQLinear_python_ref import spatial_weight_to_freq_weight
else:
    from flash_EQLinear_python_ref import spatial_weight_to_freq_weight


class FlashEQLDirectGemmAda4090Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if not x.is_cuda or weight.device != x.device:
            raise RuntimeError("Input and weights must be on the same CUDA device.")
        if bias is not None and bias.device != x.device:
            raise RuntimeError("Bias must be on the same CUDA device as the input.")
        ctx.has_bias = bias is not None

        if ctx.has_bias:
            bias = bias.contiguous()
            ctx.save_for_backward(x, weight, bias)
            return _direct_backend.flash_eq_linear_forward_direct_gemm_bias_ada4090(
                x, weight, bias.view(-1)
            )

        ctx.save_for_backward(x, weight)
        return _direct_backend.flash_eq_linear_forward_direct_gemm_ada4090(x, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()

        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
            grad_x, grad_weight, grad_bias = (
                _direct_backend.flash_eq_linear_backward_direct_gemm_bias_ada4090(
                    grad_output, x, weight
                )
            )
            return grad_x, grad_weight, grad_bias

        x, weight = ctx.saved_tensors
        grad_x, grad_weight = _direct_backend.flash_eq_linear_backward_direct_gemm_ada4090(
            grad_output, x, weight
        )
        return grad_x, grad_weight, None


class CudaFlashEQLinearDirectGemmAda4090(nn.Module):
    """FP32 direct CUDA implementation for four-element cyclic groups.

    ``math_mode`` is retained for API compatibility; both accepted values
    currently dispatch to the same FP32 CUDA kernels.
    """

    def __init__(
        self,
        inNum: int,
        outNum: int,
        tranNum: int = 4,
        bias: bool = False,
        eqlinear_weights: torch.Tensor | None = None,
        eqlinear_bias: torch.Tensor | None = None,
        math_mode: str = "tf32",
    ) -> None:
        super().__init__()
        if tranNum != 4:
            raise ValueError(f"Only tranNum=4 is supported, got {tranNum}")
        if math_mode not in {"tf32", "strict_fp32"}:
            raise ValueError(f"Unsupported math_mode={math_mode!r}. Expected 'tf32' or 'strict_fp32'.")

        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.use_bias = bias
        self.math_mode = math_mode

        if eqlinear_weights is not None:
            freq = spatial_weight_to_freq_weight(eqlinear_weights)
            self.weights = nn.Parameter(freq)
        else:
            self.weights = nn.Parameter(torch.empty(outNum, inNum, tranNum))
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if bias:
            if eqlinear_bias is not None:
                # EQLinearInter uses c with shape [outNum, 1]. Keep the same
                # parameter contract so its gradient also has shape [outNum, 1].
                eqlinear_bias = eqlinear_bias.detach().contiguous().view(outNum, 1)
                self.c = nn.Parameter(eqlinear_bias)
            else:
                self.c = nn.Parameter(torch.empty(outNum, 1))
                nn.init.zeros_(self.c)
        else:
            self.register_parameter("c", None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("CudaFlashEQLinearDirectGemmAda4090 requires CUDA tensors.")
        if x.dtype != torch.float32:
            raise RuntimeError(
                f"CudaFlashEQLinearDirectGemmAda4090 expects float32 input, got {x.dtype}."
            )
        if not x.is_contiguous():
            x = x.contiguous()

        orig_dim = x.dim()
        if orig_dim == 4:
            bsz, height, width, _ = x.shape
            x_4d = x.view(bsz, height * width, self.inNum, self.tranNum)
        elif orig_dim == 3:
            bsz, seq_len, _ = x.shape
            x_4d = x.view(bsz, seq_len, self.inNum, self.tranNum)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {orig_dim}D")

        if x_4d.size(2) != self.inNum or x_4d.size(3) != self.tranNum:
            raise RuntimeError(
                f"Input shape mismatch: expected [..., {self.inNum}, {self.tranNum}], "
                f"got {tuple(x_4d.shape)}."
            )

        weight = self.weights if self.weights.dtype == x_4d.dtype else self.weights.to(dtype=x_4d.dtype)

        bias = self.c
        if bias is not None:
            if bias.numel() != self.outNum:
                raise RuntimeError(
                    f"Bias shape mismatch: expected {self.outNum} elements "
                    f"with shape [{self.outNum}, 1], got {tuple(bias.shape)}."
                )
            bias = bias.view(self.outNum, 1)
            if bias.dtype != x_4d.dtype:
                bias = bias.to(dtype=x_4d.dtype)

        y = FlashEQLDirectGemmAda4090Function.apply(x_4d, weight, bias)
        y = y.reshape(y.size(0), y.size(1), -1)

        if orig_dim == 4:
            return y.view(bsz, height, width, -1)
        return y.view(bsz, seq_len, -1)
