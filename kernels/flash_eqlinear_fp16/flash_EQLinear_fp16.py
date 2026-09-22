#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import math
from importlib import import_module
import os
import torch
from torch import nn
import torch.nn.functional as F


def _try_import(module_name):
    names = ([f"{__package__}.{module_name}"] if __package__ else []) + [module_name]
    for name in names:
        try:
            return import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    return None


_fp16_native_backend = _try_import("flash_eq_cutlass_fp16")
_fp32_cuda_backend = _try_import("flash_EQLinear_cuda_fp32")
_cuda_cublas_backend = _try_import("flash_EQLinear_cuda_cublas_fp16_ada4090")

if _cuda_cublas_backend is not None:
    _FLASH_EQ_BACKEND = "cuda_hybrid"
elif _fp16_native_backend is not None:
    _FLASH_EQ_BACKEND = "fp16_native"
elif _fp32_cuda_backend is not None:
    _FLASH_EQ_BACKEND = "fp32_cuda_fallback"
else:
    _FLASH_EQ_BACKEND = "torch_fallback"


# ==========================================
# 0. Constants and helpers
# ==========================================
_DFT_TRANSFORM_LIGHT = torch.tensor(
    [[1, 1, 1, 1],
     [1, 0, -1, 0],
     [1, -1, 1, -1],
     [0, -1, 0, 1]],
    dtype=torch.float32,
)


def _get_dft(device, dtype):
    return _DFT_TRANSFORM_LIGHT.to(device=device, dtype=dtype)


def spatial_to_freq_weight(spatial_weight: torch.Tensor) -> torch.Tensor:
    dft = _get_dft(spatial_weight.device, spatial_weight.dtype)
    freq = torch.roll(torch.flip(spatial_weight, dims=(-1,)), shifts=1, dims=(-1,))
    freq = torch.einsum("kg,dcg->dck", dft, freq)
    return freq.contiguous()


def freq_grad_to_spatial(dW_freq: torch.Tensor) -> torch.Tensor:
    dft = _get_dft(dW_freq.device, dW_freq.dtype)
    dW_rev = torch.einsum("kg,dck->dcg", dft, dW_freq)
    return torch.roll(torch.flip(dW_rev, dims=(-1,)), shifts=1, dims=(-1,))


def _flash_eq_forward_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    weight_float = weight.float()

    x0, x1, x2, x3 = x_float.unbind(dim=-1)
    w0, w1, w2, w3 = weight_float.unbind(dim=-1)

    x_freq0 = x0 + x1 + x2 + x3
    x_freq1 = x0 - x2
    x_freq2 = x0 - x1 + x2 - x3
    x_freq3 = -x1 + x3

    y0 = torch.einsum("blc,dc->bld", x_freq0, w0)
    y1 = torch.einsum("blc,dc->bld", x_freq1, w1) - torch.einsum("blc,dc->bld", x_freq3, w3)
    y2 = torch.einsum("blc,dc->bld", x_freq2, w2)
    y3 = torch.einsum("blc,dc->bld", x_freq1, w3) + torch.einsum("blc,dc->bld", x_freq3, w1)

    out = torch.stack(
        [
            0.25 * y0 + 0.5 * y1 + 0.25 * y2,
            0.25 * y0 - 0.25 * y2 - 0.5 * y3,
            0.25 * y0 - 0.5 * y1 + 0.25 * y2,
            0.25 * y0 - 0.25 * y2 + 0.5 * y3,
        ],
        dim=-1,
    )
    return out.to(dtype=x.dtype)


# ==========================================
# 1. Autograd bridge
# ==========================================
class FlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        x:      [B, L, inNum, tranNum]
        weight: [outNum, inNum, tranNum]
        """
        ctx.save_for_backward(x, weight)

        if _FLASH_EQ_BACKEND == "fp16_native":
            return _fp16_native_backend.cutlass_flash_eq_forward(x, weight)

        if _FLASH_EQ_BACKEND == "fp32_cuda_fallback":
            y_float = _fp32_cuda_backend.flash_eq_linear_forward(x.float(), weight.float())
            return y_float.to(dtype=x.dtype)

        return _flash_eq_forward_reference(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        if _FLASH_EQ_BACKEND == "fp16_native" and hasattr(_fp16_native_backend, "cutlass_flash_eq_backward"):
            return _fp16_native_backend.cutlass_flash_eq_backward(grad_output, x, weight)

        if _FLASH_EQ_BACKEND == "fp32_cuda_fallback":
            grad_x, grad_weight = _fp32_cuda_backend.flash_eq_linear_backward(
                grad_output.float(), x.float(), weight.float()
            )
            return grad_x.to(dtype=x.dtype), grad_weight.to(dtype=weight.dtype)

        needs_x, needs_weight = ctx.needs_input_grad
        with torch.enable_grad():
            x_ref = x.detach().requires_grad_(needs_x)
            weight_ref = weight.detach().requires_grad_(needs_weight)
            y_ref = _flash_eq_forward_reference(x_ref, weight_ref)
            grad_x, grad_weight = torch.autograd.grad(
                y_ref,
                (x_ref, weight_ref),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        return grad_x, grad_weight


class HybridFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        if _cuda_cublas_backend is None:
            raise RuntimeError("backend='cuda_hybrid' requested but flash_EQLinear_cuda_cublas_fp16_ada4090 is unavailable")
        ctx.save_for_backward(x, weight)
        return _cuda_cublas_backend.flash_eq_linear_forward_hybrid_ada4090(x, weight, True)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_weight = _cuda_cublas_backend.flash_eq_linear_backward_hybrid_ada4090(
            grad_output, x, weight, True
        )
        return grad_x, grad_weight


class PointerNativeFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        if _cuda_cublas_backend is None:
            raise RuntimeError(
                "backend='cuda_pointer_native' requested but flash_EQLinear_cuda_cublas_fp16_ada4090 is unavailable"
            )
        ctx.save_for_backward(x, weight)
        return _cuda_cublas_backend.flash_eq_linear_forward_pointer_native_ada4090(x, weight, True)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_weight = _cuda_cublas_backend.flash_eq_linear_backward_pointer_native_ada4090(
            grad_output, x, weight, True
        )
        return grad_x, grad_weight


# ==========================================
# 2. Model definitions
# ==========================================
class EQ_linear_inter(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super().__init__()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.use_bias = bias
        self.weights = nn.Parameter(torch.empty(outNum, 1, inNum, tranNum))
        if bias:
            self.c = nn.Parameter(torch.empty(outNum, 1))
        else:
            self.register_parameter("c", None)
        self.reset_parameters()

    def forward(self, input):
        t, out_num, in_num = self.tranNum, self.outNum, self.inNum
        w = self.weights
        temp_w = torch.cat([torch.roll(w, shifts=i, dims=-1) for i in range(t)], dim=1)
        weight = temp_w.reshape(out_num * t, in_num * t).to(dtype=input.dtype)

        bias = None
        if self.use_bias:
            bias = self.c.repeat(1, t).reshape(-1).to(dtype=input.dtype)
        return F.linear(input, weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Standard_linear_inter(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super().__init__()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.use_bias = bias
        self.weights = nn.Parameter(torch.empty(outNum * tranNum, inNum * tranNum))
        if bias:
            self.c = nn.Parameter(torch.empty(outNum, 1))
        else:
            self.register_parameter("c", None)
        self.reset_parameters()

    def forward(self, input):
        bias = None
        if self.use_bias:
            bias = self.c.repeat(1, self.tranNum).reshape(-1).to(dtype=input.dtype)
        weight = self.weights if self.weights.dtype == input.dtype else self.weights.to(dtype=input.dtype)
        return F.linear(input, weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Cuda_Flash_EQ_linear(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, eqlinear_weights=None, backend="auto"):
        super().__init__()
        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.use_bias = bias
        self.backend = _FLASH_EQ_BACKEND if backend == "auto" else backend

        if eqlinear_weights is not None:
            freq = spatial_to_freq_weight(eqlinear_weights.to(dtype=torch.float16))
            self.weights = nn.Parameter(freq)
        else:
            self.weights = nn.Parameter(torch.empty(outNum, inNum, tranNum, dtype=torch.float16))
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if bias:
            self.c = nn.Parameter(torch.empty(outNum, 1, dtype=torch.float16))
            fan_in = inNum * tranNum
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)
        else:
            self.register_parameter("c", None)

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        orig_dim = x.dim()
        if orig_dim == 4:
            bsz, height, width, _ = x.shape
            x = x.view(bsz, height * width, self.inNum, self.tranNum)
        elif orig_dim == 3:
            bsz, seq_len, _ = x.shape
            x = x.view(bsz, seq_len, self.inNum, self.tranNum)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {orig_dim}D")

        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)
        weight = self.weights if self.weights.dtype == x.dtype else self.weights.to(dtype=x.dtype)
        if self.backend == "cuda_hybrid":
            y = HybridFlashEQLFunction.apply(x, weight)
        elif self.backend == "cuda_pointer_native":
            y = PointerNativeFlashEQLFunction.apply(x, weight)
        else:
            y = FlashEQLFunction.apply(x, weight)

        if self.use_bias:
            bias = self.c.repeat(1, self.tranNum).reshape(-1).to(dtype=y.dtype)
            y = y + bias

        if orig_dim == 4:
            y = y.view(bsz, height, width, -1)
        else:
            y = y.view(bsz, seq_len, -1)
        return y


# ==========================================
# 3. Utilities
# ==========================================
def print_error(name, x_test, x_ref, epsilon=1e-5):
    diff_abs = (x_test - x_ref).abs()
    diff_rel = diff_abs / (x_ref.abs() + epsilon)
    print(
        f"[{name}] | Abs Error: Max {diff_abs.max().item():.2e}, "
        f"Mean {diff_abs.mean().item():.2e} | "
        f"Rel Error: Mean {diff_rel.mean().item():.2e}"
    )


def _cuda_time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ==========================================
# 4. Precision check
# ==========================================
def test_precision(device):
    print("\n" + "=" * 50)
    print("1. Precision Test (Forward & backward)".center(50))
    print("=" * 50)

    bsz, seq_len, c_in, c_out, tran_num = 2, 56 * 56, 64, 64, 4

    x_naive = torch.randn(bsz, seq_len, c_in * tran_num, device=device, dtype=torch.float16, requires_grad=True)
    x_cuda = x_naive.clone().detach().requires_grad_(True)

    naive_model = EQ_linear_inter(c_in, c_out, tranNum=tran_num, bias=False).to(device)
    spatial_weight = naive_model.weights[:, 0].detach().contiguous()

    cuda_model = Cuda_Flash_EQ_linear(
        c_in, c_out, eqlinear_weights=spatial_weight, tranNum=tran_num, bias=False
    ).to(device).half()

    print(f"cuda_model weight dtype: {cuda_model.weights.dtype}")
    print(f"fp16 backend: {_FLASH_EQ_BACKEND}")

    y_naive = naive_model(x_naive)
    y_cuda = cuda_model(x_cuda)
    print_error("Forward Output (Y)", y_cuda, y_naive)

    grad_y = torch.randn_like(y_naive)
    y_naive.backward(grad_y)
    y_cuda.backward(grad_y)

    print_error("Backward Input Grad (dX)", x_cuda.grad, x_naive.grad)

    cuda_spatial_grad = freq_grad_to_spatial(cuda_model.weights.grad.float())
    naive_spatial_grad = naive_model.weights.grad[:, 0].contiguous()
    print_error("Backward Weight Grad (dW spatial)", cuda_spatial_grad, naive_spatial_grad)


# ==========================================
# 5. Benchmark
# ==========================================
def benchmark(device):
    print("=" * 80)
    print("2. Speed Test (Forward & backward)".center(80))
    print("=" * 80)
    print(
        f"{'Shape (B,L,C)':<18} | {'Naive Fwd':<10} | {'CUDA Fwd':<10} | {'Fwd Acc':<8} | "
        f"{'Naive Bwd':<10} | {'CUDA Bwd':<10} | {'Bwd Acc':<8}"
    )
    print("-" * 80)

    tran_num = 4
    iters = 200

    for bsz in [16, 32]:
        for seq_len in [128, 256, 512, 1024, 4096]:
            for c_in in [4, 8, 16, 32, 64, 128, 256, 512]:
                c_out = c_in

                x_naive = torch.randn(bsz, seq_len, c_in * tran_num, device=device, dtype=torch.float16, requires_grad=True)
                x_cuda = x_naive.clone().detach().requires_grad_(True)
                grad_y = torch.randn(bsz, seq_len, c_out * tran_num, device=device, dtype=torch.float16)

                naive = Standard_linear_inter(c_in, c_out, tranNum=tran_num, bias=False).to(device).half()
                cuda_layer = Cuda_Flash_EQ_linear(
                    c_in, c_out, eqlinear_weights=None, tranNum=tran_num, bias=False
                ).to(device).half()

                with torch.no_grad():
                    t_fwd_naive = _cuda_time(lambda naive=naive, x_naive=x_naive: naive(x_naive), iters)
                    t_fwd_cuda = _cuda_time(lambda cuda_layer=cuda_layer, x_cuda=x_cuda: cuda_layer(x_cuda), iters)

                yn = naive(x_naive)
                t_bwd_naive = _cuda_time(lambda grad_y=grad_y, yn=yn: yn.backward(grad_y, retain_graph=True), iters)

                yc = cuda_layer(x_cuda)
                t_bwd_cuda = _cuda_time(lambda grad_y=grad_y, yc=yc: yc.backward(grad_y, retain_graph=True), iters)

                sp_fwd = t_fwd_naive / t_fwd_cuda
                sp_bwd = t_bwd_naive / t_bwd_cuda
                shape_str = f"({bsz},{seq_len},{c_in})"
                print(
                    f"{shape_str:<18} | {t_fwd_naive:7.2f} ms | {t_fwd_cuda:7.2f} ms | "
                    f"{sp_fwd:6.1f} x | {t_bwd_naive:7.2f} ms | {t_bwd_cuda:7.2f} ms | "
                    f"{sp_bwd:6.1f} x"
                )

                del x_naive, x_cuda, grad_y, naive, cuda_layer, yn, yc
                torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_precision(device)
    if os.getenv("FLASH_EQ_ONLY_PRECISION") != "1":
        benchmark(device)
