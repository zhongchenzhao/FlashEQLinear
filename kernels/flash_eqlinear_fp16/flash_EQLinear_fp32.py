#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import math
from importlib import import_module

import torch
from torch import nn
import torch.nn.functional as F


def _try_import(module_name):
    names = ([f"{__package__}.{module_name}"] if __package__ else []) + [module_name]
    if __package__ and module_name == "flash_EQLinear_cuda_direct_gemm_fp32_ada4090":
        names.insert(0, f"{__package__.rsplit('.', 1)[0]}.flash_eqlinear_fp32.{module_name}")
    for name in names:
        try:
            return import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    return None


_cuda_extension_backend = _try_import("flash_EQLinear_cuda_fp32")
_cuda_cublas_backend = _try_import("flash_EQLinear_cuda_cublas_fp32_ada4090")
_cuda_direct_gemm_backend = _try_import("flash_EQLinear_cuda_direct_gemm_fp32_ada4090")


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
    """
    spatial_weight: [outNum, inNum, tranNum]
    return:         [outNum, inNum, tranNum] (freq domain)
    """
    dft = _get_dft(spatial_weight.device, spatial_weight.dtype)
    freq = torch.roll(torch.flip(spatial_weight, dims=(-1,)), shifts=1, dims=(-1,))
    freq = torch.einsum("kg,dcg->dck", dft, freq)
    return freq.contiguous()


def freq_grad_to_spatial(dW_freq: torch.Tensor) -> torch.Tensor:
    dft = _get_dft(dW_freq.device, dW_freq.dtype)
    dW_rev = torch.einsum("kg,dck->dcg", dft, dW_freq)
    return torch.roll(torch.flip(dW_rev, dims=(-1,)), shifts=1, dims=(-1,))


def input_spatial_to_freq4(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, C, 4]
    return: [B, L, C, 4] in frequency domain
    """
    x0, x1, x2, x3 = x.unbind(dim=-1)
    return torch.stack(
        [
            x0 + x1 + x2 + x3,
            x0 - x2,
            x0 - x1 + x2 - x3,
            -x1 + x3,
        ],
        dim=-1,
    )


def output_grad_spatial_to_freq4(grad_output: torch.Tensor) -> torch.Tensor:
    """
    grad_output: [B, L, D, 4] in spatial domain
    return:      [B, L, D, 4] in frequency domain
    """
    gy0, gy1, gy2, gy3 = grad_output.unbind(dim=-1)
    return torch.stack(
        [
            0.25 * (gy0 + gy1 + gy2 + gy3),
            0.5 * (gy0 - gy2),
            0.25 * (gy0 - gy1 + gy2 - gy3),
            0.5 * (gy3 - gy1),
        ],
        dim=-1,
    )


def output_freq_to_spatial4(z_freq: torch.Tensor) -> torch.Tensor:
    """
    z_freq: [B, L, D, 4]
    return: [B, L, D, 4] in spatial domain
    """
    z0, z1, z2, z3 = z_freq.unbind(dim=-1)
    return torch.stack(
        [
            0.25 * z0 + 0.5 * z1 + 0.25 * z2,
            0.25 * z0 - 0.25 * z2 - 0.5 * z3,
            0.25 * z0 - 0.5 * z1 + 0.25 * z2,
            0.25 * z0 - 0.25 * z2 + 0.5 * z3,
        ],
        dim=-1,
    )


def input_grad_freq_to_spatial4(grad_x_freq: torch.Tensor) -> torch.Tensor:
    """
    grad_x_freq: [B, L, C, 4]
    return:      [B, L, C, 4] in spatial domain
    """
    gf0, gf1, gf2, gf3 = grad_x_freq.unbind(dim=-1)
    return torch.stack(
        [
            gf0 + gf1 + gf2,
            gf0 - gf2 - gf3,
            gf0 - gf1 + gf2,
            gf0 - gf2 + gf3,
        ],
        dim=-1,
    )


def pack_freq_features(x_freq: torch.Tensor) -> torch.Tensor:
    """
    x_freq: [B, L, C, 4]
    return: [B, L, 4*C] with block layout [f0 | f1 | f2 | f3]
    """
    return torch.cat([x_freq[..., i] for i in range(4)], dim=-1)


def unpack_freq_features(x_freq_flat: torch.Tensor, channels: int) -> torch.Tensor:
    """
    x_freq_flat: [B, L, 4*C] with block layout [f0 | f1 | f2 | f3]
    return:      [B, L, C, 4]
    """
    chunks = [x_freq_flat[..., i * channels:(i + 1) * channels] for i in range(4)]
    return torch.stack(chunks, dim=-1)


def build_dense_weight_4x4(weight: torch.Tensor) -> torch.Tensor:
    """
    weight: [D, C, 4] in frequency domain
    return: [4*D, 4*C] dense block matrix for a single F.linear call
    """
    out_num, in_num, _ = weight.shape
    w0, w1, w2, w3 = weight.unbind(dim=-1)
    zeros = weight.new_zeros((out_num, in_num))

    row0 = torch.cat([w0, zeros, zeros, zeros], dim=1)
    row1 = torch.cat([zeros, w1, zeros, -w3], dim=1)
    row2 = torch.cat([zeros, zeros, w2, zeros], dim=1)
    row3 = torch.cat([zeros, w3, zeros, w1], dim=1)
    return torch.cat([row0, row1, row2, row3], dim=0)


def unpack_dense_weight_grad_4x4(grad_weight_dense: torch.Tensor, out_num: int, in_num: int) -> torch.Tensor:
    def block(row_idx, col_idx):
        return grad_weight_dense[
            row_idx * out_num:(row_idx + 1) * out_num,
            col_idx * in_num:(col_idx + 1) * in_num,
        ]

    dW0 = block(0, 0)
    dW1 = block(1, 1) + block(3, 3)
    dW2 = block(2, 2)
    dW3 = block(3, 1) - block(1, 3)
    return torch.stack([dW0, dW1, dW2, dW3], dim=-1)


# ==========================================
# 1. Autograd functions
# ==========================================
class ExtensionFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        if bias is None:
            return _cuda_extension_backend.flash_eq_linear_forward(x, weight)
        return _cuda_extension_backend.flash_eq_linear_forward_fused_bias(x, weight, bias.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_weight = _cuda_extension_backend.flash_eq_linear_backward(grad_output, x, weight)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=(0, 1, 3)).view(weight.size(0), 1)
        return grad_x, grad_weight, grad_bias


class HybridFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if _cuda_cublas_backend is None:
            raise RuntimeError("backend='cuda_hybrid' requested but flash_EQLinear_cuda_cublas_fp32_ada4090 is unavailable")
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        y = _cuda_cublas_backend.flash_eq_linear_forward_hybrid_ada4090(x, weight, True)
        if bias is not None:
            y = y + bias.view(1, 1, weight.size(0), 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_weight = _cuda_cublas_backend.flash_eq_linear_backward_hybrid_ada4090(
            grad_output, x, weight, True
        )
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=(0, 1, 3)).view(weight.size(0), 1)
        return grad_x, grad_weight, grad_bias


class PointerNativeFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if _cuda_cublas_backend is None:
            raise RuntimeError("backend='cuda_pointer_native' requested but flash_EQLinear_cuda_cublas_fp32_ada4090 is unavailable")
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        y = _cuda_cublas_backend.flash_eq_linear_forward_pointer_native_ada4090(x, weight, True)
        if bias is not None:
            y = y + bias.view(1, 1, weight.size(0), 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_weight = _cuda_cublas_backend.flash_eq_linear_backward_pointer_native_ada4090(
            grad_output, x, weight, True
        )
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.sum(dim=(0, 1, 3)).view(weight.size(0), 1)
        return grad_x, grad_weight, grad_bias


class DirectGemmFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        if _cuda_direct_gemm_backend is None:
            raise RuntimeError(
                "backend='cuda_direct_gemm' requested but flash_EQLinear_cuda_direct_gemm_fp32_ada4090 is unavailable"
            )
        ctx.save_for_backward(x, weight)
        return _cuda_direct_gemm_backend.flash_eq_linear_forward_direct_gemm_ada4090(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_weight = _cuda_direct_gemm_backend.flash_eq_linear_backward_direct_gemm_ada4090(
            grad_output, x, weight
        )
        return grad_x, grad_weight


class DenseLinearFlashEQLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        x:      [B, L, inNum, 4] in spatial domain
        weight: [outNum, inNum, 4] in frequency domain
        """
        bsz, seq_len, in_num, _ = x.shape
        out_num = weight.size(0)

        x_freq = input_spatial_to_freq4(x)
        x_freq_flat = pack_freq_features(x_freq)
        weight_dense = build_dense_weight_4x4(weight)

        z_freq_flat = F.linear(x_freq_flat, weight_dense)
        z_freq = unpack_freq_features(z_freq_flat, out_num)
        y_spatial = output_freq_to_spatial4(z_freq)

        ctx.save_for_backward(x_freq_flat, weight_dense)
        ctx.in_num = in_num
        ctx.out_num = out_num
        return y_spatial.reshape(bsz, seq_len, out_num * 4)

    @staticmethod
    def backward(ctx, grad_output):
        x_freq_flat, weight_dense = ctx.saved_tensors
        in_num = ctx.in_num
        out_num = ctx.out_num

        grad_output = grad_output.contiguous().view(grad_output.size(0), grad_output.size(1), out_num, 4)
        grad_z_freq = output_grad_spatial_to_freq4(grad_output)
        grad_z_freq_flat = pack_freq_features(grad_z_freq)

        grad_x_freq_flat = grad_z_freq_flat.matmul(weight_dense)
        grad_weight_dense = grad_z_freq_flat.reshape(-1, 4 * out_num).transpose(0, 1).matmul(
            x_freq_flat.reshape(-1, 4 * in_num)
        )

        grad_x_freq = unpack_freq_features(grad_x_freq_flat, in_num)
        grad_x_spatial = input_grad_freq_to_spatial4(grad_x_freq)
        grad_weight = unpack_dense_weight_grad_4x4(grad_weight_dense, out_num, in_num)
        return grad_x_spatial, grad_weight


# ==========================================
# 2. Model definitions
# ==========================================
class EQ_linear_inter(nn.Module):
    """Naive reference implementation based on cyclic shifts + F.linear."""

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
        tran_num, out_num, in_num = self.tranNum, self.outNum, self.inNum
        w = self.weights
        temp_w = torch.cat([torch.roll(w, shifts=i, dims=-1) for i in range(tran_num)], dim=1)
        weight = temp_w.reshape(out_num * tran_num, in_num * tran_num).to(dtype=input.dtype)

        bias = None
        if self.use_bias:
            bias = self.c.repeat(1, tran_num).reshape(-1).to(dtype=input.dtype)
        return F.linear(input, weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Standard_linear_inter(nn.Module):
    """Standard Linear baseline used for throughput comparison."""

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
    """Flash EQ Linear with selectable backends."""

    def __init__(self, inNum, outNum, tranNum=4, bias=True, eqlinear_weights=None, backend="cuda_extension"):
        super().__init__()
        if tranNum != 4:
            raise ValueError(f"Only tranNum=4 is supported, got {tranNum}")
        if backend not in {"dense_linear", "cuda_extension", "cuda_hybrid", "cuda_pointer_native", "cuda_direct_gemm"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.use_bias = bias
        self.backend = backend

        if eqlinear_weights is not None:
            freq = spatial_to_freq_weight(eqlinear_weights)
            self.weights = nn.Parameter(freq)
        else:
            self.weights = nn.Parameter(torch.empty(outNum, inNum, tranNum))
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if bias:
            self.c = nn.Parameter(torch.empty(outNum, 1))
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

        weight = self.weights if self.weights.dtype == x.dtype else self.weights.to(dtype=x.dtype)

        if self.backend == "dense_linear":
            y = DenseLinearFlashEQLFunction.apply(x, weight)
            if self.use_bias:
                bias = self.c.repeat(1, self.tranNum).reshape(-1).to(dtype=y.dtype)
                y = y + bias
        elif self.backend == "cuda_extension":
            if _cuda_extension_backend is None:
                raise RuntimeError("backend='cuda_extension' requested but flash_EQLinear_cuda_fp32 is unavailable")
            bias = None
            if self.use_bias:
                bias = self.c if self.c.dtype == x.dtype else self.c.to(dtype=x.dtype)
            y = ExtensionFlashEQLFunction.apply(x, weight, bias)
        elif self.backend == "cuda_hybrid":
            if _cuda_cublas_backend is None:
                raise RuntimeError("backend='cuda_hybrid' requested but flash_EQLinear_cuda_cublas_fp32_ada4090 is unavailable")
            bias = None
            if self.use_bias:
                bias = self.c if self.c.dtype == x.dtype else self.c.to(dtype=x.dtype)
            y = HybridFlashEQLFunction.apply(x, weight, bias)
        elif self.backend == "cuda_pointer_native":
            if _cuda_cublas_backend is None:
                raise RuntimeError("backend='cuda_pointer_native' requested but flash_EQLinear_cuda_cublas_fp32_ada4090 is unavailable")
            bias = None
            if self.use_bias:
                bias = self.c if self.c.dtype == x.dtype else self.c.to(dtype=x.dtype)
            y = PointerNativeFlashEQLFunction.apply(x, weight, bias)
        else:
            if _cuda_direct_gemm_backend is None:
                raise RuntimeError(
                    "backend='cuda_direct_gemm' requested but flash_EQLinear_cuda_direct_gemm_fp32_ada4090 is unavailable"
                )
            if self.use_bias:
                raise RuntimeError("backend='cuda_direct_gemm' currently only supports bias=False")
            y = DirectGemmFlashEQLFunction.apply(x, weight)

        if y.dim() == 4:
            y = y.reshape(y.size(0), y.size(1), -1)

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


def _run_model_pair(
    backend,
    x_ref,
    grad_y,
    c_in,
    c_out,
    tran_num,
    device,
    bias=False,
    as_4d=False,
    spatial_weight=None,
    bias_value=None,
):
    x_test = x_ref.clone().detach().requires_grad_(True)
    naive_model = EQ_linear_inter(c_in, c_out, tranNum=tran_num, bias=bias).to(device)
    if spatial_weight is None:
        spatial_weight = naive_model.weights[:, 0].detach().contiguous()
    else:
        naive_model.weights.data.copy_(spatial_weight.unsqueeze(1))

    test_model = Cuda_Flash_EQ_linear(
        c_in, c_out, eqlinear_weights=spatial_weight, tranNum=tran_num, bias=bias, backend=backend
    ).to(device)
    if bias and bias_value is not None:
        naive_model.c.data.copy_(bias_value)
        test_model.c.data.copy_(bias_value)
    elif bias and naive_model.c is not None:
        test_model.c.data.copy_(naive_model.c.data)

    if as_4d:
        side = int(math.isqrt(x_ref.size(1)))
        x_ref_input = x_ref.view(x_ref.size(0), side, side, -1)
        x_test_input = x_test.view(x_test.size(0), side, side, -1)
        grad_view = grad_y.view(grad_y.size(0), side, side, -1)
    else:
        x_ref_input = x_ref
        x_test_input = x_test
        grad_view = grad_y

    y_ref = naive_model(x_ref_input)
    y_test = test_model(x_test_input)
    y_ref.backward(grad_view)
    y_test.backward(grad_view)

    return {
        "y_ref": y_ref.detach(),
        "y_test": y_test.detach(),
        "dx_ref": x_ref.grad.detach(),
        "dx_test": x_test.grad.detach(),
        "dw_ref_spatial": naive_model.weights.grad[:, 0].detach().contiguous(),
        "dw_test_spatial": freq_grad_to_spatial(test_model.weights.grad.detach()),
    }


# ==========================================
# 4. Accuracy checks
# ==========================================
def test_precision(device):
    print("\n" + "=" * 50)
    print("1. Precision Test (Forward & backward)".center(50))
    print("=" * 50)

    bsz, seq_len, c_in, c_out, tran_num = 2, 56 * 56, 64, 64, 4
    x_ref = torch.randn(bsz, seq_len, c_in * tran_num, device=device, dtype=torch.float32, requires_grad=True)
    grad_y = torch.randn(bsz, seq_len, c_out * tran_num, device=device, dtype=torch.float32)

    dense_result = _run_model_pair("dense_linear", x_ref, grad_y, c_in, c_out, tran_num, device)
    print("backend: dense_linear")
    print_error("Forward Output (Y)", dense_result["y_test"], dense_result["y_ref"])
    print_error("Backward Input Grad (dX)", dense_result["dx_test"], dense_result["dx_ref"])
    print_error("Backward Weight Grad (dW spatial)", dense_result["dw_test_spatial"], dense_result["dw_ref_spatial"])

    if _cuda_extension_backend is not None:
        x_ext = torch.randn(bsz, seq_len, c_in * tran_num, device=device, dtype=torch.float32, requires_grad=True)
        grad_ext = torch.randn(bsz, seq_len, c_out * tran_num, device=device, dtype=torch.float32)
        ext_result = _run_model_pair("cuda_extension", x_ext, grad_ext, c_in, c_out, tran_num, device)
        print("backend: cuda_extension")
        print_error("Forward Output (Y)", ext_result["y_test"], ext_result["y_ref"])
        print_error("Backward Input Grad (dX)", ext_result["dx_test"], ext_result["dx_ref"])
        print_error("Backward Weight Grad (dW spatial)", ext_result["dw_test_spatial"], ext_result["dw_ref_spatial"])


def test_backend_consistency(device):
    print("\n" + "=" * 50)
    print("2. Backend Consistency".center(50))
    print("=" * 50)

    if _cuda_extension_backend is None:
        print("cuda_extension unavailable, skip backend consistency test")
        return

    for bias, as_4d in [(False, False), (True, False), (False, True), (True, True)]:
        bsz, seq_len, c_in, c_out, tran_num = 2, 64, 16, 16, 4
        x = torch.randn(bsz, seq_len, c_in * tran_num, device=device, dtype=torch.float32, requires_grad=True)
        grad_y = torch.randn(bsz, seq_len, c_out * tran_num, device=device, dtype=torch.float32)
        shared_weight = torch.empty(c_out, c_in, tran_num, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(shared_weight, a=math.sqrt(5))
        shared_bias = None
        if bias:
            shared_bias = torch.empty(c_out, 1, device=device, dtype=torch.float32)
            fan_in = c_in * tran_num
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(shared_bias, -bound, bound)

        dense = _run_model_pair(
            "dense_linear",
            x,
            grad_y,
            c_in,
            c_out,
            tran_num,
            device,
            bias=bias,
            as_4d=as_4d,
            spatial_weight=shared_weight,
            bias_value=shared_bias,
        )
        ext = _run_model_pair(
            "cuda_extension",
            x,
            grad_y,
            c_in,
            c_out,
            tran_num,
            device,
            bias=bias,
            as_4d=as_4d,
            spatial_weight=shared_weight,
            bias_value=shared_bias,
        )

        tag = f"bias={bias}, input={'4D' if as_4d else '3D'}"
        print(tag)
        print_error("Dense vs CUDA Forward", dense["y_test"], ext["y_test"])
        print_error("Dense vs CUDA dX", dense["dx_test"], ext["dx_test"])
        print_error("Dense vs CUDA dW spatial", dense["dw_test_spatial"], ext["dw_test_spatial"])


# ==========================================
# 5. Benchmark
# ==========================================
def benchmark(device):
    print("=" * 100)
    print("3. Speed Test (Forward & backward)".center(100))
    print("=" * 100)
    print(
        f"{'Shape (B,L,C)':<18} | {'Naive Fwd':<10} | {'Dense Fwd':<10} | {'CUDA Fwd':<10} | "
        f"{'Naive Bwd':<10} | {'Dense Bwd':<10} | {'CUDA Bwd':<10}"
    )
    print("-" * 100)

    tran_num = 4
    iters = 200

    for bsz in [32]:
        for seq_len in [128, 256, 512, 1024, 4096]:
            for c_in in [4, 16, 32, 64, 128, 256, 512]:
                c_out = c_in

                x_naive = torch.randn(bsz, seq_len, c_in * tran_num, device=device, dtype=torch.float32, requires_grad=True)
                x_dense = x_naive.clone().detach().requires_grad_(True)
                grad_y = torch.randn(bsz, seq_len, c_out * tran_num, device=device, dtype=torch.float32)

                naive = Standard_linear_inter(c_in, c_out, tranNum=tran_num, bias=False).to(device)
                dense_layer = Cuda_Flash_EQ_linear(
                    c_in, c_out, eqlinear_weights=None, tranNum=tran_num, bias=False, backend="dense_linear"
                ).to(device)

                with torch.no_grad():
                    t_fwd_naive = _cuda_time(lambda naive=naive, x_naive=x_naive: naive(x_naive), iters)
                    t_fwd_dense = _cuda_time(lambda dense_layer=dense_layer, x_dense=x_dense: dense_layer(x_dense), iters)

                yn = naive(x_naive)
                t_bwd_naive = _cuda_time(lambda grad_y=grad_y, yn=yn: yn.backward(grad_y, retain_graph=True), iters)
                yd = dense_layer(x_dense)
                t_bwd_dense = _cuda_time(lambda grad_y=grad_y, yd=yd: yd.backward(grad_y, retain_graph=True), iters)

                t_fwd_cuda = float("nan")
                t_bwd_cuda = float("nan")
                if _cuda_extension_backend is not None:
                    x_cuda = x_naive.clone().detach().requires_grad_(True)
                    cuda_layer = Cuda_Flash_EQ_linear(
                        c_in, c_out, eqlinear_weights=None, tranNum=tran_num, bias=False, backend="cuda_extension"
                    ).to(device)
                    with torch.no_grad():
                        t_fwd_cuda = _cuda_time(lambda cuda_layer=cuda_layer, x_cuda=x_cuda: cuda_layer(x_cuda), iters)
                    yc = cuda_layer(x_cuda)
                    t_bwd_cuda = _cuda_time(lambda grad_y=grad_y, yc=yc: yc.backward(grad_y, retain_graph=True), iters)
                    del x_cuda, cuda_layer, yc

                shape_str = f"({bsz},{seq_len},{c_in})"
                cuda_fwd_str = f"{t_fwd_cuda:7.2f} ms" if math.isfinite(t_fwd_cuda) else "   N/A   "
                cuda_bwd_str = f"{t_bwd_cuda:7.2f} ms" if math.isfinite(t_bwd_cuda) else "   N/A   "
                print(
                    f"{shape_str:<18} | {t_fwd_naive:7.2f} ms | {t_fwd_dense:7.2f} ms | {cuda_fwd_str:<10} | "
                    f"{t_bwd_naive:7.2f} ms | {t_bwd_dense:7.2f} ms | {cuda_bwd_str:<10}"
                )

                del x_naive, x_dense, grad_y, naive, dense_layer, yn, yd
                torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_precision(device)
    test_backend_consistency(device)
    benchmark(device)
