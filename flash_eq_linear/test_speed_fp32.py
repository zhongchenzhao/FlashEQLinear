#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
# Support both direct execution and python -m from the repository root.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flash_eq_linear.flash_eq_linear_torch import Flash_EQ_linear_Torch
from kernels.flash_eqlinear_fp32.flash_EQLinear_fp32_direct_gemm_ada4090 import CudaFlashEQLinearDirectGemmAda4090


# ==========================================
# 0. 常量与工具函数 (集中管理,避免重复定义)
# ==========================================
# 4x4 DFT 变换矩阵 (tranNum=4 专用)
_DFT_TRANSFORM_LIGHT = torch.tensor(
    [[1, 1, 1, 1],
     [1, 0, -1, 0],
     [1, -1, 1, -1],
     [0, -1, 0, 1]],
    dtype=torch.float32,
)


def _get_dft(device, dtype):
    """按需把 DFT 矩阵搬到目标 device/dtype (不会重复分配,只在首次访问时搬运)。"""
    return _DFT_TRANSFORM_LIGHT.to(device=device, dtype=dtype)


def spatial_to_freq_weight(spatial_weight: torch.Tensor) -> torch.Tensor:
    """空间域权重 -> 频域权重。
    spatial_weight: [outNum, inNum, tranNum]
    return:         [outNum, inNum, tranNum]  (频域)
    """
    dft = _get_dft(spatial_weight.device, spatial_weight.dtype)
    freq = torch.roll(torch.flip(spatial_weight, dims=(-1,)), shifts=1, dims=(-1,))
    freq = torch.einsum('kg,dcg->dck', dft, freq)
    return freq.contiguous()


def freq_grad_to_spatial(dW_freq: torch.Tensor) -> torch.Tensor:
    """spatial_to_freq_weight 的梯度反映射 (链式法则)。"""
    dft = _get_dft(dW_freq.device, dW_freq.dtype)
    # 前向 'kg,dcg->dck' 的反向对 g 维展开: 'kg,dck->dcg'
    dW_rev = torch.einsum('kg,dck->dcg', dft, dW_freq)
    # flip+roll 是自逆的
    return torch.roll(torch.flip(dW_rev, dims=(-1,)), shifts=1, dims=(-1,))




# ==========================================
# 2. 模型定义
# ==========================================
class EQ_linear_inter(nn.Module):
    """Naive 参考实现 (基于循环移位 + F.linear)。"""

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
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
        T, O, I = self.tranNum, self.outNum, self.inNum
        # 用 torch.roll 替换原来的 slice+cat,语义等价但更清晰
        # 原: torch.cat([w[..., -i:], w[..., :-i]], dim=3)  ==  torch.roll(w, shifts=i, dims=-1)
        w = self.weights  # [O, 1, I, T]
        tempW = torch.cat([torch.roll(w, shifts=i, dims=-1) for i in range(T)], dim=1)  # [O, T, I, T]
        weight = tempW.reshape(O * T, I * T).to(dtype=input.dtype)

        bias = None
        if self.use_bias:
            bias = self.c.repeat(1, T).reshape(-1).to(dtype=input.dtype)
        return F.linear(input, weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class EQLinearInter(nn.Module):
    """
    输入: (..., inNum * tranNum)
    输出: (..., outNum * tranNum)
    存储基权重 (outNum, 1, inNum, tranNum),前向时按块循环展开为
    (outNum*tranNum, inNum*tranNum) 的等价线性矩阵。
    """

    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super().__init__()
        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.use_bias = bias
        self.iniScale = iniScale  # 保留 API 兼容,不使用

        # 保留原 4D 形状以兼容旧 checkpoint
        self.weights = nn.Parameter(torch.empty(outNum, 1, inNum, tranNum))
        if bias:
            self.c = nn.Parameter(torch.empty(outNum, 1))
        else:
            self.register_parameter("c", None)

        # 预计算循环移位索引: shift_idx[s, t] = (t - s) mod T
        idx = torch.arange(tranNum)
        shift_idx = (idx.view(1, tranNum) - idx.view(tranNum, 1)) % tranNum
        self.register_buffer("shift_idx", shift_idx, persistent=False)

        # eval 缓存槽:不写入 state_dict
        self.register_buffer("_cached_weight", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        self.reset_parameters()

    def _expand_weight(self) -> torch.Tensor:
        # (O,1,I,T) -> (O,I,T) -> gather 成 (O,I,T_s,T_t) -> (O*T, I*T)
        w = self.weights.squeeze(1)
        rolled = w[:, :, self.shift_idx]                     # (O, I, T_s, T_t)
        return rolled.permute(0, 2, 1, 3).reshape(self.outNum * self.tranNum, self.inNum * self.tranNum)

    def _expand_bias(self):
        if not self.use_bias:
            return None
        return self.c.squeeze(-1).repeat_interleave(self.tranNum)


    def forward(self, x):
        if self.training:
            weight = self._expand_weight()
            bias = self._expand_bias()
        else:
            if self._cached_weight is None:
                self._cached_weight = self._expand_weight().detach()
                if self.use_bias:
                    self._cached_bias = self._expand_bias().detach()
            weight = self._cached_weight
            bias = self._cached_bias if self.use_bias else None
        return F.linear(x, weight, bias)

    def train(self, mode: bool = True):
        if mode:
            self._cached_weight = None
            self._cached_bias = None
        return super().train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.use_bias:
            fan_in = self.inNum * self.tranNum
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

    def extra_repr(self):
        return (f"inNum={self.inNum}, outNum={self.outNum}, "
                f"tranNum={self.tranNum}, bias={self.use_bias}")


class Standard_linear_inter(nn.Module):
    """标准 Linear (用于速度对比基线)。"""

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
            self.register_parameter('c', None)
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




# ==========================================
# 3. 辅助函数
# ==========================================
def print_error(name, x_test, x_ref, epsilon=1e-5):
    diff_abs = (x_test - x_ref).abs()
    diff_rel = diff_abs / (x_ref.abs() + epsilon)
    print(f"[{name}] | Abs Error: Max {diff_abs.max().item():.2e}, "
          f"Mean {diff_abs.mean().item():.2e} | "
          f"Rel Error: Mean {diff_rel.mean().item():.2e}")


def _cuda_time(fn, iters, warmup=5):
    """用 CUDA Event 精确计时 (返回 ms/iter)。比 time.time()+sync 更准。"""
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
    return start.elapsed_time(end) / iters  # ms


def _cuda_time_backward(layer, x, grad_y, iters, warmup=5):
    """Time backward in ms/iter, resetting gradients before each iteration."""
    y = layer(x)

    def backward_once():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        y.backward(grad_y, retain_graph=True)

    try:
        return _cuda_time(backward_once, iters, warmup)
    finally:
        layer.zero_grad(set_to_none=True)
        x.grad = None


# ==========================================
# 4. 精度与梯度对齐测试
# ==========================================
def test_precision(device):
    print("\n" + "=" * 50)
    print("1. Precision Test (Forward & backward)".center(50))
    print("=" * 50)

    B, L, C_in, C_out, T = 2, 56 * 56, 64, 64, 4

    x_naive = torch.randn(B, L, C_in * T, device=device, dtype=torch.float32, requires_grad=True)
    x_torch = x_naive.detach().reshape(B, L, C_in, T).clone().requires_grad_(True)
    x_cuda = x_naive.clone().detach().requires_grad_(True)

    naive_model = EQ_linear_inter(C_in, C_out, tranNum=T, bias=True).to(device)
    spatial_weight = naive_model.weights[:, 0].detach().contiguous()
    bias = naive_model.c.detach().contiguous()

    torch_model = Flash_EQ_linear_Torch(
        C_in, C_out, eqlinear_weight=spatial_weight, tranNum=T, bias=True, device=device
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        torch_model.c.copy_(bias)

    cuda_model = CudaFlashEQLinearDirectGemmAda4090(
        C_in, C_out, eqlinear_weights=spatial_weight, tranNum=T, bias=True, eqlinear_bias=bias
    ).to(device)

    print(f"torch_model weight dtype: {torch_model.W.dtype}, cuda_model weight dtype: {cuda_model.weights.dtype}")

    # --- 前向 ---
    y_naive = naive_model(x_naive)
    y_torch = torch_model(x_torch).reshape(B, L, C_out * T)
    y_cuda = cuda_model(x_cuda)
    print_error("Torch Forward Output (Y)", y_torch, y_naive)
    print_error("CUDA Forward Output (Y)", y_cuda, y_naive)

    # --- 反向 ---
    grad_y = torch.randn_like(y_naive)
    y_naive.backward(grad_y)
    y_torch.backward(grad_y)
    y_cuda.backward(grad_y)

    print_error("Torch Backward Input Grad (dX)", x_torch.grad.reshape_as(x_naive), x_naive.grad)
    print_error("CUDA Backward Input Grad (dX)", x_cuda.grad, x_naive.grad)

    # Map frequency-domain weight gradients back to the naive spatial weights.
    torch_spatial_grad = freq_grad_to_spatial(torch_model.W.grad)
    cuda_spatial_grad = freq_grad_to_spatial(cuda_model.weights.grad)
    naive_spatial_grad = naive_model.weights.grad[:, 0].contiguous()
    print_error("Torch Backward Weight Grad (dW spatial)", torch_spatial_grad, naive_spatial_grad)
    print_error("CUDA Backward Weight Grad (dW spatial)", cuda_spatial_grad, naive_spatial_grad)

    print_error("Torch Backward Bias Grad (db)", torch_model.c.grad, naive_model.c.grad)
    print_error("CUDA Backward Bias Grad (db)", cuda_model.c.grad, naive_model.c.grad)

# ==========================================
# 5. 前后向速度 Benchmark
# ==========================================
def benchmark(device):
    header = (
        f"{'Shape (B,L,C)':<18} | "
        f"{'NonEQ Fwd':<10} | {'Naive Fwd':<10} | {'Torch Fwd':<10} | {'CUDA Fwd':<10} | "
        f"{'Fwd C/NonEQ':<12} | {'Fwd C/Naive':<12} | {'Fwd C/Torch':<12} | "
        f"{'NonEQ Bwd':<10} | {'Naive Bwd':<10} | {'Torch Bwd':<10} | {'CUDA Bwd':<10} | "
        f"{'Bwd C/NonEQ':<12} | {'Bwd C/Naive':<12} | {'Bwd C/Torch':<12}"
    )
    print("=" * len(header))
    print("2. FP32 Speed Test (Forward & Backward)".center(len(header)))
    print("=" * len(header))
    print("Torch method: Flash_EQ_linear_Torch (FP32)")
    print("Times: ms/iter; C/X = X time / CUDA time (>1 means CUDA is faster).")
    print(header)
    print("-" * len(header))

    T = 4
    iters = 200

    for B in [32]:
        for L in [128, 256, 512, 1024, 4096]:
        # for L in [4096]:
            for C_in in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                C_out = C_in

                # ---------------- 准备数据 ----------------
                x_base = torch.randn(
                    B, L, C_in * T,
                    device=device,
                    dtype=torch.float32
                )

                x_noneq = x_base.clone().detach().requires_grad_(True)
                x_naive = x_base.clone().detach().requires_grad_(True)
                x_torch = x_base.detach().reshape(B, L, C_in, T).clone().requires_grad_(True)
                x_cuda = x_base.clone().detach().requires_grad_(True)

                grad_y = torch.randn(
                    B, L, C_out * T,
                    device=device,
                    dtype=torch.float32
                )
                grad_y_torch = grad_y.reshape(B, L, C_out, T)

                noneq_linear = Standard_linear_inter(C_in, C_out, tranNum=T, bias=False ).to(device)

                naive = EQLinearInter( C_in, C_out, tranNum=T, bias=False ).to(device)
                spatial_weight = naive.weights[:, 0].detach().contiguous()

                torch_layer = Flash_EQ_linear_Torch(
                    C_in, C_out, eqlinear_weight=spatial_weight, tranNum=T, bias=False, device=device
                ).to(device=device, dtype=torch.float32)

                cuda_layer = CudaFlashEQLinearDirectGemmAda4090(
                    C_in, C_out, eqlinear_weights=spatial_weight, tranNum=T, bias=False ).to(device)

                # ---------------- 前向测试 ----------------
                with torch.no_grad():
                    t_fwd_noneq = _cuda_time(
                        lambda noneq_linear=noneq_linear, x_noneq=x_noneq: noneq_linear(x_noneq),
                        iters
                    )

                    t_fwd_naive = _cuda_time(
                        lambda naive=naive, x_naive=x_naive: naive(x_naive),
                        iters
                    )

                    t_fwd_torch = _cuda_time(
                        lambda torch_layer=torch_layer, x_torch=x_torch: torch_layer(x_torch),
                        iters
                    )

                    t_fwd_cuda = _cuda_time(
                        lambda cuda_layer=cuda_layer, x_cuda=x_cuda: cuda_layer(x_cuda),
                        iters
                    )

                # ---------------- 反向测试 ----------------
                # Retain one graph at a time and reset gradients every iteration.
                t_bwd_noneq = _cuda_time_backward(noneq_linear, x_noneq, grad_y, iters)
                t_bwd_naive = _cuda_time_backward(naive, x_naive, grad_y, iters)
                t_bwd_torch = _cuda_time_backward(torch_layer, x_torch, grad_y_torch, iters)
                t_bwd_cuda = _cuda_time_backward(cuda_layer, x_cuda, grad_y, iters)

                # ---------------- 加速倍率 ----------------
                sp_fwd_cuda_vs_noneq = t_fwd_noneq / t_fwd_cuda
                sp_fwd_cuda_vs_naive = t_fwd_naive / t_fwd_cuda
                sp_fwd_cuda_vs_torch = t_fwd_torch / t_fwd_cuda

                sp_bwd_cuda_vs_noneq = t_bwd_noneq / t_bwd_cuda
                sp_bwd_cuda_vs_naive = t_bwd_naive / t_bwd_cuda
                sp_bwd_cuda_vs_torch = t_bwd_torch / t_bwd_cuda

                shape_str = f"({B},{L},{C_in})"

                print(
                    f"{shape_str:<18} | "
                    f"{t_fwd_noneq:7.2f} ms | "
                    f"{t_fwd_naive:7.2f} ms | "
                    f"{t_fwd_torch:7.2f} ms | "
                    f"{t_fwd_cuda:7.2f} ms | "
                    f"{sp_fwd_cuda_vs_noneq:10.2f} x | "
                    f"{sp_fwd_cuda_vs_naive:10.2f} x | "
                    f"{sp_fwd_cuda_vs_torch:10.2f} x | "
                    f"{t_bwd_noneq:7.2f} ms | "
                    f"{t_bwd_naive:7.2f} ms | "
                    f"{t_bwd_torch:7.2f} ms | "
                    f"{t_bwd_cuda:7.2f} ms | "
                    f"{sp_bwd_cuda_vs_noneq:10.2f} x | "
                    f"{sp_bwd_cuda_vs_naive:10.2f} x | "
                    f"{sp_bwd_cuda_vs_torch:10.2f} x"
                )

                # ---------------- 释放资源 ----------------
                del (
                    x_base,
                    x_noneq,
                    x_naive,
                    x_torch,
                    x_cuda,
                    grad_y,
                    grad_y_torch,
                    spatial_weight,
                    noneq_linear,
                    naive,
                    torch_layer,
                    cuda_layer,
                )
                torch.cuda.empty_cache()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-enabled PyTorch installation and GPU.")
    device = torch.device("cuda")
    print(f"device: {device}")

    test_precision(device)
    benchmark(device)
