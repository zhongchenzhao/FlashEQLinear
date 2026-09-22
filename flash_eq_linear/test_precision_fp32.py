#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare forward outputs and gradients with the PyTorch reference."""

import math
import torch
from torch import nn
import torch.nn.functional as F

# Support both direct execution and python -m from the repository root.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.flash_eqlinear_fp32.flash_EQLinear_fp32_direct_gemm_ada4090 import CudaFlashEQLinearDirectGemmAda4090 as Cuda_Flash_EQ_linear
from kernels.flash_eqlinear_fp32.flash_EQLinear_fp32_direct_gemm_ada4090 import FlashEQLDirectGemmAda4090Function

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
def print_error(x_test, x_ref, name='', epsilon=1e-12):
    diff_abs = (x_test - x_ref).abs()
    diff_rel = diff_abs / (x_ref.abs() + epsilon)
    print(f"[{name}] | Abs Error: Max {diff_abs.max().item():.2e}, "
          f"Mean {diff_abs.mean().item():.2e} | "
          f"Rel Error: Mean {diff_rel.mean().item():.2e}")
    diff = (x_test - x_ref).abs()
    rel = diff / (x_ref.abs() + epsilon)
    print(f"  rel max: {rel.max().item():.4e}", f" | rel mean: {rel.mean().item():.4e}",
          f" | rel p50: {torch.quantile(rel.float(), 0.50).item():.4e}",
          f" | rel p95: {torch.quantile(rel.float(), 0.95).item():.4e}",
          f" | rel p99: {torch.quantile(rel.float(), 0.99).item():.4e}",
          f"  ||  ref max: {x_ref.abs().max().item():.4e}", f" | ref mean: {x_ref.abs().mean().item():.4e}")


def check_error(name, test, ref, eps=1e-12):
    test = test.detach().float()
    ref = ref.detach().float()

    diff = (test - ref).abs()
    rel = diff / (ref.abs() + eps)

    ref_abs = ref.abs()
    masks = {
        "|ref|>1e-3": ref_abs > 1e-3,
        "|ref|>1e-2": ref_abs > 1e-2,
        "|ref|>1e-1": ref_abs > 1e-1,
    }

    print(f"\n[{name}]")
    print(f"  abs max       : {diff.max().item():.6e}")
    print(f"  abs mean      : {diff.mean().item():.6e}")
    print(f"  ref abs mean  : {ref_abs.mean().item():.6e}")
    print(f"  mean_abs/ref_mean_abs: {(diff.mean() / (ref_abs.mean() + eps)).item():.6e}")

    print(f"  rel mean      : {rel.mean().item():.6e}")
    print(f"  rel p50/p95/p99/max: "
          f"{torch.quantile(rel, 0.50).item():.6e} / "
          f"{torch.quantile(rel, 0.95).item():.6e} / "
          f"{torch.quantile(rel, 0.99).item():.6e} / "
          f"{rel.max().item():.6e}")

    for k, m in masks.items():
        if m.any():
            r = diff[m] / ref_abs[m]
            print(f"  rel masked {k} mean/p99/max: "
                  f"{r.mean().item():.6e} / "
                  f"{torch.quantile(r, 0.99).item():.6e} / "
                  f"{r.max().item():.6e}")




def compute_paper_metrics(
    test: torch.Tensor,
    ref: torch.Tensor,
    rel_mask_threshold: float = 1e-1,
    eps: float = 1e-12,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    use_fp64: bool = True,
):
    """
    Compute paper-friendly numerical accuracy metrics.

    Args:
        test, ref           : tested and reference tensors (any matching shape/dtype)
        rel_mask_threshold  : keep only elements with |ref| > threshold for p99 RelErr.
                              Use absolute threshold (e.g. 1e-1 for activations,
                              1e-3 for gradients) — adjust per tensor scale.
        eps                 : numerical floor for divisions
        rtol, atol          : thresholds for the boolean Match verdict.
                              FP32  recommended: rtol=1e-5, atol=1e-6
                              FP16  recommended: rtol=1e-3, atol=1e-3
        use_fp64            : compute metrics in float64 (recommended).
                              Set False if memory is tight on huge tensors.

    Returned metrics:
      Max AbsErr   = max(|test - ref|)
      Mean AbsErr  = mean(|test - ref|)              # raw MAE, has dimension
      NMAE         = mean(|test - ref|) / mean(|ref|)  # scale-invariant
      Rel-L2       = ||test - ref||_2 / ||ref||_2     # PRIMARY metric
      p99 RelErr   = p99(|test - ref| / |ref|) on |ref| > threshold  # ROBUST
      Match        = torch.allclose(test, ref, rtol, atol)            # VERDICT
      Cosine       = <test, ref> / (||test|| * ||ref||)               # aux only
    """
    # Keep originals for the `allclose` verdict (uses native dtype, conventional)
    test_orig, ref_orig = test.detach(), ref.detach()
    assert test_orig.shape == ref_orig.shape, (
        f"shape mismatch: {test_orig.shape} vs {ref_orig.shape}"
    )

    # Cast for metric computation -> avoid the metric itself being rounded
    metric_dtype = torch.float64 if use_fp64 else torch.float32
    test = test_orig.to(metric_dtype).flatten()
    ref  = ref_orig.to(metric_dtype).flatten()

    diff     = test - ref
    abs_diff = diff.abs()
    abs_ref  = ref.abs()

    # ---- core scalar reductions ----
    max_abs  = abs_diff.max()
    mean_abs = abs_diff.mean()
    nmae     = mean_abs / (abs_ref.mean() + eps)
    rel_l2   = diff.norm() / (ref.norm() + eps)


    # ---- p99 relative error, magnitude-masked ----
    mask = abs_ref > rel_mask_threshold
    if mask.any():
        rel_err_masked = abs_diff[mask] / (abs_ref[mask] + eps)
        p99_rel = torch.quantile(rel_err_masked, 0.99)
    else:
        p99_rel = torch.tensor(float("nan"), dtype=metric_dtype)

    # ---- allclose verdict (native dtype, conventional) ----
    # match = torch.allclose(test_orig.float(), ref_orig.float(), rtol=rtol, atol=atol)
    match = rel_l2.item() < rtol

    # ---- auxiliary (NOT for paper table; sanity check only) ----
    cosine = (test * ref).sum() / ((test.norm() + eps) * (ref.norm() + eps))

    return {
        "Max AbsErr":   max_abs.item(),
        "Mean AbsErr":  mean_abs.item(),
        "NMAE":         nmae.item(),
        "Rel-L2":       rel_l2.item(),
        "p99 RelErr":   p99_rel.item(),
        "Match":        bool(match),
        "Cosine":       cosine.item(),
        "numel":        test.numel(),
        "masked_numel": int(mask.sum().item()),
    }


def print_paper_metrics_table(metrics_dict):
    """
    Print only the statistics needed by the LaTeX table:
    Tensor | Dtype | Shape | Rel L2 | Max Abs | Mean Abs | p99 Rel (masked) | Match
    """
    print("\n" + "=" * 120)
    print("Flash EQ-Linear Precision Errors (fp32)".center(120))
    print("=" * 120)

    print(
        f"{'Tensor':<36} "
        f"{'Dtype':<8} "
        f"{'Shape':<34} "
        f"{'Rel L2':>12} "
        f"{'Max Abs':>12} "
        f"{'Mean Abs':>12} "
        f"{'p99 Rel':>12} "
        f"{'Match':>8}"
    )

    print("-" * 120)

    for name, m in metrics_dict.items():
        print(
            f"{m['Tensor']:<36} "
            f"{m['Dtype']:<8} "
            f"{m['Shape']:<34} "
            f"{m['Rel-L2']:>12.2e} "
            f"{m['Max AbsErr']:>12.2e} "
            f"{m['Mean AbsErr']:>12.2e} "
            f"{m['p99 RelErr']:>12.2e} "
            f"{str(m['Match']):>8}"
        )

    print("=" * 120)


# ==========================================
# 4. 精度与梯度对齐测试
# ==========================================
def test_precision(device):
    print("\n" + "=" * 50)
    print("1. Precision Test (Forward & backward)".center(50))
    print("=" * 50)

    B, L, C_in, C_out, T = 1, 4096, 64, 64, 4
    B, L, C_in, C_out, T = 1, 1024, 64, 64, 4

    x_naive = torch.randn(B, L, C_in * T, device=device, dtype=torch.float32, requires_grad=True)
    x_cuda = x_naive.clone().detach().requires_grad_(True)

    naive_model = EQ_linear_inter(C_in, C_out, tranNum=T, bias=True).to(device)
    spatial_weight = naive_model.weights[:, 0].detach().contiguous()
    bias = naive_model.c.detach().contiguous()

    cuda_model = Cuda_Flash_EQ_linear(
        C_in, C_out, eqlinear_weights=spatial_weight, tranNum=T, bias=True, eqlinear_bias=bias
    ).to(device)

    print(f"cuda_model weight dtype: {cuda_model.weights.dtype}")
    print(f"cuda_model bias dtype: {cuda_model.c.dtype}")

    # --- 前向 ---
    y_naive = naive_model(x_naive)
    y_cuda = cuda_model(x_cuda)
    print_error("Forward Output (Y)", y_cuda, y_naive)
    check_error("Check Forward Output (Y)", y_cuda, y_naive)

    # --- 反向 ---
    grad_y = torch.randn_like(y_naive)
    y_naive.backward(grad_y)
    y_cuda.backward(grad_y)

    print_error("Backward Input Grad (dX)", x_cuda.grad, x_naive.grad)
    check_error("Check Backward Input Grad (dX)", x_cuda.grad, x_naive.grad)

    # 把 CUDA 算出的频域权重梯度映射回空间域再对比
    cuda_spatial_grad = freq_grad_to_spatial(cuda_model.weights.grad)
    naive_spatial_grad = naive_model.weights.grad[:, 0].contiguous()
    print_error("Backward Weight Grad (dW spatial)", cuda_spatial_grad, naive_spatial_grad)
    check_error("Check Backward Weight Grad (dW spatial)", cuda_spatial_grad, naive_spatial_grad)


    # bias 梯度不是频域量，不能调用 freq_grad_to_spatial。
    # EQLinearInter.c 的语义是 [C_out, 1]，每个 c[o] 被 repeat 到 4 个 transform 分量，
    # 因此 db[o] = sum_{b,l,t} grad_y[b,l,o,t]。
    expected_db = grad_y.view(B, L, C_out, T).sum(dim=(0, 1, 3)).view(C_out, 1)

    print_error("Backward Bias Grad (db)", cuda_model.c.grad, naive_model.c.grad)
    check_error("Check Backward Bias Grad (db)", cuda_model.c.grad, naive_model.c.grad)
    print_error("Backward Bias Grad (db) vs expected sum", cuda_model.c.grad, expected_db)
    check_error("Check Backward Bias Grad (db) vs expected sum", cuda_model.c.grad, expected_db)

    metrics = {
        "Y": compute_paper_metrics(y_cuda, y_naive, rel_mask_threshold=1e-1),
        "dX": compute_paper_metrics(x_cuda.grad, x_naive.grad, rel_mask_threshold=1e-3),
        "dW": compute_paper_metrics(cuda_spatial_grad, naive_spatial_grad, rel_mask_threshold=1e-3),
        "db": compute_paper_metrics(cuda_model.c.grad, naive_model.c.grad, rel_mask_threshold=1e-3),
        "db_expected": compute_paper_metrics(cuda_model.c.grad, expected_db, rel_mask_threshold=1e-3),
    }
    metrics["Y"].update({"Tensor": "Forward output Y", "Dtype": "FP32", "Shape": str(tuple(y_cuda.shape))})
    metrics["dX"].update({"Tensor": "Backward input grad dX", "Dtype": "FP32", "Shape": str(tuple(x_cuda.grad.shape))})
    metrics["dW"].update({"Tensor": "Backward weight grad dW", "Dtype": "FP32", "Shape": str(tuple(cuda_spatial_grad.shape))})
    metrics["db"].update({"Tensor": "Backward Bias Grad db", "Dtype": "FP32", "Shape": str(tuple(cuda_model.c.grad.shape))})
    metrics["db_expected"].update({"Tensor": "db vs expected sum", "Dtype": "FP32", "Shape": str(tuple(expected_db.shape))})
    print_paper_metrics_table(metrics)




def torch_cuda_math_forward(x, w_freq, bias=None):
    """
    x:      [B, L, I, 4], float
    w_freq: [O, I, 4],   float
    bias:   None or [O, 1]/[O]. Same semantics as EQLinearInter.c:
            one scalar per output channel, repeated across T=4.
    return: [B, L, O, 4]
    """
    x = x.float()
    w = w_freq.float()

    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

    # same DFT-like transform as CUDA
    X0 = x0 + x1 + x2 + x3
    X1 = x0 - x2
    X2 = x0 - x1 + x2 - x3
    X3 = -x1 + x3

    W0 = w[..., 0]
    W1 = w[..., 1]
    W2 = w[..., 2]
    W3 = w[..., 3]

    A0 = torch.einsum("bli,oi->blo", X0, W0)
    A2 = torch.einsum("bli,oi->blo", X2, W2)

    # paired frequency branch
    A1 = torch.einsum("bli,oi->blo", X1, W1) - torch.einsum("bli,oi->blo", X3, W3)
    A3 = torch.einsum("bli,oi->blo", X1, W3) + torch.einsum("bli,oi->blo", X3, W1)

    Y0 = 0.25 * (A0 + A2) + 0.5 * A1
    Y1 = 0.25 * (A0 - A2) - 0.5 * A3
    Y2 = 0.25 * (A0 + A2) - 0.5 * A1
    Y3 = 0.25 * (A0 - A2) + 0.5 * A3

    y = torch.stack([Y0, Y1, Y2, Y3], dim=-1)
    if bias is not None:
        y = y + bias.view(1, 1, -1, 1).to(dtype=y.dtype)
    return y


def diagnose_cuda_vs_simulated_paper_metrics(
    device,
    B=2,
    L=1024,
    C_in=64,
    C_out=64,
    seed=0,
    rel_mask_threshold=1e-1,
):
    """
    Paper-style numerical accuracy test.

    Compare:
      CUDA kernel
      vs
      PyTorch simulated CUDA math reference

    Report:
      | Tensor | Rel-L2 | NMAE | p99 RelErr, |ref|>1e-1 | Max AbsErr | Cosine |
    """
    torch.manual_seed(seed)
    T = 4

    # -------------------------
    # 1. Prepare inputs
    # -------------------------
    x0 = torch.randn(B, L, C_in, T, device=device, dtype=torch.float32)

    spatial_w = torch.randn(
        C_out, C_in, T, device=device, dtype=torch.float32
    ) * 0.05

    w_freq0 = spatial_to_freq_weight(spatial_w).contiguous()
    bias0 = (torch.randn(C_out, 1, device=device, dtype=torch.float32) * 0.05).contiguous()

    grad_y0 = torch.randn(B, L, C_out, T, device=device, dtype=torch.float32)

    # -------------------------
    # 2. PyTorch simulated CUDA math reference
    # -------------------------
    x_ref = x0.detach().float().requires_grad_(True)

    # Important:
    # real CUDA uses weight, so reference should use rounded weight too.
    w_ref = w_freq0.detach().float().requires_grad_(True)
    b_ref = bias0.detach().float().requires_grad_(True)

    gy_ref = grad_y0.float()

    y_ref_fp32 = torch_cuda_math_forward(x_ref, w_ref, b_ref)
    y_ref_fp32.backward(gy_ref)

    # Simulate CUDA returned dtype / grad dtype
    y_ref = y_ref_fp32.float()
    dx_ref = x_ref.grad.float()
    dw_freq_ref = w_ref.grad.float()
    db_ref = b_ref.grad.float()

    # Independent closed-form reference for db:
    # bias [C_out, 1] is repeated over T, so sum over B, L, T.
    db_expected = grad_y0.float().sum(dim=(0, 1, 3)).view(C_out, 1)

    # -------------------------
    # 3. Real CUDA kernel
    # -------------------------
    x_cuda = x0.detach().clone().requires_grad_(True)
    w_cuda = w_freq0.detach().contiguous().requires_grad_(True)
    b_cuda = bias0.detach().contiguous().requires_grad_(True)

    y_cuda = FlashEQLDirectGemmAda4090Function.apply(x_cuda, w_cuda, b_cuda)
    y_cuda.backward(grad_y0)

    y_cuda_f = y_cuda.float()
    dx_cuda_f = x_cuda.grad.float()
    dw_freq_cuda_f = w_cuda.grad.float()
    db_cuda_f = b_cuda.grad.float()

    # -------------------------
    # 4. Spatial dW
    # -------------------------
    dw_spatial_cuda = freq_grad_to_spatial(dw_freq_cuda_f)
    dw_spatial_ref = freq_grad_to_spatial(dw_freq_ref)


    # -------------------------
    # 6. Paper-style summary table
    # -------------------------
    metrics = {
        "Y": compute_paper_metrics(
            y_cuda_f,
            y_ref,
            rel_mask_threshold=rel_mask_threshold,
        ),
        "dX": compute_paper_metrics(
            dx_cuda_f,
            dx_ref,
            rel_mask_threshold=rel_mask_threshold,
        ),
        "dW": compute_paper_metrics(
            dw_spatial_cuda,
            dw_spatial_ref,
            rel_mask_threshold=rel_mask_threshold,
        ),
        "db": compute_paper_metrics(
            db_cuda_f,
            db_ref,
            rel_mask_threshold=1e-3,
        ),
        "db_expected": compute_paper_metrics(
            db_cuda_f,
            db_expected,
            rel_mask_threshold=1e-3,
        ),
    }

    metrics["Y"].update({
        "Tensor": "Forward output Y",
        "Dtype": "FP32",
        "Shape": "B x N x D x T",
    })

    metrics["dX"].update({
        "Tensor": "Backward input grad dX",
        "Dtype": "FP32",
        "Shape": "B x N x C x T",
    })


    metrics["dW"].update({
        "Tensor": "Backward weight grad dW",
        "Dtype": "FP32",
        "Shape": "D x C x T",
    })

    metrics["db"].update({
        "Tensor": "Backward Bias Grad db",
        "Dtype": "FP32",
        "Shape": "D x 1",
    })

    metrics["db_expected"].update({
        "Tensor": "db vs expected sum",
        "Dtype": "FP32",
        "Shape": "D x 1",
    })

    # 验证 dw 的误差来自于 四舍五入的精度误差
    # {'DW_MaxAbs': 0.0002288818359375, 'DW_BoundMax': 1.5750430822372437, 'DW_RoundoffRatioMax': 0.0001531415618956089,
    # 'DW_RoundoffRatioP99': 9.403439617017284e-05, 'DW_RoundoffOK': True, 'DW_RoundoffPassRate': 1.0, 'DW_AccumK': 6272,
    # 'DW_GammaK': 0.0003739801408912939}
    # dw_roundoff = compute_dw_roundoff_bound_metrics(
    #     dw_spatial_cuda,
    #     dw_spatial_ref,
    #     x_cuda.detach(),
    #     grad_y0.detach(),
    # )
    # print(dw_roundoff)

    print_paper_metrics_table(metrics)

    # print("\n\n")
    # print("=" * 110)
    # # -------------------------
    # # 5. Detailed debug metrics
    # # -------------------------
    # check_error("Y: CUDA vs simulated_cuda_math", y_cuda_f, y_ref)
    # check_error("dX: CUDA vs simulated_cuda_math", dx_cuda_f, dx_ref)
    # check_error("dW_freq: CUDA vs simulated_cuda_math", dw_freq_cuda_f, dw_freq_ref)
    # check_error("dW_spatial: CUDA vs simulated_cuda_math", dw_spatial_cuda, dw_spatial_ref)


    return metrics




def exact_small_test(device):
    """
    用很小的、可精确表示的数字，检查公式/符号/索引有没有错
    比如输入里用了：0, 1, -1, 0.5, 0.25, 0.125

    它主要用来抓：

    transform 符号错
    tran index 顺序错
    W0/W1/W2/W3 搭配错
    Y0/Y1/Y2/Y3 组合公式错
    dX 反向链式法则错
    dW_freq 公式错

    如果这个测试不过，通常说明：

    不是 FP16 精度问题，而是公式/索引/符号真的有 bug

    如果这个测试过，说明：

    小规模、精确输入下，forward/backward 的数学关系是对的
    """
    B, L, C_in, C_out, T = 1, 3, 2, 2, 4

    vals = torch.tensor(
        [
            0.0, 1.0, -1.0, 0.5,
            2.0, -0.5, 0.25, -0.25,
            1.5, -1.5, 0.75, -0.75,
            0.125, -0.125, 0.0, 1.0,
            -2.0, 0.5, -0.5, 0.25,
            1.0, 1.0, -1.0, -1.0,
        ],
        device=device,
        dtype=torch.float32,
    )

    x = vals[:B * L * C_in * T].view(B, L, C_in, T).contiguous()

    spatial_w = torch.tensor(
        [
            [[1.0, 0.5, -1.0, 0.25],
             [0.5, -0.5, 1.0, -0.25]],

            [[-1.0, 0.25, 0.5, -0.5],
             [1.0, -1.0, 0.5, 0.25]],
        ],
        device=device,
        dtype=torch.float32,
    )

    w_freq = spatial_to_freq_weight(spatial_w).contiguous()

    grad_y = torch.ones(B, L, C_out, T, device=device, dtype=torch.float32)

    x_ref = x.float().requires_grad_(True)
    w_ref = w_freq.float().requires_grad_(True)
    y_ref = torch_cuda_math_forward(x_ref, w_ref)
    y_ref.backward(grad_y.float())

    x_cuda = x.detach().clone().requires_grad_(True)
    w_cuda = w_freq.detach().clone().requires_grad_(True)
    y_cuda = FlashEQLDirectGemmAda4090Function.apply(x_cuda, w_cuda, None)
    y_cuda.backward(grad_y)

    check_error("small exact Y", y_cuda.float(), y_ref.float())
    check_error("small exact dX", x_cuda.grad.float(), x_ref.grad.float())
    check_error("small exact dW_freq", w_cuda.grad.float(), w_ref.grad.float())




def repeat_stability_test(device, repeats=20):
    """
    检查的是：

    同一组输入，重复运行 backward，dW 是否每次一样

    它主要用来抓 CUDA 里的非确定性问题，比如：

    未初始化内存
    越界写
    数据竞争 race condition
    atomicAdd 顺序导致的不稳定
    shared memory 没清零
    block 间归约不稳定
    """
    B, L, C_in, C_out = 16, 4096, 64, 64
    T = 4
    torch.manual_seed(0)

    x = torch.randn(B, L, C_in, T, device=device, dtype=torch.float32)
    w = torch.randn(C_out, C_in, T, device=device, dtype=torch.float32)
    gy = torch.randn(B, L, C_out, T, device=device, dtype=torch.float32)

    grads = []

    for _ in range(repeats):
        x1 = x.detach().clone().requires_grad_(True)
        w1 = w.detach().clone().requires_grad_(True)
        y = FlashEQLDirectGemmAda4090Function.apply(x1, w1, None)
        y.backward(gy)
        grads.append(w1.grad.detach().float().clone())

    base = grads[0]
    for i, g in enumerate(grads[1:], 1):
        diff = (g - base).abs()
        print(i, "max diff:", diff.max().item(), "mean diff:", diff.mean().item())


def compute_dw_roundoff_bound_metrics(
    dw_test: torch.Tensor,
    dw_ref: torch.Tensor,
    x: torch.Tensor,
    grad_y: torch.Tensor,
    eps: float = 1e-30,
):
    """
    Verify whether dW error is consistent with FP32 accumulation roundoff.

    dW element is a reduction over K = B * L terms.
    FP32 accumulation bound:
        |error| <= gamma_K * sum_k |term_k|
        gamma_K = K*u / (1 - K*u), u = 2^-24
    """
    dw_test = dw_test.detach().float()
    dw_ref = dw_ref.detach().float()

    diff = (dw_test - dw_ref).abs()

    B, L, C_in, T = x.shape
    K = B * L

    u = 2.0 ** -24
    gamma_k = (K * u) / (1.0 - K * u)

    # x:      [B, L, C_in, T]
    # grad_y: [B, L, C_out, T]
    #
    # 对每个 [out, in, t]，估计 sum |grad_y * x|
    # 这里是保守 bound，用同一个 t 对齐。
    sum_abs_terms = torch.einsum(
        "blot,blit->oit",
        grad_y.detach().float().abs(),
        x.detach().float().abs(),
    )

    roundoff_bound = gamma_k * sum_abs_terms

    ratio = diff / (roundoff_bound + eps)

    return {
        "DW_MaxAbs": diff.max().item(),
        "DW_BoundMax": roundoff_bound.max().item(),
        "DW_RoundoffRatioMax": ratio.max().item(),
        "DW_RoundoffRatioP99": torch.quantile(ratio.flatten(), 0.99).item(),
        "DW_RoundoffOK": bool((ratio <= 1.0).float().mean().item() > 0.999),
        "DW_RoundoffPassRate": (ratio <= 1.0).float().mean().item(),
        "DW_AccumK": K,
        "DW_GammaK": gamma_k,
    }



if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-enabled PyTorch installation and GPU.")
    device = torch.device("cuda")
    print(f"device: {device}")

    test_precision(device)

    diagnose_cuda_vs_simulated_paper_metrics(device)



    # exact_small_test(device)
    # repeat_stability_test(device)
