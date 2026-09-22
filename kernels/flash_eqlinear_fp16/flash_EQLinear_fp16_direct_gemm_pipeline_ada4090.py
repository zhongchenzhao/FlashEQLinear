#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Direct fp16 pipeline variants with routed autograd for Ada4090."""

import math
from importlib import import_module

import torch
from torch import nn

def _try_import(module_name):
    names = ([f"{__package__}.{module_name}"] if __package__ else []) + [module_name]
    for name in names:
        try:
            return import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    return None


_cute_backend = _try_import("flash_EQLinear_cuda_direct_gemm_fp16_cute_ada4090")
_mma_backend = _try_import("flash_EQLinear_cuda_direct_gemm_fp16_mma_ada4090")
_old_direct_backend = _try_import("flash_EQLinear_cuda_direct_gemm_fp16_ada4090")
_persistent_backend = _try_import("flash_EQLinear_cuda_direct_gemm_fp16_persistent_ada4090")
_persistent_ptx_backend = _try_import("flash_EQLinear_cuda_direct_gemm_fp16_persistent_ptx_ada4090")

if __package__:
    from .flash_EQLinear_python_ref import spatial_weight_to_freq_weight
else:
    from flash_EQLinear_python_ref import spatial_weight_to_freq_weight


VALID_BACKENDS = {
    "pure_cute_fused",
    "mixed_cute_fused",
    "pure_cutlass_direct_spatial",
    "pure_cutlass_spatial_epilogue",
    "pure_cutlass_spatial_fanout_epilogue",
    "pure_cutlass_no_yfreq_materialize",
    "pure_cutlass_staged",
    "pure_cutlass_staged_gauss",
    "pure_cutlass_staged_gauss_n64k64",
    "pure_cutlass_staged_n64k64",
    "pure_cutlass_staged_k64",
    "pure_cutlass_staged_gauss_k64",
    "pure_cutlass_staged_s4",
    "pure_cutlass_staged_gauss_s4",
    "pure_cutlass_staged_s2",
    "pure_cutlass_staged_gauss_s2",
    "pure_cutlass_staged_k64_s2",
    "pure_cutlass_staged_gauss_k64_s2",
    "pure_cutlass_staged_k64_s4",
    "pure_cutlass_staged_gauss_k64_s4",
    "pure_cutlass_staged_n64k64_s2",
    "pure_cutlass_staged_gauss_n64k64_s2",
    "pure_cutlass_staged_n64k64_s4",
    "pure_cutlass_staged_gauss_n64k64_s4",
    "pure_cutlass_staged_m64n128k64_s2",
    "pure_cutlass_staged_gauss_m64n128k64_s2",
    "pure_cutlass_staged_m64n128k64",
    "pure_cutlass_staged_gauss_m64n128k64",
    "pure_cutlass_staged_m64n128k64_s4",
    "pure_cutlass_staged_gauss_m64n128k64_s4",
    "pure_mma_pipe",
    "mixed_mma_pipe",
    "pure_mma_pipe_v2",
    "pure_mma_pipe_m128n128",
    "pure_mma_pipe_m128n64",
    "pure_mma_pipe_m128n64_db",
    "pure_mma_pipe_m128n64_w4",
    "pure_mma_pipe_m128n128_w2",
    "pure_mma_pipe_m128n64_gauss",
    "pure_mma_pipe_m128n64_w4_gauss",
    "pure_mma_pipe_m128n32",
    "pure_mma_pipe_m128n32_modc",
    "pure_mma_pipe_m64n32_modc",
    "pure_mma_pipe_m64n64",
    "pure_mma_pipe_m64n64_w4",
    "pure_mma_pipe_m64n256",
    "pure_mma_pipe_m128n64_x2d",
    "pure_mma_pipe_m128n64_x4d",
    "pure_mma_pipe_m64n64_x2d",
    "mixed_mma_pipe_v2",
    "pure_mma_pipe_cpasync_w",
    "pure_mma_pipe_db",
    "pure_mma_pipe_warpspec",
    "pure_mma_pipe_warpspec15",
    "pure_old_m32n32_smalld",
    "pure_old_m32n32_gauss_smalld",
    "pure_old_m32n64_smalld",
    "pure_old_m32n64_smalld_pad64",
    "pure_old_m32n64_gauss_smalld_pad64",
    "pure_old_m16n64_smalld",
    "pure_old_m16n32_smalld",
    "pure_old_m16n16_smalld",
    "pure_old_m64n64_smalld",
    "pure_old_m64n16_smalld",
    "pure_old_m32n128_smalld",
    "pure_persistent_fused_m32n64",
    "pure_persistent_fused_m32n64_cta288",
    "pure_persistent_fused_m32n64_cta576",
    "pure_persistent_fused_m32n64_cta1024",
    "pure_persistent_fused_m32n64_cta2048",
    "pure_persistent_fused_m32n64_cta4096",
    "pure_persistent_ptx_m64n64_cta4096",
    "pure_persistent_ptx_m128n64_cta4096",
    "pure_persistent_ptx_m64n64_cta2048",
    "pure_persistent_ptx_m128n64_cta2048",
    "pure_persistent_ptx_m64n64_cta1024",
    "pure_persistent_ptx_m128n64_cta1024",
    "pure_persistent_ptx_dbvec_m64n64_cta4096",
    "pure_persistent_ptx_dbvec_m128n64_cta4096",
    "pure_persistent_ptx_dbvec_m64n64_cta288",
    "pure_persistent_ptx_dbvec_m64n64_cta512",
    "pure_persistent_ptx_dbvec_m64n64_cta576",
    "pure_persistent_ptx_dbvec_m64n64_cta2048",
    "pure_persistent_ptx_dbvec_m128n64_cta2048",
    "pure_persistent_ptx_dbvec_m64n128_cta2048",
    "pure_persistent_ptx_dbvec_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_m128n64_cta1024",
    "pure_persistent_ptx_dbvec_m64n128_cta1024",
    "pure_persistent_ptx_dbvec_nobounds_m32n64_cta256",
    "pure_persistent_ptx_dbvec_nobounds_m32n64_cta512",
    "pure_persistent_ptx_dbvec_nobounds_m32n64_cta768",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m32n64_cta256",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m32n64_cta512",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m32n64_cta768",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta224",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta288",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta320",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta128",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta192",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta256",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta384",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta512",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta640",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta768",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta2048",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta4096",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta128",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta192",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta256",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta384",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta512",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta640",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta768",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta2048",
    "pure_persistent_ptx_dbvec_nobounds_m64n128_cta2048",
    "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    "pure_persistent_ptx_dbvec_mid",
    "pure_persistent_ptx_dbvec_large",
    "pure_formula_scalar",
    "mixed_formula_scalar",
}

CUTE_BACKENDS = {
    "pure_cute_fused",
    "mixed_cute_fused",
    "pure_cutlass_direct_spatial",
    "pure_cutlass_spatial_epilogue",
    "pure_cutlass_spatial_fanout_epilogue",
    "pure_cutlass_no_yfreq_materialize",
    "pure_cutlass_staged",
    "pure_cutlass_staged_gauss",
    "pure_cutlass_staged_gauss_n64k64",
    "pure_cutlass_staged_n64k64",
    "pure_cutlass_staged_k64",
    "pure_cutlass_staged_gauss_k64",
    "pure_cutlass_staged_s4",
    "pure_cutlass_staged_gauss_s4",
    "pure_cutlass_staged_s2",
    "pure_cutlass_staged_gauss_s2",
    "pure_cutlass_staged_k64_s2",
    "pure_cutlass_staged_gauss_k64_s2",
    "pure_cutlass_staged_k64_s4",
    "pure_cutlass_staged_gauss_k64_s4",
    "pure_cutlass_staged_n64k64_s2",
    "pure_cutlass_staged_gauss_n64k64_s2",
    "pure_cutlass_staged_n64k64_s4",
    "pure_cutlass_staged_gauss_n64k64_s4",
    "pure_cutlass_staged_m64n128k64_s2",
    "pure_cutlass_staged_gauss_m64n128k64_s2",
    "pure_cutlass_staged_m64n128k64",
    "pure_cutlass_staged_gauss_m64n128k64",
    "pure_cutlass_staged_m64n128k64_s4",
    "pure_cutlass_staged_gauss_m64n128k64_s4",
}
MMA_BACKENDS = {
    "pure_mma_pipe",
    "mixed_mma_pipe",
    "pure_mma_pipe_v2",
    "pure_mma_pipe_m128n128",
    "pure_mma_pipe_m128n64",
    "pure_mma_pipe_m128n64_db",
    "pure_mma_pipe_m128n64_w4",
    "pure_mma_pipe_m128n128_w2",
    "pure_mma_pipe_m128n64_gauss",
    "pure_mma_pipe_m128n64_w4_gauss",
    "pure_mma_pipe_m128n32",
    "pure_mma_pipe_m128n32_modc",
    "pure_mma_pipe_m64n32_modc",
    "pure_mma_pipe_m64n64",
    "pure_mma_pipe_m64n64_w4",
    "pure_mma_pipe_m64n256",
    "pure_mma_pipe_m128n64_x2d",
    "pure_mma_pipe_m128n64_x4d",
    "pure_mma_pipe_m64n64_x2d",
    "mixed_mma_pipe_v2",
    "pure_mma_pipe_cpasync_w",
    "pure_mma_pipe_db",
    "pure_mma_pipe_warpspec",
    "pure_mma_pipe_warpspec15",
    "pure_formula_scalar",
    "mixed_formula_scalar",
}
OLD_DIRECT_BACKENDS = {
    "pure_old_m32n32_smalld",
    "pure_old_m32n32_gauss_smalld",
    "pure_old_m32n64_smalld",
    "pure_old_m32n64_smalld_pad64",
    "pure_old_m32n64_gauss_smalld_pad64",
    "pure_old_m16n64_smalld",
    "pure_old_m16n32_smalld",
    "pure_old_m16n16_smalld",
    "pure_old_m64n64_smalld",
    "pure_old_m64n16_smalld",
    "pure_old_m32n128_smalld",
}
PERSISTENT_BACKENDS = {
    "pure_persistent_fused_m32n64",
    "pure_persistent_fused_m32n64_cta288",
    "pure_persistent_fused_m32n64_cta576",
    "pure_persistent_fused_m32n64_cta1024",
    "pure_persistent_fused_m32n64_cta2048",
    "pure_persistent_fused_m32n64_cta4096",
}
PERSISTENT_PTX_BACKENDS = {
    "pure_persistent_ptx_m64n64_cta4096",
    "pure_persistent_ptx_m128n64_cta4096",
    "pure_persistent_ptx_m64n64_cta2048",
    "pure_persistent_ptx_m128n64_cta2048",
    "pure_persistent_ptx_m64n64_cta1024",
    "pure_persistent_ptx_m128n64_cta1024",
    "pure_persistent_ptx_dbvec_m64n64_cta4096",
    "pure_persistent_ptx_dbvec_m128n64_cta4096",
    "pure_persistent_ptx_dbvec_m64n64_cta288",
    "pure_persistent_ptx_dbvec_m64n64_cta512",
    "pure_persistent_ptx_dbvec_m64n64_cta576",
    "pure_persistent_ptx_dbvec_m64n64_cta2048",
    "pure_persistent_ptx_dbvec_m128n64_cta2048",
    "pure_persistent_ptx_dbvec_m64n128_cta2048",
    "pure_persistent_ptx_dbvec_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_m128n64_cta1024",
    "pure_persistent_ptx_dbvec_m64n128_cta1024",
    "pure_persistent_ptx_dbvec_nobounds_m32n64_cta256",
    "pure_persistent_ptx_dbvec_nobounds_m32n64_cta512",
    "pure_persistent_ptx_dbvec_nobounds_m32n64_cta768",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m32n64_cta256",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m32n64_cta512",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m32n64_cta768",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta224",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta288",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta320",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta128",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta192",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta256",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta384",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta512",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta640",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta768",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta2048",
    "pure_persistent_ptx_dbvec_nobounds_m64n64_cta4096",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta128",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta192",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta256",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta384",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta512",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta640",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta768",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta2048",
    "pure_persistent_ptx_dbvec_nobounds_m64n128_cta2048",
    "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    "pure_persistent_ptx_dbvec_mid",
    "pure_persistent_ptx_dbvec_large",
}

MIDSHAPE_TARGET_ROUTER = {
    # Router evidence:
    # results_fp16/direct_gemm_fp16_midshape_router_candidates_targets_20260506.json
    # results_fp16/direct_gemm_fp16_midshape_router_candidates_35shape_20260506.json
    (32, 128, 256, 256): "pure_cutlass_staged_gauss_n64k64",
    (32, 128, 512, 512): "pure_cutlass_staged",
    (32, 256, 256, 256): "pure_cutlass_staged",
    (32, 256, 512, 512): "pure_cutlass_staged",
    (32, 512, 256, 256): "pure_cutlass_staged",
    (32, 512, 512, 512): "pure_cutlass_staged_gauss",
    (32, 1024, 256, 256): "naive16",
    (32, 1024, 512, 512): "pure_cutlass_staged_gauss",
}

MIDSHAPE_2X_ROUTER = {
    (32, 128, 512, 512): "pure_cutlass_staged",
}


def backend_available(backend: str) -> bool:
    if backend in CUTE_BACKENDS:
        return _cute_backend is not None
    if backend in MMA_BACKENDS:
        return _mma_backend is not None
    if backend in OLD_DIRECT_BACKENDS:
        return _old_direct_backend is not None
    if backend in PERSISTENT_BACKENDS:
        return _persistent_backend is not None
    if backend in PERSISTENT_PTX_BACKENDS:
        return _persistent_ptx_backend is not None
    return False


def recommend_midshape_backend(
    batch_size: int,
    seq_len: int,
    in_num: int,
    out_num: int,
    *,
    require_2x: bool = False,
) -> str | None:
    """Return the measured mid-shape route, or None when no route is promoted."""
    key = (batch_size, seq_len, in_num, out_num)
    router = MIDSHAPE_2X_ROUTER if require_2x else MIDSHAPE_TARGET_ROUTER
    return router.get(key)


def _backend_module(backend: str):
    if backend in OLD_DIRECT_BACKENDS:
        if _old_direct_backend is None:
            raise RuntimeError("Old direct fp16 extension is unavailable.")
        return _old_direct_backend
    if backend in PERSISTENT_BACKENDS:
        if _persistent_backend is None:
            raise RuntimeError("Persistent fused fp16 extension is unavailable.")
        return _persistent_backend
    if backend in PERSISTENT_PTX_BACKENDS:
        if _persistent_ptx_backend is None:
            raise RuntimeError("Persistent PTX fused fp16 extension is unavailable.")
        return _persistent_ptx_backend
    if backend in CUTE_BACKENDS:
        if _cute_backend is None:
            raise RuntimeError("CuTe fp16 pipeline extension is unavailable.")
        return _cute_backend
    if backend in MMA_BACKENDS:
        if _mma_backend is None:
            raise RuntimeError("MMA fp16 pipeline extension is unavailable.")
        return _mma_backend
    raise ValueError(f"Unknown backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}")


def pack_weight_for_backend(weight_freq: torch.Tensor, backend: str) -> torch.Tensor:
    module = _backend_module(backend)
    if backend in OLD_DIRECT_BACKENDS or backend in PERSISTENT_BACKENDS:
        return weight_freq.contiguous()
    if backend in PERSISTENT_PTX_BACKENDS:
        return module.pack_weight(weight_freq.contiguous())
    if backend in {
        "pure_cutlass_staged_gauss",
        "pure_cutlass_staged_gauss_n64k64",
        "pure_cutlass_staged_gauss_k64",
        "pure_cutlass_staged_gauss_s4",
        "pure_cutlass_staged_gauss_s2",
        "pure_cutlass_staged_gauss_k64_s2",
        "pure_cutlass_staged_gauss_k64_s4",
        "pure_cutlass_staged_gauss_n64k64_s2",
        "pure_cutlass_staged_gauss_n64k64_s4",
        "pure_cutlass_staged_gauss_m64n128k64_s2",
        "pure_cutlass_staged_gauss_m64n128k64",
        "pure_cutlass_staged_gauss_m64n128k64_s4",
    }:
        return module.pack_weight_fp16_cute_gauss_ada4090(weight_freq.contiguous())
    if backend in CUTE_BACKENDS:
        return module.pack_weight_fp16_cute_ada4090(weight_freq.contiguous())
    return module.pack_weight_fp16_mma_ada4090(weight_freq.contiguous())


def forward_backend(x_4d: torch.Tensor, weight_packed: torch.Tensor, backend: str) -> torch.Tensor:
    module = _backend_module(backend)
    if backend in OLD_DIRECT_BACKENDS:
        return module.flash_eq_linear_forward_direct_gemm_fp16_variant_ada4090(x_4d, weight_packed, backend)
    if backend in PERSISTENT_BACKENDS:
        return module.forward(x_4d, weight_packed, backend)
    if backend in PERSISTENT_PTX_BACKENDS:
        return module.forward(x_4d, weight_packed, backend)
    if backend in CUTE_BACKENDS:
        return module.flash_eq_linear_forward_direct_gemm_fp16_cute_ada4090(x_4d, weight_packed, backend)
    return module.flash_eq_linear_forward_direct_gemm_fp16_mma_ada4090(x_4d, weight_packed, backend)


def backward_backend(grad_y_4d: torch.Tensor, x_4d: torch.Tensor, weight_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _old_direct_backend is None:
        raise RuntimeError("Old direct fp16 extension is required for routed backward but is unavailable.")
    return tuple(
        _old_direct_backend.flash_eq_linear_backward_direct_gemm_fp16_ada4090(
            grad_y_4d.contiguous(),
            x_4d.contiguous(),
            weight_freq.contiguous(),
        )
    )


class FlashEQLinearDirectGemmFp16RoutedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_4d: torch.Tensor, weight_freq: torch.Tensor, bias: torch.Tensor | None, backend: str):
        weight = weight_freq if weight_freq.dtype == torch.float16 else weight_freq.to(dtype=torch.float16)
        x_contig = x_4d.contiguous()
        weight_packed = pack_weight_for_backend(weight, backend)
        y = forward_backend(x_contig, weight_packed, backend)
        ctx.save_for_backward(x_contig, weight)
        ctx.has_bias = bias is not None
        if bias is not None:
            y = y + bias.to(device=y.device, dtype=y.dtype).view(1, 1, weight.size(0), 1)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_4d, weight_freq = ctx.saved_tensors
        grad_y = grad_output.contiguous()
        grad_x, grad_weight = backward_backend(grad_y, x_4d, weight_freq)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.sum(dim=(0, 1, 3)).view(weight_freq.size(0), 1)
        return grad_x, grad_weight, grad_bias, None


def flash_eq_linear_direct_gemm_fp16_routed(
    x_4d: torch.Tensor,
    weight_freq: torch.Tensor,
    bias: torch.Tensor | None,
    backend: str,
) -> torch.Tensor:
    return FlashEQLinearDirectGemmFp16RoutedFunction.apply(x_4d, weight_freq, bias, backend)


def _cuda_time_ms(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def profile_backend(
    x_4d: torch.Tensor,
    weight_packed: torch.Tensor,
    backend: str,
    repeats: int = 80,
    warmup: int = 20,
) -> dict:
    module = _backend_module(backend)
    if backend in OLD_DIRECT_BACKENDS:
        return {
            "forward_kernel_ms": _cuda_time_ms(
                lambda: module.flash_eq_linear_forward_direct_gemm_fp16_variant_ada4090(x_4d, weight_packed, backend),
                repeats,
                warmup,
            )
        }
    if backend in PERSISTENT_BACKENDS:
        return module.profile_forward(x_4d, weight_packed, backend, repeats, warmup)
    if backend in PERSISTENT_PTX_BACKENDS:
        return module.profile_forward(x_4d, weight_packed, backend, repeats, warmup)
    if backend in CUTE_BACKENDS:
        return module.flash_eq_linear_profile_direct_gemm_fp16_cute_ada4090(
            x_4d, weight_packed, backend, repeats, warmup
        )
    return module.flash_eq_linear_profile_direct_gemm_fp16_mma_ada4090(
        x_4d, weight_packed, backend, repeats, warmup
    )


class CudaFlashEQLinearDirectGemmFp16PipelineAda4090(nn.Module):
    def __init__(
        self,
        inNum: int,
        outNum: int,
        tranNum: int = 4,
        bias: bool = False,
        eqlinear_weights: torch.Tensor | None = None,
        backend: str = "pure_cute_fused",
    ) -> None:
        super().__init__()
        if tranNum != 4:
            raise ValueError(f"Only tranNum=4 is supported, got {tranNum}")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}")

        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.use_bias = bias
        self.backend = backend

        if eqlinear_weights is not None:
            freq = spatial_weight_to_freq_weight(eqlinear_weights.to(dtype=torch.float16))
            self.weights = nn.Parameter(freq)
        else:
            self.weights = nn.Parameter(torch.empty(outNum, inNum, tranNum, dtype=torch.float16))
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.register_buffer("weight_packed", torch.empty(0, dtype=torch.float16), persistent=False)
        if self.use_bias:
            self.c = nn.Parameter(torch.empty(outNum, 1, dtype=torch.float16))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)
        else:
            self.register_parameter("c", None)
        self._cache_weight_version: int | None = None

    @torch.no_grad()
    def repack_weight(self) -> torch.Tensor:
        if not self.weights.is_cuda:
            raise RuntimeError("repack_weight requires CUDA weights.")
        weight = self.weights if self.weights.dtype == torch.float16 else self.weights.to(dtype=torch.float16)
        self.weight_packed = pack_weight_for_backend(weight, self.backend)
        self._cache_weight_version = self._weights_version()
        return self.weight_packed

    def _weights_version(self) -> int:
        return int(getattr(self.weights, "_version", 0))

    def _packed_weight(self) -> torch.Tensor:
        if (
            self.weight_packed.numel() == 0
            or self.weight_packed.device != self.weights.device
            or self.weight_packed.dtype != torch.float16
            or self._cache_weight_version != self._weights_version()
        ):
            self.repack_weight()
        return self.weight_packed

    def _apply_bias(self, y: torch.Tensor) -> torch.Tensor:
        if self.c is None:
            return y
        bias = self.c.to(device=y.device, dtype=y.dtype).view(1, 1, self.outNum, 1)
        return y + bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("CudaFlashEQLinearDirectGemmFp16PipelineAda4090 requires CUDA tensors.")
        if self.weights.device != x.device:
            raise RuntimeError("Input and weights must be on the same CUDA device.")
        if self.c is not None and self.c.device != x.device:
            raise RuntimeError("Bias must be on the same CUDA device as the input.")
        if x.dtype != torch.float16:
            raise RuntimeError(f"Expected float16 input, got {x.dtype}.")
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

        if torch.is_grad_enabled() and (
            x_4d.requires_grad or self.weights.requires_grad or (self.c is not None and self.c.requires_grad)
        ):
            weight = self.weights if self.weights.dtype == torch.float16 else self.weights.to(dtype=torch.float16)
            y = flash_eq_linear_direct_gemm_fp16_routed(x_4d, weight, self.c, self.backend)
        else:
            y = forward_backend(x_4d, self._packed_weight(), self.backend)
            y = self._apply_bias(y)
        y = y.reshape(y.size(0), y.size(1), -1)
        if orig_dim == 4:
            return y.view(bsz, height, width, -1)
        return y.view(bsz, seq_len, -1)
