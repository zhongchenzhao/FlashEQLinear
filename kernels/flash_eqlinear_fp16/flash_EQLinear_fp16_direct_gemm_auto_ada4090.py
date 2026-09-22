# -*- coding: utf-8 -*-
"""Shape-routed fp16 direct-GEMM EQLinear module for Ada/RTX4090.

The router exposes one PyTorch module while dispatching to the measured best
backend for each benchmark shape. Weight packing is cached per backend so the
timed forward path matches the underlying branch cost after warmup/preload.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

if __package__:
    from .flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        VALID_BACKENDS,
        backend_available,
        flash_eq_linear_direct_gemm_fp16_routed,
        forward_backend,
        pack_weight_for_backend,
        profile_backend,
    )
    from .flash_EQLinear_python_ref import spatial_weight_to_freq_weight
else:
    from flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        VALID_BACKENDS,
        backend_available,
        flash_eq_linear_direct_gemm_fp16_routed,
        forward_backend,
        pack_weight_for_backend,
        profile_backend,
    )
    from flash_EQLinear_python_ref import spatial_weight_to_freq_weight


PAPER_EXTENDED_ROUTE_TABLE: dict[tuple[int, int, int, int], str] = {
    (32, 128, 64, 64): "pure_old_m32n32_smalld",
    (32, 128, 128, 128): "pure_old_m32n32_smalld",
    (32, 128, 256, 256): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 128, 512, 512): "pure_cutlass_staged",
    (32, 128, 1024, 1024): "pure_cutlass_staged_gauss",
    (32, 128, 2048, 2048): "pure_cutlass_staged_gauss",
    (32, 256, 64, 64): "pure_old_m32n32_smalld",
    (32, 256, 128, 128): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 256, 256, 256): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 256, 512, 512): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 256, 1024, 1024): "pure_cutlass_staged_gauss",
    (32, 256, 2048, 2048): "pure_cutlass_staged_gauss",
    (32, 512, 64, 64): "pure_old_m64n64_smalld",
    (32, 512, 128, 128): "pure_persistent_fused_m32n64_cta1024",
    (32, 512, 256, 256): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 512, 512, 512): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 512, 1024, 1024): "pure_cutlass_staged_gauss",
    (32, 512, 2048, 2048): "pure_cutlass_staged_gauss",
    (32, 1024, 64, 64): "pure_old_m32n64_smalld_pad64",
    (32, 1024, 128, 128): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 1024, 256, 256): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 1024, 512, 512): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 1024, 1024, 1024): "pure_cutlass_staged_gauss",
    (32, 1024, 2048, 2048): "pure_mma_pipe_m128n64",
    (32, 4096, 64, 64): "pure_persistent_fused_m32n64_cta4096",
    (32, 4096, 128, 128): "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta256",
    (32, 4096, 256, 256): "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta512",
    (32, 4096, 512, 512): "pure_persistent_ptx_dbvec_nobounds_m128n64_cta2048",
    (32, 4096, 1024, 1024): "pure_mma_pipe_m128n128_w2",
    (32, 4096, 2048, 2048): "pure_mma_pipe_m128n64",
    (128, 196, 60, 60): "pure_old_m32n64_gauss_smalld_pad64",
    (128, 196, 120, 120): "pure_persistent_fused_m32n64_cta1024",
    (32, 4096, 192, 192): "pure_persistent_ptx_dbvec_vecld_nobounds_m64n64_cta224",
}

PAPER_EXTENDED_SHAPES = tuple(PAPER_EXTENDED_ROUTE_TABLE)
PAPER_EXTENDED_BACKENDS = tuple(dict.fromkeys(PAPER_EXTENDED_ROUTE_TABLE.values()))

__all__ = [
    "CudaFlashEQLinearDirectGemmFp16AutoAda4090",
    "PAPER_EXTENDED_BACKENDS",
    "PAPER_EXTENDED_ROUTE_TABLE",
    "PAPER_EXTENDED_SHAPES",
    "create_auto_eqlinear_fp16_ada4090",
    "create_release_eqlinear_fp16_ada4090",
    "route_backend_for_shape",
]


def route_backend_for_shape(
    batch_size: int,
    seq_len: int,
    in_num: int,
    out_num: int,
    *,
    fallback_backend: str | None = "pure_cute_fused",
    route_table: dict[tuple[int, int, int, int], str] | None = None,
) -> str:
    """Return the routed backend for a flattened ``(B, L, C, D)`` shape."""
    table = PAPER_EXTENDED_ROUTE_TABLE if route_table is None else route_table
    key = (int(batch_size), int(seq_len), int(in_num), int(out_num))
    backend = table.get(key)
    if backend is not None:
        return backend
    if fallback_backend is None:
        raise KeyError(f"No routed backend for shape {key}")
    return fallback_backend


class CudaFlashEQLinearDirectGemmFp16AutoAda4090(nn.Module):
    """One-call shape-routed EQLinear module.

    The module accepts the same flattened input convention as the existing
    direct-GEMM wrappers: ``(B, L, C * 4)`` or ``(B, H, W, C * 4)`` and returns
    ``(B, L, D * 4)`` or ``(B, H, W, D * 4)``.
    """

    def __init__(
        self,
        inNum: int,
        outNum: int,
        tranNum: int = 4,
        bias: bool = False,
        eqlinear_weights: torch.Tensor | None = None,
        eqlinear_bias: torch.Tensor | None = None,
            *,
        route_table: dict[tuple[int, int, int, int], str] | None = None,
        fallback_backend: str | None = "pure_cute_fused",
        preload: bool = False,
    ) -> None:
        super().__init__()
        if tranNum != 4:
            raise ValueError(f"Only tranNum=4 is supported, got {tranNum}")
        self.inNum = int(inNum)
        self.outNum = int(outNum)
        self.tranNum = int(tranNum)
        self.use_bias = bool(bias)
        self.route_table = dict(PAPER_EXTENDED_ROUTE_TABLE if route_table is None else route_table)
        self.fallback_backend = fallback_backend
        self._packed_cache: dict[str, torch.Tensor] = {}
        self._cache_weight_version: int | None = None

        for backend in set(self.route_table.values()) | ({fallback_backend} if fallback_backend else set()):
            if backend not in VALID_BACKENDS:
                raise ValueError(f"Unknown backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}")

        if eqlinear_weights is not None:
            freq = spatial_weight_to_freq_weight(eqlinear_weights.to(dtype=torch.float16))
            self.weights = nn.Parameter(freq)
        else:
            self.weights = nn.Parameter(torch.empty(outNum, inNum, tranNum, dtype=torch.float16))
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.use_bias:
            if eqlinear_bias is not None:
                # EQLinearInter uses c with shape [outNum, 1]. Keep the same
                # parameter contract so its gradient also has shape [outNum, 1].
                eqlinear_bias = eqlinear_bias.detach().contiguous().view(outNum, 1)
                self.c = nn.Parameter(eqlinear_bias)
            else:
                self.c = nn.Parameter(torch.empty(outNum, 1, dtype=torch.float16))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.c, -bound, bound)
        else:
            self.register_parameter("c", None)

        if preload and self.weights.is_cuda:
            self.preload_branches()

    @property
    def available_routes(self) -> dict[tuple[int, int, int, int], str]:
        return dict(self.route_table)

    @property
    def cached_backends(self) -> tuple[str, ...]:
        return tuple(self._packed_cache)

    def clear_cache(self) -> None:
        self._packed_cache.clear()
        self._cache_weight_version = None

    def select_backend_for_shape(self, batch_size: int, seq_len: int) -> str:
        return route_backend_for_shape(
            batch_size,
            seq_len,
            self.inNum,
            self.outNum,
            fallback_backend=self.fallback_backend,
            route_table=self.route_table,
        )

    def select_backend(self, x: torch.Tensor) -> str:
        _, seq_len, _, _ = self._as_4d_input(x)
        return self.select_backend_for_shape(int(x.size(0)), seq_len)

    @torch.no_grad()
    def repack_weight(self, backend: str | None = None) -> dict[str, torch.Tensor] | torch.Tensor:
        if not self.weights.is_cuda:
            raise RuntimeError("repack_weight requires CUDA weights.")
        self._ensure_cache_fresh()
        if backend is not None:
            self._packed_cache[backend] = self._pack_for_backend(backend)
            return self._packed_cache[backend]
        self.preload_branches()
        return dict(self._packed_cache)

    @torch.no_grad()
    def preload_branches(self, backends: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        if not self.weights.is_cuda:
            raise RuntimeError("preload_branches requires CUDA weights.")
        self._ensure_cache_fresh()
        selected = tuple(dict.fromkeys(backends or self._backends_for_module()))
        for backend in selected:
            if backend_available(backend) and backend not in self._packed_cache:
                self._packed_cache[backend] = self._pack_for_backend(backend)
        return dict(self._packed_cache)

    def packed_weight_for_backend(self, backend: str) -> torch.Tensor:
        return self._packed_weight(backend)

    @torch.no_grad()
    def warmup(
        self,
        shapes: Iterable[tuple[int, int] | tuple[int, int, int, int]] | None = None,
        *,
        iters: int = 3,
    ) -> None:
        if not self.weights.is_cuda:
            raise RuntimeError("warmup requires CUDA weights.")
        warm_shapes = tuple(shapes or self._warmup_shapes_for_module())
        normalized_shapes = tuple(self._normalize_warmup_shape(shape) for shape in warm_shapes)
        warm_backends = [self.select_backend_for_shape(batch_size, seq_len) for batch_size, seq_len in normalized_shapes]
        self.preload_branches(warm_backends)
        device = self.weights.device
        for batch_size, seq_len in normalized_shapes:
            x = torch.empty(
                batch_size,
                seq_len,
                self.inNum * self.tranNum,
                device=device,
                dtype=torch.float16,
            )
            for _ in range(iters):
                self.forward(x)
        torch.cuda.synchronize(device)

    def _warmup_shapes_for_module(self) -> tuple[tuple[int, int], ...]:
        shapes = [
            (batch_size, seq_len)
            for batch_size, seq_len, in_num, out_num in self.route_table
            if in_num == self.inNum and out_num == self.outNum
        ]
        return tuple(dict.fromkeys(shapes))

    def _backends_for_module(self) -> tuple[str, ...]:
        backends = [
            backend
            for (_, _, in_num, out_num), backend in self.route_table.items()
            if in_num == self.inNum and out_num == self.outNum
        ]
        if self.fallback_backend is not None:
            backends.append(self.fallback_backend)
        return tuple(dict.fromkeys(backends))

    def _normalize_warmup_shape(self, shape: tuple[int, ...]) -> tuple[int, int]:
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
        if len(shape) == 4:
            batch_size, seq_len, in_num, out_num = [int(x) for x in shape]
            if in_num != self.inNum or out_num != self.outNum:
                raise ValueError(f"Warmup shape {shape} does not match module C/D {(self.inNum, self.outNum)}")
            return batch_size, seq_len
        raise ValueError(f"Expected warmup shape (B, L) or (B, L, C, D), got {shape!r}")

    def _pack_for_backend(self, backend: str) -> torch.Tensor:
        if not backend_available(backend):
            raise RuntimeError(f"Backend {backend!r} is unavailable.")
        weight = self.weights if self.weights.dtype == torch.float16 else self.weights.to(dtype=torch.float16)
        return pack_weight_for_backend(weight, backend)

    def _weights_version(self) -> int:
        return int(getattr(self.weights, "_version", 0))

    def _ensure_cache_fresh(self) -> None:
        version = self._weights_version()
        if self._cache_weight_version != version:
            self._packed_cache.clear()
            self._cache_weight_version = version

    def _packed_weight(self, backend: str) -> torch.Tensor:
        self._ensure_cache_fresh()
        cached = self._packed_cache.get(backend)
        if cached is None or cached.device != self.weights.device or cached.dtype != torch.float16:
            self._packed_cache[backend] = self._pack_for_backend(backend)
        return self._packed_cache[backend]

    def _apply_bias(self, y: torch.Tensor) -> torch.Tensor:
        if self.c is None:
            return y
        bias = self.c.to(device=y.device, dtype=y.dtype).view(1, 1, self.outNum, 1)
        return y + bias

    def _as_4d_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        if not x.is_cuda:
            raise RuntimeError("CudaFlashEQLinearDirectGemmFp16AutoAda4090 requires CUDA tensors.")
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
            batch_size, height, width, channels = x.shape
            if channels != self.inNum * self.tranNum:
                raise ValueError(f"Expected last dim {self.inNum * self.tranNum}, got {channels}")
            return x.view(batch_size, height * width, self.inNum, self.tranNum), height * width, orig_dim, width
        if orig_dim == 3:
            batch_size, seq_len, channels = x.shape
            if channels != self.inNum * self.tranNum:
                raise ValueError(f"Expected last dim {self.inNum * self.tranNum}, got {channels}")
            return x.view(batch_size, seq_len, self.inNum, self.tranNum), seq_len, orig_dim, 0
        raise ValueError(f"Expected 3D or 4D input, got {orig_dim}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_4d, seq_len, orig_dim, width = self._as_4d_input(x)
        batch_size = int(x_4d.size(0))
        backend = self.select_backend_for_shape(batch_size, seq_len)
        if torch.is_grad_enabled() and (
            x_4d.requires_grad or self.weights.requires_grad or (self.c is not None and self.c.requires_grad)
        ):
            weight = self.weights if self.weights.dtype == torch.float16 else self.weights.to(dtype=torch.float16)
            y = flash_eq_linear_direct_gemm_fp16_routed(x_4d, weight, self.c, backend)
        else:
            y = forward_backend(x_4d, self._packed_weight(backend), backend)
            y = self._apply_bias(y)
        y = y.reshape(y.size(0), y.size(1), -1)
        if orig_dim == 4:
            height = seq_len // width
            return y.view(batch_size, height, width, -1)
        return y.view(batch_size, seq_len, -1)

    def profile_forward(self, x: torch.Tensor, repeats: int = 80, warmup: int = 20) -> dict:
        x_4d, seq_len, _, _ = self._as_4d_input(x)
        batch_size = int(x_4d.size(0))
        backend = self.select_backend_for_shape(batch_size, seq_len)
        result = profile_backend(x_4d, self._packed_weight(backend), backend, repeats, warmup)
        result["backend"] = backend
        result["shape"] = [batch_size, seq_len, self.inNum, self.outNum]
        return result


def create_auto_eqlinear_fp16_ada4090(*args, **kwargs) -> CudaFlashEQLinearDirectGemmFp16AutoAda4090:
    return CudaFlashEQLinearDirectGemmFp16AutoAda4090(*args, **kwargs)


def create_release_eqlinear_fp16_ada4090(*args, **kwargs) -> CudaFlashEQLinearDirectGemmFp16AutoAda4090:
    """Create the strict release router.

    The release factory disables fallback by default so an unsupported shape is
    reported immediately instead of silently taking a non-tuned backend.
    """
    kwargs.setdefault("fallback_backend", None)
    return CudaFlashEQLinearDirectGemmFp16AutoAda4090(*args, **kwargs)
