# -*- coding: utf-8 -*-
"""Stable release API for the Ada/RTX4090 fp16 direct-GEMM EQLinear router."""

from __future__ import annotations

if __package__:
    from .flash_EQLinear_fp16_direct_gemm_auto_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16AutoAda4090,
        PAPER_EXTENDED_BACKENDS,
        PAPER_EXTENDED_ROUTE_TABLE,
        PAPER_EXTENDED_SHAPES,
        create_release_eqlinear_fp16_ada4090,
        route_backend_for_shape,
    )
    from .flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        FlashEQLinearDirectGemmFp16RoutedFunction,
        backward_backend,
        flash_eq_linear_direct_gemm_fp16_routed,
    )
else:
    from flash_EQLinear_fp16_direct_gemm_auto_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16AutoAda4090,
        PAPER_EXTENDED_BACKENDS,
        PAPER_EXTENDED_ROUTE_TABLE,
        PAPER_EXTENDED_SHAPES,
        create_release_eqlinear_fp16_ada4090,
        route_backend_for_shape,
    )
    from flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        FlashEQLinearDirectGemmFp16RoutedFunction,
        backward_backend,
        flash_eq_linear_direct_gemm_fp16_routed,
    )


FlashEQLinearFp16DirectGemmAda4090 = CudaFlashEQLinearDirectGemmFp16AutoAda4090
create_flash_eqlinear_fp16_direct_gemm_ada4090 = create_release_eqlinear_fp16_ada4090

__all__ = [
    "CudaFlashEQLinearDirectGemmFp16AutoAda4090",
    "FlashEQLinearDirectGemmFp16RoutedFunction",
    "FlashEQLinearFp16DirectGemmAda4090",
    "PAPER_EXTENDED_BACKENDS",
    "PAPER_EXTENDED_ROUTE_TABLE",
    "PAPER_EXTENDED_SHAPES",
    "backward_backend",
    "create_flash_eqlinear_fp16_direct_gemm_ada4090",
    "create_release_eqlinear_fp16_ada4090",
    "flash_eq_linear_direct_gemm_fp16_routed",
    "route_backend_for_shape",
]
