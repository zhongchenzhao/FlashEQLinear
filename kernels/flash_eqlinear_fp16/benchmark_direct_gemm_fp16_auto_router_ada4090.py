# -*- coding: utf-8 -*-
"""Benchmark the shape-routed fp16 direct-GEMM EQLinear module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .flash_EQLinear_fp16 import Standard_linear_inter
    from .flash_EQLinear_fp16_direct_gemm_auto_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16AutoAda4090,
        PAPER_EXTENDED_ROUTE_TABLE,
    )
    from .flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16PipelineAda4090,
        forward_backend,
    )
    from .flash_EQLinear_fp32 import spatial_to_freq_weight
else:
    from flash_EQLinear_fp16 import Standard_linear_inter
    from flash_EQLinear_fp16_direct_gemm_auto_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16AutoAda4090,
        PAPER_EXTENDED_ROUTE_TABLE,
    )
    from flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16PipelineAda4090,
        forward_backend,
    )
    from flash_EQLinear_fp32 import spatial_to_freq_weight


PAPER_HIGHLIGHT_SHAPE_SET = {
    (32, 4096, 64, 64),
    (32, 4096, 128, 128),
    (32, 4096, 256, 256),
    (32, 4096, 512, 512),
    (32, 4096, 1024, 1024),
    (32, 4096, 2048, 2048),
}

SHAPES_DEFAULT = list(PAPER_EXTENDED_ROUTE_TABLE)


def parse_shapes(value: str) -> list[tuple[int, int, int, int]]:
    shapes = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(x.strip()) for x in item.split(",")]
        if len(parts) == 3:
            batch, seqlen, channels = parts
            shapes.append((batch, seqlen, channels, channels))
        elif len(parts) == 4:
            shapes.append(tuple(parts))
        else:
            raise ValueError(f"Expected B,L,C or B,L,C,D shape, got {item!r}")
    return shapes


def _cuda_time(fn, iters: int, warmup: int = 10) -> float:
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


def _set_freq_weight(model, weight_freq: torch.Tensor) -> None:
    with torch.no_grad():
        model.weights.copy_(weight_freq.to(device=model.weights.device, dtype=model.weights.dtype))
        if hasattr(model, "repack_weight"):
            model.repack_weight()


def _max_mean_abs(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item()}


def _tflops(batch: int, seqlen: int, c_in: int, c_out: int, mac_factor: float, ms: float) -> float:
    if ms <= 0:
        return 0.0
    macs = mac_factor * batch * seqlen * c_in * c_out
    return 2.0 * macs / (ms / 1000.0) / 1e12


def _iters(seqlen: int, c_in: int, c_out: int | None = None) -> int:
    dim = max(c_in, c_out if c_out is not None else c_in)
    iters = 200 if dim <= 128 else 80
    if seqlen >= 4096 and dim >= 256:
        iters = 30
    if seqlen >= 4096 and dim >= 1024:
        iters = 10
    return iters


def _apply_flat_bias(y_flat: torch.Tensor, bias: torch.Tensor | None, tran_num: int = 4) -> torch.Tensor:
    if bias is None:
        return y_flat
    bias_flat = bias.repeat(1, tran_num).reshape(-1).to(device=y_flat.device, dtype=y_flat.dtype)
    return y_flat + bias_flat


def run_router_benchmark(
    shapes: list[tuple[int, int, int, int]],
    *,
    warmup_iters: int,
    include_naive: bool,
    bias: bool,
) -> list[dict]:
    rows = []
    device = torch.device("cuda")
    for batch, seqlen, c_in, c_out in shapes:
        if (batch, seqlen, c_in, c_out) not in PAPER_EXTENDED_ROUTE_TABLE:
            raise ValueError(f"Shape {(batch, seqlen, c_in, c_out)} is not in PAPER_EXTENDED_ROUTE_TABLE.")
        torch.manual_seed(batch * 1000000000 + seqlen * 100000 + c_in * 1000 + c_out)
        x = torch.randn(batch, seqlen, c_in, 4, device=device, dtype=torch.float16)
        x_flat = x.reshape(batch, seqlen, c_in * 4)
        weight_spatial = torch.randn(c_out, c_in, 4, device=device, dtype=torch.float16)
        weight_freq = spatial_to_freq_weight(weight_spatial.float()).half()

        router = CudaFlashEQLinearDirectGemmFp16AutoAda4090(
            c_in,
            c_out,
            bias=bias,
            eqlinear_weights=weight_spatial,
        ).to(device)
        _set_freq_weight(router, weight_freq)
        backend = router.select_backend_for_shape(batch, seqlen)
        router.preload_branches([backend])
        router.warmup([(batch, seqlen, c_in, c_out)], iters=warmup_iters)

        selected_module = CudaFlashEQLinearDirectGemmFp16PipelineAda4090(
            c_in,
            c_out,
            bias=bias,
            eqlinear_weights=weight_spatial,
            backend=backend,
        ).to(device)
        _set_freq_weight(selected_module, weight_freq)
        if bias:
            with torch.no_grad():
                selected_module.c.copy_(router.c)

        weight_packed = router.packed_weight_for_backend(backend)

        def selected_backend_raw_with_optional_bias() -> torch.Tensor:
            y_flat = forward_backend(x, weight_packed, backend).reshape(batch, seqlen, c_out * 4)
            return _apply_flat_bias(y_flat, router.c if bias else None).view(batch, seqlen, c_out, 4)

        with torch.inference_mode():
            y_router = router(x_flat).view(batch, seqlen, c_out, 4)
            y_selected_module = selected_module(x_flat).view(batch, seqlen, c_out, 4)
            y_backend = selected_backend_raw_with_optional_bias()
            torch.cuda.synchronize()

        iters = _iters(seqlen, c_in, c_out)
        row = {
            "shape": [batch, seqlen, c_in, c_out],
            "highlight": (batch, seqlen, c_in, c_out) in PAPER_HIGHLIGHT_SHAPE_SET,
            "bias": bias,
            "iters": iters,
            "backend": backend,
            "cached_backends": list(router.cached_backends),
            "correctness_vs_selected_backend": _max_mean_abs(y_router, y_backend),
            "router_full_ms": None,
            "selected_backend_module_ms": None,
            "selected_backend_raw_ms": None,
            "router_over_selected_module": None,
            "router_over_selected_raw": None,
            "naive16_full_ms": None,
            "naive16_tflops": None,
            "speedup_vs_naive": None,
            "selected_module_vs_selected_backend": _max_mean_abs(y_selected_module, y_backend),
        }

        with torch.inference_mode():
            router_ms = _cuda_time(lambda: router(x_flat), iters, warmup=warmup_iters)
            selected_module_ms = _cuda_time(lambda: selected_module(x_flat), iters, warmup=warmup_iters)
            backend_ms = _cuda_time(selected_backend_raw_with_optional_bias, iters, warmup=warmup_iters)
        row["router_full_ms"] = router_ms
        row["selected_backend_module_ms"] = selected_module_ms
        row["selected_backend_raw_ms"] = backend_ms
        row["router_over_selected_module"] = router_ms / selected_module_ms if selected_module_ms > 0 else 0.0
        row["router_over_selected_raw"] = router_ms / backend_ms if backend_ms > 0 else 0.0

        if include_naive:
            naive = Standard_linear_inter(c_in, c_out, tranNum=4, bias=bias).to(device).half()
            if bias:
                with torch.no_grad():
                    naive.c.copy_(router.c)
            with torch.inference_mode():
                naive_ms = _cuda_time(lambda: naive(x_flat), iters, warmup=warmup_iters)
            row["naive16_full_ms"] = naive_ms
            row["naive16_tflops"] = _tflops(batch, seqlen, c_in, c_out, 16.0, naive_ms)
            row["speedup_vs_naive"] = naive_ms / router_ms if router_ms > 0 else 0.0

        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=None)
    parser.add_argument("--out", default="results_fp16/direct_gemm_fp16_auto_router_20260511.json")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--skip-naive", action="store_true")
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()

    shapes = parse_shapes(args.shapes) if args.shapes else SHAPES_DEFAULT
    rows = run_router_benchmark(
        shapes,
        warmup_iters=args.warmup_iters,
        include_naive=not args.skip_naive,
        bias=args.bias,
    )
    payload = {
        "bias": args.bias,
        "route_table": {",".join(map(str, key)): value for key, value in PAPER_EXTENDED_ROUTE_TABLE.items()},
        "performance_shapes": [list(shape) for shape in shapes],
        "performance": rows,
        "highlight_performance": [row for row in rows if row.get("highlight")],
    }
    out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
