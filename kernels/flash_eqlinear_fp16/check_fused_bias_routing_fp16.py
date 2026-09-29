# -*- coding: utf-8 -*-
"""Check fused-bias epilogues and bucketed routing of the fp16 kernels.

Run from the repository root after rebuilding the extensions:

    python -m kernels.flash_eqlinear_fp16.check_fused_bias_routing_fp16
    python -m kernels.flash_eqlinear_fp16.check_fused_bias_routing_fp16 --all-backends

1. Prints the backend chosen for the Flash EQ-ViT layer shapes (no GPU needed).
2. Runs each backend in its own process, so a failing kernel cannot leave CUDA
   error state behind for the next one. Per test shape, the bias-free output is
   compared with the fp32 reference and the fused-bias output with the unfused
   ``y + bias`` bit for bit. A failing bias-free call is reported as
   pre-existing, since it does not involve the bias.
3. Checks the module inference and autograd paths on routed shapes, including
   the Flash EQ-ViT-Base layers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

if __package__:
    from .flash_EQLinear_fp16_direct_gemm_auto_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16AutoAda4090,
        PAPER_EXTENDED_ROUTE_TABLE,
        route_backend_for_shape,
    )
    from .flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        VALID_BACKENDS,
        backend_available,
        backend_supports_shape,
        forward_backend,
        pack_weight_for_backend,
    )
    from .flash_EQLinear_python_ref import flash_eq_linear_forward_python
else:
    from flash_EQLinear_fp16_direct_gemm_auto_ada4090 import (
        CudaFlashEQLinearDirectGemmFp16AutoAda4090,
        PAPER_EXTENDED_ROUTE_TABLE,
        route_backend_for_shape,
    )
    from flash_EQLinear_fp16_direct_gemm_pipeline_ada4090 import (
        VALID_BACKENDS,
        backend_available,
        backend_supports_shape,
        forward_backend,
        pack_weight_for_backend,
    )
    from flash_EQLinear_python_ref import flash_eq_linear_forward_python


# embed_dim of the ViT scales in flash_eq_network/test_throughput.py.
VIT_EMBED_DIMS = {"tiny": 384, "small": 480, "base": 768, "large": 1024, "huge": 1280}

# Small shapes that cover divisible, ViT-like, and ragged problems.
KERNEL_SHAPES = [
    (2, 64, 64, 128),
    (2, 197, 192, 576),
    (1, 250, 60, 120),
]

# Module shapes route to old-direct, persistent WMMA, and persistent PTX backends.
MODULE_SHAPES = [
    (2, 64, 64, 128),
    (1, 250, 60, 120),
    (128, 197, 120, 360),
    (128, 197, 192, 576),
    (128, 197, 768, 192),
]

FALLBACK_BACKEND = "pure_cute_fused"
ROUTED_BACKENDS = tuple(dict.fromkeys([*PAPER_EXTENDED_ROUTE_TABLE.values(), FALLBACK_BACKEND]))
# One backend per remaining kernel template, so every modified epilogue runs.
REPRESENTATIVE_BACKENDS = (
    "mixed_cute_fused",
    "pure_cutlass_staged_n64k64",
    "pure_cutlass_staged_gauss_n64k64",
    "pure_cutlass_no_yfreq_materialize",
    "pure_cutlass_direct_spatial",
    "pure_cutlass_spatial_fanout_epilogue",
    "pure_mma_pipe_m64n64",
    "mixed_mma_pipe_v2",
    "pure_mma_pipe_m128n64_x2d",
    "pure_mma_pipe_m128n64_gauss",
    "pure_mma_pipe_cpasync_w",
    "pure_mma_pipe_db",
    "pure_mma_pipe_warpspec",
    "pure_formula_scalar",
    "mixed_formula_scalar",
    "pure_old_m32n32_gauss_smalld",
    "pure_persistent_ptx_m64n64_cta1024",
    "pure_persistent_ptx_dbvec_m64n64_cta1024",
)
RESULT_PREFIX = "RESULT "
BIAS_PROBLEMS = ("bias-fail", "mismatch")


def first_line(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def emit(**result) -> None:
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def print_vit_routes(batch_size: int, seq_len: int) -> None:
    print(f"Routes for Flash EQ-ViT layers at B={batch_size}, L={seq_len}:")
    if not any(backend_available(backend) for backend in VALID_BACKENDS):
        print("  (no fp16 extension is built, so every shape falls back; run python kernels/setup.py)")
    for scale, dim in VIT_EMBED_DIMS.items():
        c = dim // 4
        for name, c_in, c_out in (("qkv", c, 3 * c), ("proj", c, c), ("fc1", c, 4 * c), ("fc2", 4 * c, c)):
            exact = (batch_size, seq_len, c_in, c_out) in PAPER_EXTENDED_ROUTE_TABLE
            backend = route_backend_for_shape(batch_size, seq_len, c_in, c_out)
            print(f"  ViT-{scale:<5} {name:<4} {c_in:>4}->{c_out:<5} {'exact ' if exact else 'bucket'} {backend}")
    print()


def check_one_backend(backend: str) -> None:
    """Child process: check one backend on every kernel shape."""
    device = torch.device("cuda")
    for shape in KERNEL_SHAPES:
        batch, seq, c_in, c_out = shape
        if not backend_available(backend) or not backend_supports_shape(backend, batch * seq, c_in, c_out):
            emit(backend=backend, shape=shape, status="skip", detail="unavailable or unsupported shape")
            continue
        torch.manual_seed(batch * 100003 + seq * 1009 + c_in * 31 + c_out)
        x = torch.randn(batch, seq, c_in, 4, device=device, dtype=torch.float16)
        weight = (torch.randn(c_out, c_in, 4, device=device) / (4 * c_in) ** 0.5).half()
        bias = torch.randn(c_out, 1, device=device, dtype=torch.float16)
        reference = flash_eq_linear_forward_python(x.float(), weight.float())
        try:
            packed = pack_weight_for_backend(weight, backend)
            y = forward_backend(x, packed, backend)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            emit(backend=backend, shape=shape, status="pre-existing", detail=f"bias-free call failed: {first_line(exc)}")
            return  # the CUDA context may be unusable now
        rel_err = (y.float() - reference).abs().max().item() / reference.abs().max().item()
        if rel_err > 5e-2:
            emit(backend=backend, shape=shape, status="pre-existing", detail=f"bias-free rel err {rel_err:.1e}")
            continue
        try:
            y_fused = forward_backend(x, packed, backend, bias)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            emit(backend=backend, shape=shape, status="bias-fail", detail=first_line(exc))
            return
        unfused = y + bias.view(1, 1, c_out, 1)
        if torch.equal(y_fused, unfused):
            emit(backend=backend, shape=shape, status="ok", detail=f"bias-free rel err {rel_err:.1e}")
        else:
            diff = (y_fused.float() - unfused.float()).abs().max().item()
            emit(backend=backend, shape=shape, status="mismatch", detail=f"fused vs y + bias max abs {diff:.3e}")


def check_modules() -> None:
    """Child process: module inference and autograd paths on routed shapes."""
    device = torch.device("cuda")
    for shape in MODULE_SHAPES:
        batch, seq, c_in, c_out = shape
        problems = []
        backend = "?"
        try:
            torch.manual_seed(7)
            layer = CudaFlashEQLinearDirectGemmFp16AutoAda4090(c_in, c_out, bias=True).to(device)
            backend = layer.select_backend_for_shape(batch, seq)
            x = torch.randn(batch, seq, c_in * 4, device=device, dtype=torch.float16)
            packed = pack_weight_for_backend(layer.weights.detach(), backend)
            bias = layer.c.detach().view(1, 1, c_out, 1)
            unfused = (forward_backend(x.view(batch, seq, c_in, 4), packed, backend) + bias).reshape(batch, seq, -1)

            with torch.inference_mode():
                if not torch.equal(layer(x), unfused):
                    problems.append("inference output differs from y + bias")

            x_grad = x.clone().requires_grad_(True)
            y_train = layer(x_grad)
            if not torch.equal(y_train.detach(), unfused):
                problems.append("autograd forward differs from y + bias")
            grad_y = torch.randn_like(y_train)
            y_train.backward(grad_y)
            expected_bias_grad = grad_y.view(batch, seq, c_out, 4).sum(dim=(0, 1, 3)).view(c_out, 1)
            if not torch.equal(layer.c.grad, expected_bias_grad):
                problems.append("bias grad differs from the reduced output grad")
            for name, grad in (("x", x_grad.grad), ("weights", layer.weights.grad)):
                if grad is None or not torch.isfinite(grad.float()).all():
                    problems.append(f"{name} grad is missing or non-finite")
            torch.cuda.synchronize()
        except RuntimeError as exc:
            emit(shape=shape, backend=backend, status="fail", detail=first_line(exc))
            return
        emit(shape=shape, backend=backend, status="fail" if problems else "ok", detail="; ".join(problems))


def run_child(extra_args: list[str], timeout: float) -> tuple[list[dict], str]:
    spec = globals().get("__spec__")
    command = [sys.executable, "-m", spec.name] if spec is not None and spec.name else [sys.executable, os.path.abspath(__file__)]
    try:
        proc = subprocess.run(command + extra_args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"timed out after {timeout:.0f}s"
    results = [json.loads(line[len(RESULT_PREFIX):]) for line in proc.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    error_tail = ""
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        error_tail = f"exit code {proc.returncode}: " + " | ".join(tail)
    return results, error_tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=197)
    parser.add_argument("--all-backends", action="store_true", help="Check every backend in VALID_BACKENDS.")
    parser.add_argument("--backends", default=None, help="Comma-separated backends to check instead of the default set.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Seconds allowed per child process.")
    parser.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--module-checks", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.backend:
        check_one_backend(args.backend)
        return
    if args.module_checks:
        check_modules()
        return

    print_vit_routes(args.batch_size, args.seq_len)
    if not torch.cuda.is_available():
        print("CUDA is unavailable; skipped kernel checks.")
        return

    if args.backends:
        backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    elif args.all_backends:
        backends = list(ROUTED_BACKENDS) + sorted(set(VALID_BACKENDS) - set(ROUTED_BACKENDS))
    else:
        backends = list(dict.fromkeys(ROUTED_BACKENDS + REPRESENTATIVE_BACKENDS))
    unknown = [name for name in backends if name not in VALID_BACKENDS]
    if unknown:
        raise SystemExit(f"Unknown backends: {unknown}")

    shape_labels = [",".join(map(str, shape)) for shape in KERNEL_SHAPES]
    print("Fused bias vs y + bias, one process per backend (routed backends first):")
    print(f"  {'backend':<56} " + " ".join(f"{label:<14}" for label in shape_labels))
    rows = []
    for backend in backends:
        results, error_tail = run_child(["--backend", backend], args.timeout)
        by_shape = {tuple(r["shape"]): r for r in results}
        statuses = []
        for shape in KERNEL_SHAPES:
            result = by_shape.get(shape)
            if result is None:
                result = {"backend": backend, "shape": list(shape), "status": "not-run",
                          "detail": error_tail or "stopped after an earlier failure in this process"}
            rows.append(result)
            statuses.append(result["status"])
        print(f"  {backend:<56} " + " ".join(f"{status:<14}" for status in statuses), flush=True)

    modules, module_error = run_child(["--module-checks"], args.timeout)
    print("\nModule inference and autograd paths:")
    for result in modules:
        line = f"  {str(tuple(result['shape'])):<22} via {result['backend']:<52} {result['status']}"
        print(line + (f"  ({result['detail']})" if result["detail"] else ""))
    if module_error:
        print(f"  module check process: {module_error}")

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("\nSummary:", ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    notes = [row for row in rows if row["status"] not in ("ok", "skip")]
    for row in notes:
        print(f"  [{row['status']}] {row['backend']} {tuple(row['shape'])}: {row['detail']}")

    bias_problems = [row for row in rows if row["status"] in BIAS_PROBLEMS]
    module_problems = [row for row in modules if row["status"] != "ok"] or ([{}] if module_error else [])
    if bias_problems or module_problems:
        print("\nFused-bias or routing problems found.")
        sys.exit(1)
    print("\nNo fused-bias or routing problem found"
          + (" (pre-existing backend failures are listed above)." if any(r["status"] == "pre-existing" for r in rows) else "."))


if __name__ == "__main__":
    main()
