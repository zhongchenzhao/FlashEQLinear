# -*- coding: utf-8 -*-
"""Smoke-test the release routed autograd path.

This script intentionally keeps the shapes small but covers both an old-direct
route and a non-old routed branch. It verifies that module and functional
interfaces produce finite gradients for x, weights, and bias.
"""

from __future__ import annotations

import argparse

import torch

if __package__:
    from .flash_EQLinear_fp16_direct_gemm_release_ada4090 import (
        create_flash_eqlinear_fp16_direct_gemm_ada4090,
        flash_eq_linear_direct_gemm_fp16_routed,
        route_backend_for_shape,
    )
else:
    from flash_EQLinear_fp16_direct_gemm_release_ada4090 import (
        create_flash_eqlinear_fp16_direct_gemm_ada4090,
        flash_eq_linear_direct_gemm_fp16_routed,
        route_backend_for_shape,
    )


DEFAULT_SHAPES = [
    (32, 128, 64, 64),
    (32, 128, 256, 256),
]


def parse_shapes(value: str | None) -> list[tuple[int, int, int, int]]:
    if not value:
        return DEFAULT_SHAPES
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


def assert_finite_grad(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        raise AssertionError(f"{name} is None")
    if not torch.isfinite(tensor.float()).all().item():
        raise AssertionError(f"{name} contains non-finite values")
    print(f"{name}: shape={tuple(tensor.shape)} mean_abs={tensor.float().abs().mean().item():.6e}")


def run_module_case(shape: tuple[int, int, int, int]) -> None:
    batch, seqlen, c_in, c_out = shape
    backend = route_backend_for_shape(batch, seqlen, c_in, c_out, fallback_backend=None)
    torch.manual_seed(batch * 1000000 + seqlen * 1000 + c_in + c_out)
    x = torch.randn(batch, seqlen, c_in * 4, device="cuda", dtype=torch.float16, requires_grad=True)
    weight_spatial = torch.randn(c_out, c_in, 4, device="cuda", dtype=torch.float16)
    layer = create_flash_eqlinear_fp16_direct_gemm_ada4090(
        c_in,
        c_out,
        bias=True,
        eqlinear_weights=weight_spatial,
    ).cuda().half()
    y = layer(x)
    loss = y.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"MODULE shape={shape} backend={backend} y_shape={tuple(y.shape)} loss={loss.item():.6e}")
    assert_finite_grad("module x.grad", x.grad)
    assert_finite_grad("module weights.grad", layer.weights.grad)
    assert_finite_grad("module bias.grad", layer.c.grad)


def run_functional_case(shape: tuple[int, int, int, int]) -> None:
    batch, seqlen, c_in, c_out = shape
    backend = route_backend_for_shape(batch, seqlen, c_in, c_out, fallback_backend=None)
    torch.manual_seed(123)
    x = torch.randn(batch, seqlen, c_in, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(c_out, c_in, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(c_out, 1, device="cuda", dtype=torch.float16, requires_grad=True)
    y = flash_eq_linear_direct_gemm_fp16_routed(x, weight, bias, backend)
    loss = y.float().abs().mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"FUNCTIONAL shape={shape} backend={backend} y_shape={tuple(y.shape)} loss={loss.item():.6e}")
    assert_finite_grad("functional x.grad", x.grad)
    assert_finite_grad("functional weight.grad", weight.grad)
    assert_finite_grad("functional bias.grad", bias.grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=None)
    args = parser.parse_args()

    shapes = parse_shapes(args.shapes)
    for shape in shapes:
        run_module_case(shape)
    run_functional_case(shapes[-1])
    print("backward smoke passed")


if __name__ == "__main__":
    main()
