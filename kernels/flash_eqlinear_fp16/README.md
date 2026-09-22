# FP16 FlashEQLinear kernels

The FP16 implementation targets Ada GPUs (RTX 4090, CUDA `sm_89`) and cyclic equivariant linear layers with `tranNum=4`. It chooses a CUDA forward backend using the input shape and caches packed weights for inference.

## Build

Requirements: Python 3.10 or later, CUDA-enabled PyTorch, a compatible CUDA Toolkit with `nvcc`, a C++ compiler, and CUTLASS headers. The CuTe backend also uses CUTLASS's `examples/35_gemm_softmax/gemm_with_epilogue_visitor.h`, so keep the complete CUTLASS source tree.

From the repository root, build FP32 and all five FP16 extensions:

```bash
python kernels/setup.py
```

The extensions are written alongside their wrappers in `kernels/flash_eqlinear_fp32/` and `kernels/flash_eqlinear_fp16/`.

To build only FP16:

```bash
cd kernels/flash_eqlinear_fp16
python setup_direct_gemm_fp16_release_ada4090.py build_ext --inplace
```

CUTLASS is searched in `CUTLASS_DIR`, this directory's `cutlass/`, and the repository root's `cutlass/`, in that order. To use another checkout:

```bash
export CUTLASS_DIR=/path/to/cutlass
```

The setup script resolves CUDA sources relative to its own directory. For an individual build, `build/` and the compiled extension output still follow the command's working directory. The Python loader supports extensions built beside the wrapper or installed as top-level modules.

## Usage

Run this example from the repository root after building:

```python
import torch
from kernels.flash_eqlinear_fp16.flash_EQLinear_fp16_direct_gemm_release_ada4090 import (
    create_flash_eqlinear_fp16_direct_gemm_ada4090,
)

B, L, C, D = 32, 128, 64, 64
x = torch.randn(B, L, C * 4, device="cuda", dtype=torch.float16)
spatial_weight = torch.randn(D, C, 4, device="cuda", dtype=torch.float16)
layer = create_flash_eqlinear_fp16_direct_gemm_ada4090(
    C, D, bias=True, eqlinear_weights=spatial_weight,
).cuda().half()

with torch.inference_mode():
    layer.warmup([(B, L, C, D)], iters=5)
    y = layer(x)
print(y.shape)  # (32, 128, 256)
```

The module accepts `(B, L, C * 4)` or `(B, H, W, C * 4)` and returns the corresponding shape with `D * 4` channels. Spatial weights have shape `(D, C, 4)`; the constructor converts them to its frequency-domain parameterization. Optional bias has shape `(D, 1)` and is shared across the four group elements.

The strict factory above raises `KeyError` for shapes outside the 33-entry route table. The class alias `FlashEQLinearFp16DirectGemmAda4090` instead uses `pure_cute_fused` as its default fallback for untuned shapes. All backend shape and alignment constraints still apply.

Training uses the selected forward backend and the shared direct FP16 backward kernel for input and weight gradients. Bias gradients are reduced with PyTorch. Inference with `torch.no_grad()` or `torch.inference_mode()` uses cached packed weights; ordinary in-place parameter updates invalidate that cache.

## Benchmarks and checks

Run from the repository root:

```bash
python -m kernels.flash_eqlinear_fp16.benchmark_direct_gemm_fp16_auto_router_ada4090 --shapes "32,128,64;32,128,256" --out results_fp16/router.json --bias
python -m kernels.flash_eqlinear_fp16.smoke_train_backward_fp16_release
```

Omit `--shapes` to benchmark all 33 routes. Use `--shapes "B,L,C;B,L,C,D"` to select routes. The benchmark reports full module forward time, raw backend time, speedup over the naive FP16 baseline, and output differences. Warmup is excluded from timing. The backward smoke checks finite input, weight, and bias gradients; it does not replace a full numerical accuracy test.

For paper results and operator precision checks, see the [repository README](../../README.md).
