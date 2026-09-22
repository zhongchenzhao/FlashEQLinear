import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = os.path.dirname(os.path.abspath(__file__))


def find_cutlass_dir() -> str:
    candidates = [
        os.environ.get("CUTLASS_DIR"),
        os.path.join(HERE, "cutlass"),
        os.path.abspath(os.path.join(HERE, "..", "..", "cutlass")),
    ]
    for candidate in candidates:
        if candidate and all(os.path.isfile(os.path.join(candidate, header)) for header in (
            "include/cute/tensor.hpp",
            "include/cutlass/cutlass.h",
            "examples/35_gemm_softmax/gemm_with_epilogue_visitor.h",
        )):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "CUTLASS headers were not found. Set CUTLASS_DIR or place/copy/symlink "
        f"the CUTLASS tree at {os.path.join(HERE, 'cutlass')}."
    )


CUTLASS_DIR = find_cutlass_dir()

COMMON_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-gencode=arch=compute_89,code=sm_89",
]

COMMON_CXX_FLAGS = ["-O3", "-std=c++17"]


def cuda_extension(name: str, source: str, *, cutlass: bool = False, relaxed_constexpr: bool = False) -> CUDAExtension:
    nvcc_flags = list(COMMON_NVCC_FLAGS)
    if relaxed_constexpr:
        nvcc_flags.append("--expt-relaxed-constexpr")
    include_dirs = []
    if cutlass:
        include_dirs = [
            CUTLASS_DIR,
            os.path.join(CUTLASS_DIR, "include"),
            os.path.join(CUTLASS_DIR, "tools", "util", "include"),
        ]
    return CUDAExtension(
        name=name,
        # Resolve sources from this script; build_ext may run from the repository root.
        sources=[os.path.join(HERE, source)],
        include_dirs=include_dirs,
        extra_compile_args={"cxx": COMMON_CXX_FLAGS, "nvcc": nvcc_flags},
    )


setup(
    name="flash_eqlinear_direct_gemm_fp16_release_ada4090",
    ext_modules=[
        cuda_extension(
            "flash_EQLinear_cuda_direct_gemm_fp16_ada4090",
            "flash_EQLinear_cuda_direct_gemm_fp16_ada4090.cu",
        ),
        cuda_extension(
            "flash_EQLinear_cuda_direct_gemm_fp16_persistent_ada4090",
            "flash_EQLinear_cuda_direct_gemm_fp16_persistent_ada4090.cu",
        ),
        cuda_extension(
            "flash_EQLinear_cuda_direct_gemm_fp16_persistent_ptx_ada4090",
            "flash_EQLinear_cuda_direct_gemm_fp16_persistent_ptx_ada4090.cu",
            relaxed_constexpr=True,
        ),
        cuda_extension(
            "flash_EQLinear_cuda_direct_gemm_fp16_cute_ada4090",
            "flash_EQLinear_cuda_direct_gemm_fp16_cute_ada4090.cu",
            cutlass=True,
            relaxed_constexpr=True,
        ),
        cuda_extension(
            "flash_EQLinear_cuda_direct_gemm_fp16_mma_ada4090",
            "flash_EQLinear_cuda_direct_gemm_fp16_mma_ada4090.cu",
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
