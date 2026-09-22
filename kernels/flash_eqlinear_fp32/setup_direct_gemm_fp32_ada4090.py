import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


nvcc_args = ["-O3", "-use_fast_math", "--expt-relaxed-constexpr"]
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
if not arch_list:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


setup(
    name="flash_EQLinear_cuda_direct_gemm_fp32_ada4090",
    ext_modules=[
        CUDAExtension(
            name="flash_EQLinear_cuda_direct_gemm_fp32_ada4090",
            sources=[str(Path(__file__).resolve().with_name("flash_EQLinear_cuda_direct_gemm_fp32_ada4090.cu"))],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
