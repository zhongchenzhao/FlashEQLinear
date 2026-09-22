from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


SOURCE_DIR = Path(__file__).resolve().parent


setup(name='swin_window_process',
    ext_modules=[
        CUDAExtension('swin_window_process', [
            str(SOURCE_DIR / 'swin_window_process.cpp'),
            str(SOURCE_DIR / 'swin_window_process_kernel.cu'),
        ])
    ],
    cmdclass={'build_ext': BuildExtension})