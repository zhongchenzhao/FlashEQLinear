import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILDS = [
    # (setup script, working directory / inplace output directory)
    (
        ROOT / "flash_eqlinear_fp32" / "setup_direct_gemm_fp32_ada4090.py",
        ROOT / "flash_eqlinear_fp32",
    ),
    (
        ROOT / "flash_eqlinear_fp16" / "setup_direct_gemm_fp16_release_ada4090.py",
        ROOT / "flash_eqlinear_fp16",
    ),
]


def run(cmd, cwd):
    print({"cwd": str(cwd), "cmd": cmd})
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    for setup_path, cwd in BUILDS:
        run([sys.executable, str(setup_path), "build_ext", "--inplace"], cwd)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
    print("COMPILATION SUCCESSFUL".center(60))
    print("=" * 60 + "\n")
