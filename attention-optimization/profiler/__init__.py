"""Profiler utilities for attention-optimization project.

Provides wrappers for:
- torch.profiler
- NVTX annotations
- Nsight Systems/Compute helpers
"""

import os
import subprocess
from pathlib import Path

PROFILER_DIR = Path(__file__).resolve().parent


def run_nsys_benchmark(script_path: str, output_name: str, extra_args: str = ""):
    """Run a Python script under Nsight Systems profiling."""
    cmd = (
        f"nsys profile --trace=cuda,nvtx,osrt "
        f"-o {PROFILER_DIR / output_name} "
        f"{extra_args} "
        f"python {script_path}"
    )
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)


def run_ncu_benchmark(
    binary_path: str, kernel_name: str, output_name: str, extra_args: str = ""
):
    """Profile a CUDA binary's kernel with Nsight Compute."""
    cmd = (
        f"ncu --kernel-name {kernel_name} --set full "
        f"-o {PROFILER_DIR / output_name} "
        f"{extra_args} "
        f"{binary_path}"
    )
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
