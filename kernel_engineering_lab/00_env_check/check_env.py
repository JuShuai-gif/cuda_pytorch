#!/usr/bin/env python3
"""
Environment Check for CUDA/PyTorch/Triton Development.

Run: python check_env.py

This script verifies that all required components for GPU kernel development
are correctly installed and configured. It outputs a formatted report.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional


def get_nvidia_smi_info() -> dict[str, str]:
    """Extract GPU info from nvidia-smi."""
    info: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1:
                    info[f"GPU #{i} Name"] = parts[0]
                if len(parts) >= 2:
                    info[f"GPU #{i} Compute Capability"] = parts[1]
    except Exception:
        pass
    return info


def get_cuda_driver_version() -> Optional[str]:
    """Get CUDA driver version via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def main() -> None:
    print("=" * 60)
    print("  GPU KERNEL ENGINEERING LAB - Environment Check")
    print("=" * 60)
    print()

    # Python
    print(f"{'Python version':<35} {sys.version}")

    # PyTorch
    try:
        import torch

        print(f"{'PyTorch version':<35} {torch.__version__}")
        print(f"{'CUDA available':<35} {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"{'PyTorch CUDA version':<35} {torch.version.cuda}")
            print(f"{'cuDNN version':<35} {torch.backends.cudnn.version()}")
        else:
            print(f"{'PyTorch CUDA version':<35} N/A (CPU-only build)")
    except ImportError as e:
        print(f"{'PyTorch':<35} NOT INSTALLED ({e})")

    # Triton
    try:
        import triton

        print(f"{'Triton version':<35} {triton.__version__}")
    except ImportError:
        print(f"{'Triton version':<35} NOT INSTALLED")

    # CUDA toolkit
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            version_line = lines[-1] if lines else ""
            print(f"{'CUDA Toolkit (nvcc)':<35} {version_line.strip()}")
        else:
            print(f"{'CUDA Toolkit (nvcc)':<35} not found in PATH")
    except FileNotFoundError:
        print(f"{'CUDA Toolkit (nvcc)':<35} nvcc not found in PATH")
    except Exception:
        print(f"{'CUDA Toolkit (nvcc)':<35} could not be detected")

    # Driver version
    driver_ver = get_cuda_driver_version()
    if driver_ver:
        print(f"{'CUDA Driver version':<35} {driver_ver}")

    # GPU details
    gpu_info = get_nvidia_smi_info()
    print()
    print("-" * 60)
    print("  GPU Details")
    print("-" * 60)

    try:
        import torch as _torch

        cuda_available = _torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        num_gpus = _torch.cuda.device_count()
        print(f"\n{'Number of GPUs':<35} {num_gpus}")

        for i in range(num_gpus):
            props = _torch.cuda.get_device_properties(i)
            print(f"\n  GPU #{i}:")
            print(f"  {'Name':<33} {props.name}")
            print(f"  {'Compute Capability':<33} {props.major}.{props.minor}")
            print(f"  {'Total Memory':<33} {props.total_memory / (1024**3):.2f} GB")
            print(f"  {'Multiprocessors':<33} {props.multi_processor_count}")
            print(f"  {'Max Threads per Block':<33} {props.max_threads_per_block}")
            for attr in (
                "max_shared_mem_per_block",
                "shared_memory_per_block",
                "max_shared_memory_per_block",
            ):
                val = getattr(props, attr, None)
                if val is not None:
                    print(f"  {'Max Shared Memory per Block':<33} {val / 1024:.1f} KB")
                    break

            # Memory usage
            try:
                free_bytes, total_bytes = _torch.cuda.mem_get_info(i)
                used_bytes = total_bytes - free_bytes
                print(f"  {'Free Memory':<33} {free_bytes / (1024**3):.2f} GB")
                print(f"  {'Used Memory':<33} {used_bytes / (1024**3):.2f} GB")
            except Exception:
                print(f"  {'Memory Info':<33} failed to query")

        # Quick tensor test
        try:
            x = _torch.randn(1000, 1000, device="cuda")
            y = _torch.randn(1000, 1000, device="cuda")
            z = _torch.matmul(x, y)
            _torch.cuda.synchronize()
            print(f"\n{'Tensor ops test':<35} PASSED (matmul 1000x1000)")
        except Exception as e:
            print(f"\n{'Tensor ops test':<35} FAILED ({e})")
    else:
        print("  No CUDA-capable GPU detected by PyTorch.")
        if gpu_info:
            print("  However, nvidia-smi reports the following GPUs:")
            for k, v in gpu_info.items():
                print(f"  {k}: {v}")

    print()
    print("-" * 60)
    print("  Summary")
    print("-" * 60)

    checks_passed = True

    try:
        import torch  # noqa: F811

        if not torch.cuda.is_available():
            print("  [WARN] CUDA not available in PyTorch. Check your PyTorch installation.")
            checks_passed = False
    except ImportError:
        print("  [FAIL] PyTorch not installed.")
        checks_passed = False

    try:
        import triton  # noqa: F401, F811
    except ImportError:
        print("  [WARN] Triton not installed. Install with: pip install triton")
        checks_passed = False

    if checks_passed:
        print("  [PASS] Environment is ready for GPU kernel development.")
    else:
        print("  Some checks failed. See warnings above.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
