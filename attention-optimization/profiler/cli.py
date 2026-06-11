"""Profiler CLI tool for attention-optimization project.

Usage:
    python -m profiler.cli nsys --script src/chapter_03/attention_profile.py
    python -m profiler.cli ncu --binary build/chapter_04/flash_attention_v1 --kernel flash_attention_v1_fwd
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profiler import run_nsys_benchmark, run_ncu_benchmark


def main():
    parser = argparse.ArgumentParser(description="Attention Profiling CLI")
    subparsers = parser.add_subparsers(dest="tool", help="Profiling tool")

    # Nsight Systems
    nsys_parser = subparsers.add_parser("nsys", help="Profile with Nsight Systems")
    nsys_parser.add_argument("--script", required=True, help="Python script to profile")
    nsys_parser.add_argument(
        "--output", default="profile_output", help="Output file name"
    )

    # Nsight Compute
    ncu_parser = subparsers.add_parser("ncu", help="Profile with Nsight Compute")
    ncu_parser.add_argument("--binary", required=True, help="CUDA binary to profile")
    ncu_parser.add_argument("--kernel", required=True, help="Kernel name")
    ncu_parser.add_argument(
        "--output", default="kernel_profile", help="Output file name"
    )

    args = parser.parse_args()

    if args.tool == "nsys":
        run_nsys_benchmark(args.script, args.output)
    elif args.tool == "ncu":
        run_ncu_benchmark(args.binary, args.kernel, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
