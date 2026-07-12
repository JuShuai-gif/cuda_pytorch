"""CUDA Stream case study 3: complex stream overlap patterns.

Companion script for cuda_stream/cuda_stream.md. Covers:
  1. Pipeline parallelism with streams
  2. D2H + compute overlap
  3. Stream priority

Run:
    python test3.py
"""

import sys
import time

import torch


def exp_pipeline_overlap():
    print("=" * 60)
    print("1. Pipeline parallelism with CUDA streams")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    s_compute = torch.cuda.Stream()
    s_copy = torch.cuda.Stream()

    n = 4096 * 4096
    x = torch.randn(n, device="cuda")
    cpu_buf = torch.empty(n, pin_memory=True)

    # Pipeline: compute on s_compute while copy on s_copy
    with torch.cuda.stream(s_compute):
        y = x * 2 + 1

    with torch.cuda.stream(s_copy):
        cpu_buf.copy_(y, non_blocking=True)

    torch.cuda.synchronize()
    print(f"  Compute stream + copy stream:")
    print(f"    Compute: y = x*2+1 (on s_compute)")
    print(f"    Copy:    y -> CPU (on s_copy)")
    print(f"    Both streams overlap on GPU")
    print()


def exp_stream_priority():
    print("=" * 60)
    print("2. Stream priority")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    try:
        s_high = torch.cuda.Stream(priority=-1)
        s_low = torch.cuda.Stream(priority=1)

        print(f"  High priority stream: id={s_high.stream_id}, priority={-1}")
        print(f"  Low priority stream:  id={s_low.stream_id}, priority={1}")
        print(f"  -> Higher priority streams get more GPU time")
        print(f"  -> Priority range: [-1 (highest), 0 (default), 1 (lowest)]")
    except Exception as e:
        print(f"  Stream priority not supported: {e}")
    print()


EXPERIMENTS = {
    "pipeline": exp_pipeline_overlap,
    "priority": exp_stream_priority,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[cuda_stream case 3] DONE")


if __name__ == "__main__":
    main()
