"""Device Copy case study 1: pinned memory + non_blocking benchmark.

Companion script for device_copy/device_copy.md. Covers:
  1. Pinned vs pageable memory copy speed
  2. non_blocking impact on H2D/D2H
  3. Overlap measurement

Run:
    python 01_pinned_benchmark.py
"""

import sys
import time

import torch


def exp_pinned_vs_pageable():
    print("=" * 60)
    print("1. Pinned vs pageable memory H2D copy speed")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    size_mb = 128
    n_elems = size_mb * 1024 * 1024 // 4  # float32

    # Pageable (default)
    data_pageable = torch.randn(n_elems)

    # Pinned
    data_pinned = torch.randn(n_elems, pin_memory=True)

    n_iter = 20

    # Pageable -> CUDA
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = data_pageable.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
    t_pageable = (time.perf_counter() - t0) / n_iter

    # Pinned -> CUDA
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n_iter):
        _ = data_pinned.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
    t_pinned = (time.perf_counter() - t1) / n_iter

    print(f"  Data size: {size_mb} MB, {n_iter} iterations")
    print(f"  Pageable -> CUDA: {t_pageable*1000:.2f} ms")
    print(f"  Pinned   -> CUDA: {t_pinned*1000:.2f} ms")
    if t_pageable > 0:
        print(f"  Speedup: {t_pageable / t_pinned:.1f}x")

    # non_blocking test
    t2 = time.perf_counter()
    for _ in range(n_iter):
        _ = data_pinned.to("cuda", non_blocking=True)
    t3 = time.perf_counter()
    # Still need to sync to ensure completion
    torch.cuda.synchronize()
    t_nb = (time.perf_counter() - t2) / n_iter

    print(f"\n  Pinned + non_blocking: {t_nb*1000:.2f} ms (CPU launch time)")
    print(f"  -> non_blocking returns immediately, copy happens in background")
    print()


def exp_overlap_measure():
    print("=" * 60)
    print("2. Measure H2D copy + CPU compute overlap")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    data_pinned = torch.randn(64 * 1024 * 1024 // 4, pin_memory=True)  # 64MB

    # Without overlap: sync copy then compute
    t0 = time.perf_counter()
    for _ in range(10):
        gpu_data = data_pinned.to("cuda", non_blocking=False)
        # Some CPU compute (simulated)
        _ = sum(range(100000))
    t_no_overlap = time.perf_counter() - t0

    # With overlap: async copy + CPU compute
    t1 = time.perf_counter()
    for _ in range(10):
        gpu_data = data_pinned.to("cuda", non_blocking=True)
        # CPU work happens WHILE H2D copy is in progress
        _ = sum(range(100000))
    torch.cuda.synchronize()
    t_with_overlap = time.perf_counter() - t1

    print(f"  64MB H2D + CPU work, 10 iterations:")
    print(f"  Without overlap: {t_no_overlap:.3f}s")
    print(f"  With overlap:    {t_with_overlap:.3f}s")
    if t_no_overlap > 0:
        print(f"  Speedup: {t_no_overlap / t_with_overlap:.1f}x")
    print()


def exp_d2h_overlap():
    print("=" * 60)
    print("3. D2H copy overlap")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    gpu_data = torch.randn(64 * 1024 * 1024 // 4, device="cuda")  # 64MB
    pinned_buf = torch.empty_like(gpu_data, pin_memory=True)

    n_iter = 20

    # Sync D2H
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        cpu_data = gpu_data.cpu()
    torch.cuda.synchronize()
    t_sync = (time.perf_counter() - t0) / n_iter

    # Async D2H with pinned memory
    t1 = time.perf_counter()
    for _ in range(n_iter):
        pinned_buf.copy_(gpu_data, non_blocking=True)
    torch.cuda.synchronize()
    t_async = (time.perf_counter() - t1) / n_iter

    print(f"  D2H 64MB, {n_iter} iterations:")
    print(f"  Sync:  {t_sync*1000:.2f} ms")
    print(f"  Async: {t_async*1000:.2f} ms (with pinned memory)")

    # Key insight
    print(f"\n  non_blocking=True works best with:")
    print(f"    1. Source/dest is pinned memory")
    print(f"    2. Copy happens on a non-default CUDA stream")
    print(f"    3. CPU work is done between copy launch and sync")
    print()


EXPERIMENTS = {
    "pinned": exp_pinned_vs_pageable,
    "overlap": exp_overlap_measure,
    "d2h": exp_d2h_overlap,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 1] DONE")


if __name__ == "__main__":
    main()
