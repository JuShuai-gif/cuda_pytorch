"""Device Copy case study 4: CUDA memory allocator interaction with device.

Companion script for device_copy/device_copy.md. Covers:
  1. CUDA caching allocator behavior
  2. CUDA memory pool and device
  3. Memory pinning strategies

Run:
    python 04_allocator_device.py
"""

import sys

import torch


def exp_caching_allocator():
    print("=" * 60)
    print("1. CUDA caching allocator behavior")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    # Observe allocator cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"  Initial:")
    print(f"    allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"    cached:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    x = torch.randn(1024, 1024, device="cuda")
    print(f"\n  After allocating (1x1024x1024 float32 = 4MB):")
    print(f"    allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"    cached:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    del x
    torch.cuda.empty_cache()
    print(f"\n  After del + empty_cache:")
    print(f"    allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"    cached:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print()


def exp_pinned_strategies():
    print("=" * 60)
    print("2. Pinned memory strategies for training pipeline")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    size_mb = 256
    n_elems = size_mb * 1024 * 1024 // 4

    # Monitoring pinned memory usage
    prev_alloc = torch.cuda.memory_allocated()

    # Pinned memory is host-side, not GPU-side
    data_pinned = torch.randn(n_elems, pin_memory=True)
    print(f"  Pinned memory allocated ({size_mb} MB on host)")
    print(f"  GPU memory unchanged: {torch.cuda.memory_allocated() == prev_alloc}")

    # Multiple pinned tensors consume HOST memory
    import psutil
    try:
        host_mem = psutil.Process().memory_info().rss / 1024**3
        print(f"  Host memory usage: {host_mem:.2f} GB")
    except ModuleNotFoundError:
        print(f"  (install psutil to see host memory: pip install psutil)")

    print(f"\n  Pinned memory best practices:")
    print(f"    1. Pin only transfer buffers (DataLoader handles this)")
    print(f"    2. Use pin_memory=True in DataLoader, not manually")
    print(f"    3. Host memory is limited -> too much pinning causes OOM")
    print()


def exp_device_interaction():
    print("=" * 60)
    print("3. Multi-device memory interaction")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    n_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {n_gpus}")

    if n_gpus < 2:
        return

    # Allocate on GPU 0
    with torch.cuda.device(0):
        x0 = torch.randn(1024, 1024, device="cuda")
        mem0 = torch.cuda.memory_allocated(0)

    # Allocate on GPU 1
    with torch.cuda.device(1):
        x1 = torch.randn(1024, 1024, device="cuda")
        mem1 = torch.cuda.memory_allocated(1)

    print(f"  GPU 0 allocated: {mem0 / 1024**2:.1f} MB")
    print(f"  GPU 1 allocated: {mem1 / 1024**2:.1f} MB")

    # Cross-device copy goes through host or P2P
    with torch.cuda.device(1):
        y = x1 + x0.to(1)
        print(f"  Cross-device compute: OK, shape={list(y.shape)}")

    print(f"\n  Cross-device access:")
    print(f"    P2P (NVLink): direct GPU-to-GPU, fast")
    print(f"    No P2P: goes through CPU host memory, slow")
    print()


EXPERIMENTS = {
    "allocator": exp_caching_allocator,
    "pinned": exp_pinned_strategies,
    "device": exp_device_interaction,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 4] DONE")


if __name__ == "__main__":
    main()
