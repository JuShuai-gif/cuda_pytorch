"""Memory Allocator case study 1: CUDA caching allocator tuning.

Companion script for memory_allocator/ directory. Covers:
  1. CUDA caching allocator internals
  2. Memory fragmentation monitoring
  3. Allocator configuration

Run:
    python 03_allocator_tuning.py
"""

import sys

import torch


def exp_allocator_stats():
    print("=" * 60)
    print("1. CUDA caching allocator statistics")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Allocate progressively
    sizes_mb = [10, 50, 100, 200, 50]
    for size_mb in sizes_mb:
        n_elems = size_mb * 1024 * 1024 // 4
        _ = torch.randn(n_elems, device="cuda")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Allocated {size_mb:3d}MB: allocated={allocated:.0f}MB, reserved(cached)={reserved:.0f}MB")

    torch.cuda.empty_cache()
    reserved_after = torch.cuda.memory_reserved() / 1024**2
    print(f"\n  After empty_cache(): allocated={torch.cuda.memory_allocated()/1024**2:.0f}MB, reserved={reserved_after:.0f}MB")
    print(f"  -> empty_cache releases cached but not allocated memory")
    print()


def exp_fragmentation():
    print("=" * 60)
    print("2. Memory fragmentation simulation")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tensors = []
    # Interleave large and small allocations
    for i in range(20):
        if i % 2 == 0:
            tensors.append(torch.randn(1024, 1024, device="cuda"))  # 4MB
        else:
            tensors.append(torch.randn(128, 128, device="cuda"))    # 64KB

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  After 20 allocations:  allocated={allocated:.1f}MB, reserved={reserved:.1f}MB, peak={peak:.1f}MB")
    print(f"  Fragmentation ratio: {reserved/allocated:.2f}x (reserved/allocated)")

    # Delete alternating tensors -> create holes
    for i in range(0, len(tensors), 2):
        del tensors[i]
    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated() / 1024**2
    reserved_after = torch.cuda.memory_reserved() / 1024**2
    print(f"  After deleting half: allocated={allocated_after:.1f}MB, reserved={reserved_after:.1f}MB")
    print()


def exp_allocator_config():
    print("=" * 60)
    print("3. Allocator configuration options")
    print("=" * 60)

    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,roundup_power2_divisions:16",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1 (disable caching, debug only)",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True": "Allow segments to expand (PyTorch >= 2.0)",
    }

    for var, desc in env_vars.items():
        print(f"  {var}")
        print(f"    {desc}")

    print(f"\n  Common tunings:")
    print(f"    max_split_size_mb: limits block splitting (reduces fragmentation)")
    print(f"    garbage_collection_threshold: triggers GC at memory threshold")
    print(f"    expandable_segments:True -> avoids OOM in dynamic workloads")
    print()


EXPERIMENTS = {
    "stats": exp_allocator_stats,
    "frag": exp_fragmentation,
    "config": exp_allocator_config,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[memory_allocator case 1] DONE")


if __name__ == "__main__":
    main()
