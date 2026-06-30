"""Memory allocator advanced: channels_last perf, coalescing, allocation pattern.

Companion script for memory_allocator/memory_allocator.md.
  1. channels_last perf:   NCHW vs NHWC for Conv2d
  2. coalescing demo:      contiguous vs strided memory access
  3. alloc pattern:        small vs large allocation strategies
  4. pin_memory async:     non_blocking DMA transfer timing
  5. reusable buffer:      avoid alloc/free jitter

Run:
    python test2.py                  # full demo
    python test2.py channels         # NHWC vs NCHW
    python test2.py coalescing       # memory coalescing
    python test2.py alloc_pattern    # allocation strategy
    python test2.py pin_async        # async pin_memory + DMA
"""

import sys
import time
import torch
import torch.nn as nn


def _cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return False
    return True


# ============ 1. channels_last (NHWC) ============
def exp_channels():
    if not _cuda():
        return
    print("=" * 60)
    print("1. channels_last (NHWC) vs contiguous (NCHW) for Conv2d")
    print("=" * 60)

    N, C, H, W = 32, 64, 56, 56
    conv = nn.Conv2d(C, C, 3, padding=1).cuda()

    # NCHW
    x_nchw = torch.randn(N, C, H, W, device="cuda").contiguous()
    # NHWC (channels_last)
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)

    print(f"  NCHW stride: {x_nchw.stride()}")
    print(f"  NHWC stride: {x_nhwc.stride()}")

    def bench(tensor, n_iter=100):
        for _ in range(5):
            conv(tensor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            conv(tensor)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

    t_nchw = bench(x_nchw)
    t_nhwc = bench(x_nhwc)

    print(f"\n  NCHW conv: {t_nchw:.3f} ms")
    print(f"  NHWC conv: {t_nhwc:.3f} ms")
    print(f"  speedup:   {t_nchw / t_nhwc:.2f}x")

    # Why NHWC is faster for Conv:
    # NCHW: reading a 3×3 patch across C channels → scattered memory access
    # NHWC: C dimension is innermost → 3×3 patch elements are contiguous in memory
    print(f"\n  → NHWC makes channel dimension contiguous → better coalescing")
    print(f"  → Use for Conv-heavy models (ResNet, EfficientNet)")
    print(f"  → Don't use for Linear-heavy models (1×1 convs don't benefit)")
    print()


# ============ 2. Memory coalescing ============
def exp_coalescing():
    if not _cuda():
        return
    print("=" * 60)
    print("2. Memory coalescing: contiguous vs strided access")
    print("=" * 60)

    N = 1024 * 1024 * 32 // 4  # 32M fp32 elements = 128 MB

    # Contiguous access (coalesced)
    x_contig = torch.randn(N, device="cuda")

    # Strided access (non-coalesced)
    x_strided = torch.randn(N * 8, device="cuda")[::8]  # every 8th element

    def bench_read(tensor, n_iter=50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = tensor * 2
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

    t_contig = bench_read(x_contig)
    t_strided = bench_read(x_strided)

    print(f"  Contiguous (stride=1):   {t_contig:.3f} ms ({N * 4 / 1e6:.0f} MB)")
    print(f"  Strided (stride=8):      {t_strided:.3f} ms ({N * 4 / 1e6:.0f} MB)")
    print(f"  Contiguous speedup:      {t_strided / t_contig:.1f}x")

    s1 = x_contig.stride()[0]
    s2 = x_strided.stride()[0]
    print(f"\n  contiguous stride: {s1}")
    print(f"  strided stride:    {s2}")
    print(f"  → stride=1 → warp 32 threads read contiguous 128B → 1 transaction")
    print(
        f"  → stride=8 → warp 32 threads read 32 scattered locations → 32 transactions"
    )
    print()


# ============ 3. Allocation pattern ============
def exp_alloc_pattern():
    if not _cuda():
        return
    print("=" * 60)
    print("3. Allocation pattern: reuse vs re-allocate")
    print("=" * 60)

    B, D = 256, 1024
    N = 200

    # Strategy A: reuse buffer
    buf = torch.empty(B, D, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        buf.normal_()  # reuse
        y = buf @ buf.T
    torch.cuda.synchronize()
    t_reuse = (time.perf_counter() - t0) * 1000

    # Strategy B: re-allocate each time
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(N):
        x = torch.randn(B, D, device="cuda")
        y = x @ x.T
    torch.cuda.synchronize()
    t_realloc = (time.perf_counter() - t1) * 1000

    print(f"  Reuse buffer:  {t_reuse:.1f} ms ({N} iterations)")
    print(f"  Re-allocate:   {t_realloc:.1f} ms ({N} iterations)")
    print(f"  Overhead/iter: {(t_realloc - t_reuse) / N:.3f} ms")

    # After re-alloc, check cached pool size
    torch.cuda.empty_cache()
    print(f"\n  → Re-alloc overhead = ~{(t_realloc - t_reuse) / N:.3f} ms per iter")
    print(f"  → Caching allocator reduces this, but still has lookup cost")
    print(f"  → For hot loops: pre-allocate and reuse buffers")
    print()


# ============ 4. Pin memory + async DMA ============
def exp_pin_async():
    if not _cuda():
        return
    print("=" * 60)
    print("4. Pin memory + async DMA: non_blocking transfer")
    print("=" * 60)

    N = 32 * 1024 * 1024 // 4  # 32M elements = 128 MB
    x_cpu = torch.randn(N)  # pageable
    x_pinned = x_cpu.pin_memory().clone()  # page-locked (clone breaks storage link)

    # Sync transfer (pageable)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = x_cpu.cuda()  # blocking
    torch.cuda.synchronize()
    t_sync = (time.perf_counter() - t0) * 1000 / 10

    # Async transfer (pinned)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(10):
        _ = x_pinned.cuda(non_blocking=True)
    torch.cuda.synchronize()
    t_async = (time.perf_counter() - t1) * 1000 / 10

    print(f"  Pageable (sync):    {t_sync:.3f} ms/transfer")
    print(f"  Pinned (async DMA): {t_async:.3f} ms/transfer")
    print(f"  speedup:            {t_sync / t_async:.1f}x")
    print(f"\n  → Pageable: driver must first copy to staging buffer → then DMA")
    print(f"  → Pinned:   GPU DMA engine directly accesses CPU memory")
    print(f"  → DataLoader(pin_memory=True) enables this automatically")
    print()


EXPERIMENTS = {
    "channels": exp_channels,
    "coalescing": exp_coalescing,
    "alloc_pattern": exp_alloc_pattern,
    "pin_async": exp_pin_async,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[memory_allocator test2] DONE")


if __name__ == "__main__":
    main()
