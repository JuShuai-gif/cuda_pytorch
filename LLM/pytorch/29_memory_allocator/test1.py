"""GPU memory management demo: allocated vs reserved, OOM patterns, snapshot.

Companion script for memory_allocator/memory_allocator.md.
  1. allocated vs reserved:  track both during training
  2. peak memory:           find the OOM point
  3. empty_cache:           when it works and when it doesn't
  4. memory snapshot:       diagnose fragmentation
  5. gradient accumulation: simulate large batch on limited VRAM

Run:
    python test1.py                     # full demo (needs CUDA)
    python test1.py basic               # allocated vs reserved
    python test1.py peak                # peak memory tracking
    python test1.py fragmentation       # simulate fragmentation
    python test1.py grad_accumulation   # gradient accumulation
"""

import sys
import torch
import torch.nn as nn


def _cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return False
    return True


# ============ 1. allocated vs reserved ============
def exp_basic():
    if not _cuda():
        return
    print("=" * 60)
    print("1. allocated vs reserved: two key metrics")
    print("=" * 60)

    # 1. Allocate
    x = torch.randn(1024, 1024, device="cuda")  # 4 MB
    alloc = torch.cuda.memory_allocated() / 1e6
    rsvd = torch.cuda.memory_reserved() / 1e6
    print(f"  After 1 tensor: allocated={alloc:.1f} MB  reserved={rsvd:.1f} MB")

    # 2. Free → goes to cache (not cudaFree)
    del x
    alloc_after = torch.cuda.memory_allocated() / 1e6
    rsvd_after = torch.cuda.memory_reserved() / 1e6
    print(
        f"  After del:      allocated={alloc_after:.1f} MB  reserved={rsvd_after:.1f} MB"
    )
    print(f"    allocated → 0 (tensor freed)")
    print(f"    reserved  → stays high (cache pool)")

    # 3. empty_cache
    torch.cuda.empty_cache()
    rsvd_final = torch.cuda.memory_reserved() / 1e6
    print(f"  After empty_cache(): reserved={rsvd_final:.1f} MB")
    print()


# ============ 2. Peak memory tracking ============
def exp_peak():
    if not _cuda():
        return
    print("=" * 60)
    print("2. Peak memory: find the OOM point")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).cuda()

    x = torch.randn(256, 1024, device="cuda")

    y = model(x)
    loss = y.sum()
    loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1e6
    current = torch.cuda.memory_allocated() / 1e6
    print(f"  Current allocated: {current:.1f} MB")
    print(f"  Peak allocated:    {peak:.1f} MB")
    print(f"  Peak happens at backward → stores grad + intermediate activations")

    del model, x, y, loss
    torch.cuda.empty_cache()
    print()


# ============ 3. Fragmentation simulation ============
def exp_fragmentation():
    if not _cuda():
        return
    print("=" * 60)
    print("3. Fragmentation: when reserved > allocated but OOM")
    print("=" * 60)

    # Alternate large and small allocations
    tensors = []
    for _ in range(20):
        tensors.append(torch.randn(64, 1024, 1024, device="cuda"))  # 256 MB
        t = torch.randn(1, 512, device="cuda")
        del t  # small free in between → fragmentation

    alloc = torch.cuda.memory_allocated() / 1e9
    rsvd = torch.cuda.memory_reserved() / 1e9
    cached = rsvd - alloc

    print(f"  allocated: {alloc:.2f} GB")
    print(f"  reserved:  {rsvd:.2f} GB")
    print(f"  cached:    {cached:.2f} GB (fragmented)")
    print(f"  fragmentation ratio: {cached / rsvd * 100:.0f}%")

    # Check segment sizes
    snap = torch.cuda.memory_snapshot()
    segments = snap.get("segments", [])
    sizes = sorted([s["total_size"] / 1e6 for s in segments], reverse=True)
    print(f"\n  Segment sizes (MB): {[f'{s:.0f}' for s in sizes[:8]]}")

    del tensors
    torch.cuda.empty_cache()
    print()


# ============ 4. Gradient accumulation ============
def exp_grad_accumulation():
    if not _cuda():
        return
    print("=" * 60)
    print("4. Gradient accumulation: simulate large batch")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(256, 128).cuda()

    # Target batch = 256, but VRAM only fits 64
    accum_steps = 4
    micro_batch = 64

    torch.cuda.reset_peak_memory_stats()

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x_all = torch.randn(256, 256, device="cuda")
    y_all = torch.randn(256, 128, device="cuda")

    for step in range(accum_steps):
        start = step * micro_batch
        end = start + micro_batch
        x = x_all[start:end]
        y = y_all[start:end]

        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y) / accum_steps
        loss.backward()

    opt.step()
    opt.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1e6
    final_loss = nn.functional.mse_loss(model(x_all), y_all).item()

    print(f"  Simulated batch: 256 (4 × 64)")
    print(f"  Peak memory:     {peak:.1f} MB")
    print(f"  Final loss:      {final_loss:.4f}")
    print(f"  → Each micro-batch has loss/4, gradient accumulates across steps")
    print(f"  → opt.step() only once per accum_steps")
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "peak": exp_peak,
    "fragmentation": exp_fragmentation,
    "grad_accumulation": exp_grad_accumulation,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[memory_allocator demo] DONE")


if __name__ == "__main__":
    main()
