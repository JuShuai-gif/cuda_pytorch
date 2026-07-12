"""Device Copy case study 3: copy_ vs to() semantics and allocation.

Companion script for device_copy/device_copy.md. Covers:
  1. to() creates new tensor, copy_() writes to existing
  2. DataLoader pinned memory pipeline
  3. Allocation patterns

Run:
    python 03_copy_vs_to.py
"""

import sys
import time

import torch


def exp_to_vs_copy():
    print("=" * 60)
    print("1. to() vs copy_(): different semantics")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    x_cpu = torch.randn(256, 256)

    # to(): creates new tensor + copy
    t0 = time.perf_counter()
    for _ in range(100):
        y = x_cpu.to("cuda")
    t_to = (time.perf_counter() - t0) / 100

    # copy_(): writes to pre-allocated tensor
    y_prealloc = torch.empty(256, 256, device="cuda")
    t1 = time.perf_counter()
    for _ in range(100):
        y_prealloc.copy_(x_cpu)
    t_copy = (time.perf_counter() - t1) / 100

    print(f"  to()   (alloc+copy): {t_to*1000:.3f} ms")
    print(f"  copy_() (copy only):  {t_copy*1000:.3f} ms")

    # to() = empty() + copy_() internally
    print(f"\n  to() does:  at::empty() + out.copy_(self)")
    print(f"  copy_() does: direct cudaMemcpy to existing tensor")
    print(f"  -> copy_() is faster when you can reuse the output buffer")
    print()


def exp_pinned_loader_pipeline():
    print("=" * 60)
    print("2. DataLoader pinned memory pipeline")
    print("=" * 60)

    from torch.utils.data import DataLoader, TensorDataset

    n_samples = 200
    n_features = 1024
    data = torch.randn(n_samples, n_features)
    labels = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(data, labels)

    # Without pin_memory
    loader_no_pin = DataLoader(dataset, batch_size=32, shuffle=False)
    if torch.cuda.is_available():
        t0 = time.perf_counter()
        for x, y in loader_no_pin:
            x.cuda()
            y.cuda()
        torch.cuda.synchronize()
        t_no_pin = time.perf_counter() - t0
        print(f"  Without pin_memory:  {t_no_pin:.3f}s")

    # With pin_memory
    loader_pin = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
    if torch.cuda.is_available():
        t1 = time.perf_counter()
        for x, y in loader_pin:
            x.cuda(non_blocking=True)
            y.cuda(non_blocking=True)
        torch.cuda.synchronize()
        t_pin = time.perf_counter() - t1
        print(f"  With pin_memory:     {t_pin:.3f}s")
        if t_no_pin > 0:
            print(f"  Speedup: {t_no_pin / t_pin:.1f}x")

    print(f"\n  pin_memory=True enables:")
    print(f"    1. Page-locked memory -> faster H2D DMA")
    print(f"    2. non_blocking=True -> async copy overlap")
    print()


def exp_same_device_no_copy():
    print("=" * 60)
    print("3. Same device: no copy optimization")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    x = torch.randn(256, 256, device="cuda")

    # Same device: to() returns self (no copy)
    y = x.to("cuda")
    print(f"  Same device to():")
    print(f"    x is y: {x is y}")
    print(f"    same data_ptr: {x.data_ptr() == y.data_ptr()}")

    # Different device: copy (new tensor)
    z = x.to("cpu")
    print(f"\n  Cross device to():")
    print(f"    x is z: {x is z}")
    print(f"    same data_ptr: {x.data_ptr() == z.data_ptr()}")

    # Different dtype: copy
    f = x.to(torch.float64)
    print(f"\n  Different dtype to():")
    print(f"    x is f: {x is f}")
    print(f"    same data_ptr: {x.data_ptr() == f.data_ptr()}")
    print()


EXPERIMENTS = {
    "to_copy": exp_to_vs_copy,
    "pipeline": exp_pinned_loader_pipeline,
    "same": exp_same_device_no_copy,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 3] DONE")


if __name__ == "__main__":
    main()
