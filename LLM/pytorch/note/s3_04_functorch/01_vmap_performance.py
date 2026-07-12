"""functorch case study 1: vmap vs for-loop performance benchmark.

Companion script for functorch/functorch.md. Covers:
  1. Per-sample gradient: for-loop vs vmap
  2. Per-batch inference: for-loop vs vmap
  3. Overhead analysis

Run:
    python 01_vmap_performance.py
"""

import sys
import time

import torch
from torch.func import vmap, grad


def loss_fn(w, x, y):
    return ((x @ w) - y).pow(2).mean()


def exp_per_sample_grad():
    print("=" * 60)
    print("1. Per-sample gradients: for-loop vs vmap")
    print("=" * 60)

    w = torch.randn(8)
    n_samples = 256
    xs = torch.randn(n_samples, 8)
    ys = torch.randn(n_samples)

    # For-loop version
    def per_sample_grad_loop(w, xs, ys):
        grads = []
        for i in range(len(xs)):
            g, = torch.autograd.grad(loss_fn(w, xs[i], ys[i]), w)
            grads.append(g)
        return torch.stack(grads)

    t0 = time.perf_counter()
    g_loop = per_sample_grad_loop(w.detach().requires_grad_(), xs, ys)
    t1 = time.perf_counter()

    # Vmap version
    per_sample_grad_vmap = vmap(grad(loss_fn), in_dims=(None, 0, 0))

    t2 = time.perf_counter()
    g_vmap = per_sample_grad_vmap(w.detach().requires_grad_(), xs, ys)
    t3 = time.perf_counter()

    print(f"  Samples: {n_samples}, weight dim: 8")
    print(f"  For-loop:  {t1 - t0:.4f}s")
    print(f"  Vmap:      {t3 - t2:.4f}s")
    if (t1 - t0) > 0:
        print(f"  Speedup:   {(t1 - t0) / (t3 - t2):.1f}x")

    print(f"  Results match: {torch.allclose(g_loop, g_vmap)}")
    print()


def exp_batch_inference():
    print("=" * 60)
    print("2. Batched model inference: for-loop vs vmap")
    print("=" * 60)

    model = torch.nn.Linear(16, 4)
    n_batches = 128
    batch_size = 32

    xs = torch.randn(n_batches, batch_size, 16)

    # For-loop
    t0 = time.perf_counter()
    results_loop = []
    for i in range(n_batches):
        results_loop.append(model(xs[i]))
    results_loop = torch.cat(results_loop)
    t1 = time.perf_counter()

    # Vmap
    model_vmap = vmap(model)
    t2 = time.perf_counter()
    results_vmap = model_vmap(xs)
    results_vmap = results_vmap.reshape(-1, 4)
    t3 = time.perf_counter()

    print(f"  Batches: {n_batches}, batch_size: {batch_size}")
    print(f"  For-loop: {t1 - t0:.4f}s")
    print(f"  Vmap:     {t3 - t2:.4f}s")
    if (t1 - t0) > 0:
        print(f"  Speedup:  {(t1 - t0) / (t3 - t2):.1f}x")
    print()


def exp_gpu_benchmark():
    print("=" * 60)
    print("3. GPU: vmap advantage grows with batch size")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    def loss_fn_gpu(w, x):
        return (w * x).sum().sin()

    w = torch.randn(128, device="cuda")
    sizes = [32, 128, 512, 2048]

    for n in sizes:
        xs = torch.randn(n, 128, device="cuda")

        # For-loop
        t0 = time.perf_counter()
        for i in range(n):
            loss_fn_gpu(w, xs[i])
        torch.cuda.synchronize()
        t_loop = time.perf_counter() - t0

        # Vmap
        fn = vmap(lambda x: loss_fn_gpu(w, x))
        t1 = time.perf_counter()
        fn(xs)
        torch.cuda.synchronize()
        t_vmap = time.perf_counter() - t1

        speedup = t_loop / t_vmap if t_vmap > 0 else 0
        print(f"  n={n:4d}:  loop={t_loop*1000:.2f}ms  vmap={t_vmap*1000:.2f}ms  speedup={speedup:.1f}x")
    print()


EXPERIMENTS = {
    "grad": exp_per_sample_grad,
    "batch": exp_batch_inference,
    "gpu": exp_gpu_benchmark,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 1] DONE")


if __name__ == "__main__":
    main()
