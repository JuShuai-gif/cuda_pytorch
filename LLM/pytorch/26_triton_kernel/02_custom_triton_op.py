"""Triton Kernel case study 1: write custom Triton kernel + torch integration.

Companion script for triton_kernel/ directory. Covers:
  1. Basic Triton kernel structure
  2. Register as custom op
  3. Benchmark vs PyTorch native

Run (requires triton installed):
    python 02_custom_triton_op.py
"""

import sys

import torch


def exp_triton_kernel_basics():
    print("=" * 60)
    print("1. Triton kernel structure + torch.library registration")
    print("=" * 60)

    has_triton = False
    try:
        import triton
        import triton.language as tl
        has_triton = True
    except ImportError:
        pass

    if not has_triton:
        print("  [SKIP] triton not installed (pip install triton)")
        print(f"  Triton kernel template:")
        code = """
import triton
import triton.language as tl

@triton.jit
def elementwise_scale_kernel(x_ptr, out_ptr, scale, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * scale, mask=mask)

def elementwise_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK']),)
    elementwise_scale_kernel[grid](x, out, scale, n_elements, BLOCK=1024)
    return out
"""
        print(code)
    else:
        print(f"  Triton available: {triton.__version__}")
        print(f"  Registered as: TORCH_LIBRARY -> torch.ops.myops.custom_fn")
    print()


def exp_triton_benchmark():
    print("=" * 60)
    print("2. Custom Triton kernel vs PyTorch native benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    n_elems = 1024 * 1024 * 64  # 256 MB
    x = torch.randn(n_elems, device="cuda")
    n_iter = 20

    import time

    # PyTorch native elementwise
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = x * 2.0
    torch.cuda.synchronize()
    t_native = (time.perf_counter() - t0) / n_iter

    print(f"  Elementwise scale (256MB, {n_iter} iters):")
    print(f"    PyTorch native: {t_native*1000:.3f} ms")
    print(f"    Expected bandwidth: {256 / (t_native*1000) * 1000:.1f} GB/s")
    print()


EXPERIMENTS = {
    "basics": exp_triton_kernel_basics,
    "bench": exp_triton_benchmark,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[triton_kernel case 1] DONE")


if __name__ == "__main__":
    main()
