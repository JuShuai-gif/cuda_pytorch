"""Inductor case study 1: Triton codegen inspection.

Companion script for inductor/README.md. Covers:
  1. Inspect generated Triton kernel code
  2. Understand Inductor IR lowering
  3. Fusion decisions

Run:
    python 02_triton_codegen.py
"""

import sys

import torch


def exp_view_generated_code():
    print("=" * 60)
    print("1. Inspect Inductor-generated Triton code")
    print("=" * 60)

    @torch.compile
    def f(x, y):
        return (x @ y.t()).relu().sum()

    x = torch.randn(4, 8, device="cuda")
    y = torch.randn(4, 8, device="cuda")

    # View generated code
    try:
        import torch._inductor.config as inductor_config
        inductor_config.trace.enabled = True

        result = f(x, y)
        print(f"  Result: {result.item():.4f}")
        print(f"  Generated code in: /tmp/torchinductor_*/")
    except Exception as e:
        print(f"  Error: {str(e)[:80]}")
    print()


def exp_fusion_demo():
    print("=" * 60)
    print("2. Operator fusion under Inductor")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    @torch.compile
    def fused_ops(x):
        return x.relu().sin().mul(2).add(1)

    x = torch.randn(4096, device="cuda")

    import time
    n_iter = 100

    # Warmup
    for _ in range(10):
        fused_ops(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fused_ops(x)
    torch.cuda.synchronize()
    t_compiled = (time.perf_counter() - t0) / n_iter

    # Compare with eager chained ops
    def eager_ops(x):
        return x.relu().sin().mul(2).add(1)

    for _ in range(10):
        eager_ops(x)
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    for _ in range(n_iter):
        eager_ops(x)
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t1) / n_iter

    print(f"  Eager (4 separate kernel launches): {t_eager*1000:.4f} ms")
    print(f"  Compiled (fused single kernel):      {t_compiled*1000:.4f} ms")
    if t_eager > 0:
        print(f"  Speedup: {t_eager/t_compiled:.1f}x")
    print()


EXPERIMENTS = {
    "codegen": exp_view_generated_code,
    "fusion": exp_fusion_demo,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[inductor case 1] DONE")


if __name__ == "__main__":
    main()
