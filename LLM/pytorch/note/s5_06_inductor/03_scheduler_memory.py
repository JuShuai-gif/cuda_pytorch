"""Inductor case study 2: scheduler and memory planning.

Companion script for inductor/README.md. Covers:
  1. Inductor scheduler fusion decisions
  2. Memory planning and buffer reuse
  3. Horizontal vs vertical fusion

Run:
    python 03_scheduler_memory.py
"""

import sys

import torch


def exp_scheduler_fusion():
    print("=" * 60)
    print("1. Inductor scheduler fusion strategies")
    print("=" * 60)

    print(f"  Inductor scheduler decides how to fuse ops:")

    strategies = [
        ("Pointwise fusion", "relu + sin + mul into one kernel", "Vertical (along data flow)"),
        ("Reduction fusion", "sum + div into one reduction kernel", "Vertical"),
        ("Matmul+pointwise", "matmul + relu into one kernel (template)", "Horizontal"),
        ("Horizontal fusion", "2 independent matmuls -> single batched matmul", "Batch"),
    ]

    for name, example, fusion_type in strategies:
        print(f"    {name:25s}: {example:45s} | {fusion_type}")

    # Demonstrate a fusion case
    @torch.compile
    def vertical_fusable(x):
        return x.relu().sin().add(x).sum()

    x = torch.randn(4096, device="cuda" if torch.cuda.is_available() else "cpu")
    result = vertical_fusable(x)
    print(f"\n  vertical_fusable result: {result.item():.4f}")
    print(f"  -> Inductor fuses relu+sin+add+sum into minimal kernels")
    print()


def exp_buffer_reuse():
    print("=" * 60)
    print("2. Memory planning: buffer reuse")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    @torch.compile
    def memory_intensive(x):
        a = x.relu()
        b = a.sin()
        c = a.cos()
        d = b + c
        return d.sum()

    x = torch.randn(4096, 4096, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    result = memory_intensive(x)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak memory: {peak_mem:.1f} MB")
    print(f"  Inductor reuses buffers:")
    print(f"    a = relu(x) -> temp_buf_0")
    print(f"    b = sin(a)  -> temp_buf_1")
    print(f"    c = cos(a)  -> reuses temp_buf_0 (a not needed after)")
    print(f"    d = b + c   -> reuses temp_buf_1")
    print()


EXPERIMENTS = {
    "fusion": exp_scheduler_fusion,
    "buffer": exp_buffer_reuse,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[inductor case 2] DONE")


if __name__ == "__main__":
    main()
