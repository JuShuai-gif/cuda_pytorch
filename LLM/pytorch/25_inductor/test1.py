"""TorchInductor backend demo: how torch.compile fuses ops into Triton kernels.

Companion script for inductor/README.md.  Demonstrates:
  1. Pointwise fusion:      mul + add + relu -> single Triton kernel
  2. Reduction:             sum along axis in a persistent_reduction kernel
  3. Pointwise + reduction: floor/ceil/add -> sum, fused together
  4. CSE in action:         common subexpressions eliminated

Run with TORCH_LOGS to inspect generated code:

    python test1.py                        # full demo (silent)
    TORCH_LOGS=output_code python test1.py # see the fused Triton kernels
    TORCH_LOGS="output_code,recompiles" python test1.py

    python test1.py pointwise              # only pointwise fusion
    python test1.py reduction              # only reduction
    python test1.py fusion                 # pointwise+reduction fusion
"""

import sys
import time

import torch


# ============ 1. Pointwise fusion: mul + add + relu -> single kernel ============
def exp_pointwise():
    print("=" * 60)
    print("1. Pointwise fusion: mul + add + relu -> single kernel")
    print("=" * 60)

    @torch.compile
    def fn(x):
        a = x * 2  # mul
        b = a + 1  # add
        c = b.relu()  # relu
        return c

    x = torch.randn(1024 * 1024, device="cuda")
    # warmup
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Eager baseline
    def fn_eager(x):
        return (x * 2 + 1).relu()

    for _ in range(3):
        fn_eager(x)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(100):
        fn_eager(x)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"  compiled: {(t1 - t0) * 10:.3f} ms/iter")
    print(f"  eager:    {(t3 - t2) * 10:.3f} ms/iter")
    print(
        "  -> mul, add, relu fused into one Triton kernel (no intermediate VRAM writes)"
    )
    print("     Run: TORCH_LOGS=output_code python test1.py pointwise")
    print()


# ============ 2. Reduction: sum along axis ============
def exp_reduction():
    print("=" * 60)
    print("2. Reduction: sum along last axis")
    print("=" * 60)

    @torch.compile
    def fn(x):
        return x.sum(dim=-1)

    x = torch.randn(32, 512, 1024, device="cuda")

    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    def fn_eager(x):
        return x.sum(dim=-1)

    for _ in range(3):
        fn_eager(x)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(100):
        fn_eager(x)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"  compiled: {(t1 - t0) * 10:.3f} ms/iter")
    print(f"  eager:    {(t3 - t2) * 10:.3f} ms/iter")
    print("  -> Inductor generates a persistent_reduction Triton kernel")
    print("     (rnumel=1024 fits in RBLOCK, no for-loop needed)")
    print()


# ============ 3. Pointwise + Reduction fusion ============
def exp_fusion():
    print("=" * 60)
    print("3. Pointwise + Reduction fusion: floor + ceil + sum")
    print("=" * 60)

    @torch.compile
    def fn(x):
        a = torch.floor(x)  # pointwise
        b = torch.ceil(x)  # pointwise
        c = a + b  # pointwise
        d = c.sum(dim=-1)  # reduction
        return d + 1  # pointwise on reduction output

    x = torch.randn(32, 512, 1024, device="cuda")

    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    def fn_eager(x):
        a = torch.floor(x)
        b = torch.ceil(x)
        c = a + b
        d = c.sum(dim=-1)
        return d + 1

    for _ in range(3):
        fn_eager(x)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(100):
        fn_eager(x)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"  compiled: {(t1 - t0) * 10:.3f} ms/iter")
    print(f"  eager:    {(t3 - t2) * 10:.3f} ms/iter")
    print("  -> floor + ceil + add fused INTO the reduction kernel body")
    print("     (no intermediate floor/ceil/add tensors ever written to VRAM)")
    print("     Run: TORCH_LOGS=output_code python test1.py fusion")
    print()


# ============ 4. CSE: common subexpression elimination ============
def exp_cse():
    print("=" * 60)
    print("4. CSE: common subexpression elimination")
    print("=" * 60)

    @torch.compile
    def fn(x):
        # floor(x) and ceil(x) both load x -> Inductor CSE eliminates duplicate load
        a = torch.floor(x) + torch.ceil(x)  # 2 loads of x -> 1 load after CSE
        b = torch.floor(x) + x  # another floor(x) -> reused
        return a + b

    x = torch.randn(1024 * 1024, device="cuda")

    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f"  compiled: {(t1 - t0) * 10:.3f} ms/iter")
    print("  -> In the generated Triton kernel:")
    print("     x is loaded ONCE (not 3 times)")
    print("     floor(x) is computed ONCE (not twice)")
    print("     CSE deduplicates by expression identity, not variable name")
    print("     Run: TORCH_LOGS=output_code python test1.py cse")
    print()


# ============ 5. Matrix multiply (dot reduction type) ============
def exp_matmul():
    print("=" * 60)
    print("5. Matrix multiply: dot reduction with tiling")
    print("=" * 60)

    @torch.compile
    def fn(a, b):
        return torch.mm(a, b)

    a = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, 128, device="cuda")

    for _ in range(3):
        fn(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        fn(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f"  compiled matmul: {(t1 - t0) * 10:.3f} ms/iter")
    print("  -> Inductor generates a tiled Triton matmul kernel")
    print("     (uses Tensor Cores when available, blocking in M, N, K dimensions)")
    print("     Run: TORCH_LOGS=output_code python test1.py matmul")
    print()


EXPERIMENTS = {
    "pointwise": exp_pointwise,
    "reduction": exp_reduction,
    "fusion": exp_fusion,
    "cse": exp_cse,
    "matmul": exp_matmul,
}


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        return

    torch.cuda.set_device(0)

    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[inductor demo] DONE")


if __name__ == "__main__":
    main()
