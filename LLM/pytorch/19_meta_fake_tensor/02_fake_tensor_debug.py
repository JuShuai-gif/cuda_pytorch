"""Meta Kernel & FakeTensor case study 2: FakeTensorMode for debugging.

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. FakeTensorMode: simulate CUDA ops without GPU
  2. Detect ops lacking meta kernel
  3. Compare FakeTensor vs Meta tensor behavior

Run:
    python 02_fake_tensor_debug.py
"""

import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def exp_faketensor_basics():
    print("=" * 60)
    print("1. FakeTensorMode: trace CUDA ops without GPU")
    print("=" * 60)

    with FakeTensorMode():
        # These tensors are "fake cuda" - no real GPU allocation
        x = torch.randn(4, 8, device="cuda")
        y = torch.randn(8, 3, device="cuda")

        print(f"  x: type={type(x).__name__}, device={x.device}, shape={list(x.shape)}")
        print(f"  x data_ptr: {x.data_ptr()} (0, no real allocation)")

        z = x @ y
        print(f"\n  z = x @ y:")
        print(f"    type={type(z).__name__}")
        print(f"    device={z.device}  (preserved, not 'meta')")
        print(f"    shape={list(z.shape)}")

        # Key difference from Meta tensor:
        # FakeTensor knows the original device info
        w = z.relu().sum()
        print(f"\n  w = z.relu().sum():")
        print(f"    type={type(w).__name__}")
        print(f"    device={w.device}")
    print()


def exp_detect_missing_meta():
    print("=" * 60)
    print("2. Detect missing meta kernel with FakeTensorMode")
    print("=" * 60)

    # Register a custom op WITHOUT meta kernel
    lib = torch.library.Library("fakedemo", "DEF")
    lib.define("my_op(Tensor x) -> Tensor")

    @torch.library.impl("fakedemo::my_op", "CPU")
    def my_op_cpu(x):
        return x * 3

    # Test 1: Op WITH meta kernel (add) -> works
    print(f"  Test 1: aten::add (has meta kernel)")
    with FakeTensorMode():
        x = torch.randn(3, device="cuda")
        y = torch.randn(3, device="cuda")
        try:
            z = x + y
            print(f"    PASS: shape={list(z.shape)}")
        except Exception as e:
            print(f"    FAIL: {str(e)[:80]}")

    # Test 2: Op WITHOUT meta kernel -> fails
    print(f"\n  Test 2: fakedemo::my_op (NO meta kernel)")
    with FakeTensorMode():
        x = torch.randn(3, device="cuda")
        try:
            z = torch.ops.fakedemo.my_op(x)
            print(f"    PASS (unexpected)")
        except Exception as e:
            print(f"    FAIL: {str(e)[:80]}")
            print(f"    -> torch.compile would fail for this op")

    # Test 3: Fix by registering Meta kernel
    print(f"\n  Test 3: Register Meta kernel and retry")
    try:
        @torch.library.impl("fakedemo::my_op", "Meta")
        def my_op_meta(x):
            return x.new_empty(x.shape)

        with FakeTensorMode():
            x = torch.randn(3, device="cuda")
            z = torch.ops.fakedemo.my_op(x)
            print(f"    PASS: shape={list(z.shape)} (Meta kernel enabled)")
    except Exception as e:
        print(f"    Error during registration: {e}")
    print()


def exp_complex_model_trace():
    print("=" * 60)
    print("3. Trace a Transformer block under FakeTensorMode")
    print("=" * 60)

    class TransformerBlock(torch.nn.Module):
        def __init__(self, hidden=256):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.norm1 = torch.nn.LayerNorm(hidden)
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden * 4),
                torch.nn.GELU(),
                torch.nn.Linear(hidden * 4, hidden),
            )
            self.norm2 = torch.nn.LayerNorm(hidden)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x

    model = TransformerBlock()

    with FakeTensorMode():
        x = torch.randn(2, 128, 256, device="cuda")
        try:
            y = model(x)
            print(f"  Input:  shape={list(x.shape)}")
            print(f"  Output: shape={list(y.shape)}")
            print(f"  Device: {y.device}")
            print(f"\n  Transformer block traced successfully!")
            print(f"  All ops have meta kernel support")
        except Exception as e:
            print(f"  Trace FAILED: {str(e)[:120]}")
            print(f"  -> Some op in the model lacks meta kernel")
    print()


EXPERIMENTS = {
    "basics": exp_faketensor_basics,
    "missing": exp_detect_missing_meta,
    "model": exp_complex_model_trace,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 2] DONE")


if __name__ == "__main__":
    main()
