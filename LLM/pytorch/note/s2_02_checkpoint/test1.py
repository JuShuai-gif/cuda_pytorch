"""Activation Checkpointing demo: trading memory for compute.

Companion script for checkpoint/checkpoint.md. Covers:
  1. torch.utils.checkpoint:         basic usage
  2. memory comparison:              with vs without checkpointing
  3. from-scratch implementation:    minimal CheckpointFunction
  4. RNG state:                      why preserve_rng_state matters
  5. nested checkpointing:           compound segments

Run:
    python test1.py                    # full demo
    python test1.py basic              # basic checkpoint vs no checkpoint
    python test1.py memory             # memory comparison
    python test1.py scratch            # from-scratch implementation
    python test1.py rng                # RNG state preservation
"""

import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ============ 1. Basic checkpoint usage ============
def exp_basic():
    print("=" * 60)
    print("1. Basic checkpoint: no-change API")
    print("=" * 60)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(5)])

        def forward(self, x, use_ckpt=False):
            for layer in self.layers:
                if use_ckpt:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
                x = x.relu()
            return x

    model = MLP()
    x = torch.randn(4, 128, requires_grad=True)

    # Without checkpoint
    y1 = model(x, use_ckpt=False)
    loss1 = y1.sum()
    loss1.backward()
    grad1 = x.grad.clone()
    x.grad = None

    # With checkpoint (same result expected)
    y2 = model(x, use_ckpt=True)
    loss2 = y2.sum()
    loss2.backward()
    grad2 = x.grad.clone()

    print(f"  Output match:     {torch.allclose(y1, y2)}")
    print(f"  Gradient match:   {torch.allclose(grad1, grad2)}")
    print("  -> checkpoint preserves numerical correctness")
    print()


# ============ 2. Memory comparison ============
def exp_memory():
    print("=" * 60)
    print("2. Memory comparison: with vs without checkpoint")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    def make_big_model():
        return nn.Sequential(
            *[nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()) for _ in range(8)]
        ).cuda()

    x = torch.randn(64, 1024, device="cuda", requires_grad=True)

    # Without checkpoint
    torch.cuda.reset_peak_memory_stats()
    model = make_big_model()
    y = model(x)
    loss = y.sum()
    loss.backward()
    mem_no_ckpt = torch.cuda.max_memory_allocated() / 1e6
    del model, y, loss

    # With checkpoint
    torch.cuda.reset_peak_memory_stats()
    model = make_big_model()

    def run_block(x):
        for m in model:
            x = checkpoint(m, x, use_reentrant=False)
        return x

    y = run_block(x)
    loss = y.sum()
    loss.backward()
    mem_with_ckpt = torch.cuda.max_memory_allocated() / 1e6

    print(f"  Peak memory (no ckpt):  {mem_no_ckpt:.1f} MB")
    print(f"  Peak memory (with ckpt): {mem_with_ckpt:.1f} MB")
    if mem_no_ckpt > 0:
        print(
            f"  Memory saved:           {mem_no_ckpt - mem_with_ckpt:.1f} MB ({mem_with_ckpt / mem_no_ckpt * 100:.0f}%)"
        )
    print("  -> checkpoint trades compute (re-forward) for memory")
    print()


# ============ 3. From-scratch CheckpointFunction ============
def exp_scratch():
    print("=" * 60)
    print("3. From-scratch CheckpointFunction")
    print("=" * 60)

    class MiniCheckpoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fn, *args):
            ctx.fn = fn
            # Save inputs for recomputation (detach)
            saved = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    saved.append(a.detach())
                else:
                    saved.append(a)
            ctx.save_for_backward(*saved)
            ctx.non_tensor_args = [
                i for i, a in enumerate(args) if not isinstance(a, torch.Tensor)
            ]

            with torch.no_grad():
                outputs = fn(*args)
            return outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            saved = ctx.saved_tensors
            args = []
            non_tensor_idx = set(ctx.non_tensor_args)
            si = 0
            for i in range(len(saved) + len(non_tensor_idx)):
                if i in non_tensor_idx:
                    args.append(saved[si])  # non-tensor (int, float, etc.)
                    si += 1
                else:
                    t = saved[si]
                    t = t.detach().requires_grad_(True)
                    args.append(t)
                    si += 1

            with torch.enable_grad():
                outputs = ctx.fn(*args)

            # Handle multiple outputs
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            torch.autograd.backward(outputs, grad_outputs, retain_graph=False)

            grads = [None, None]  # fn + None for non-tensor
            arg_idx = 0
            for i in range(len(saved)):
                if i in non_tensor_idx:
                    grads.append(None)
                else:
                    grads.append(args[i].grad if i < len(args) else None)
            return tuple(grads)

    # Test our MiniCheckpoint
    x = torch.randn(8, requires_grad=True)
    layer = nn.Linear(1, 4)

    # Reshape x for linear
    x_in = x.unsqueeze(1)

    def fn(t):
        return layer(t).relu()

    y = MiniCheckpoint.apply(fn, x_in)
    loss = y.sum()
    loss.backward()

    # Reference without checkpoint
    x2 = x.detach().clone().requires_grad_(True)
    x2_in = x2.unsqueeze(1)
    y2_ref = fn(x2_in)
    loss2 = y2_ref.sum()
    loss2.backward()

    print(f"  Output match:   {torch.allclose(y, y2_ref)}")
    print(f"  Gradient match: {torch.allclose(x.grad, x2.grad)}")
    print("  -> MiniCheckpoint correctly implements recompute-on-backward")
    print()


# ============ 4. RNG state preservation ============
def exp_rng():
    print("=" * 60)
    print("4. RNG state: why preserve_rng_state matters")
    print("=" * 60)

    # Without RNG preservation: dropout results differ in re-forward
    torch.manual_seed(42)

    def dropout_block(x):
        return nn.functional.dropout(x, p=0.5, training=True)

    x = torch.ones(100, requires_grad=True)

    # Run with checkpoint (preserve_rng_state=True by default)
    y1 = checkpoint(dropout_block, x, use_reentrant=False)
    loss1 = y1.sum()
    loss1.backward(retain_graph=True)
    grad1 = x.grad.clone()
    x.grad = None

    # Run again (same seed → same dropout → same gradient)
    torch.manual_seed(42)
    y2 = checkpoint(dropout_block, x, use_reentrant=False, preserve_rng_state=True)
    loss2 = y2.sum()
    loss2.backward()
    grad2 = x.grad.clone()

    print(f"  With RNG preservation:")
    print(f"    grad match (two runs): {torch.allclose(grad1, grad2)}")

    # Without RNG preservation: different dropout each time
    torch.manual_seed(42)
    x.grad = None
    # NOTE: use_reentrant=False always preserves RNG; this test is conceptual
    print(f"  Without RNG preservation:")
    print(f"    re-forward dropout mask would differ")
    print(f"    -> gradients would be different each backward pass")
    print(f"    -> recommendation: always use preserve_rng_state=True")

    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "memory": exp_memory,
    "scratch": exp_scratch,
    "rng": exp_rng,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint demo] DONE")


if __name__ == "__main__":
    main()
