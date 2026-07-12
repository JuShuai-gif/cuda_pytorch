"""Checkpoint advanced: nested, selective, compile interaction.

Companion script for checkpoint/checkpoint.md.
  1. nested checkpoint:      checkpoint inside checkpoint
  2. selective (manual):     only checkpoint expensive ops
  3. compile + checkpoint:   interaction test
  4. memory profiling:       quantify savings per layer

Run:
    python test2.py                  # full demo
    python test2.py nested           # nested checkpoint
    python test2.py selective        # selective checkpointing
    python test2.py compile_ckpt     # compile + checkpoint together
    python test2.py memory           # memory saving quantification
"""

import sys
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ============ 1. Nested checkpoint ============
def exp_nested():
    print("=" * 60)
    print("1. Nested checkpoint: checkpoint inside checkpoint")
    print("=" * 60)

    class NestedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner1 = nn.Linear(128, 128)
            self.inner2 = nn.Linear(128, 128)
            self.outer = nn.Linear(128, 128)

        def inner_fn(self, x):
            x = self.inner1(x).relu()
            x = self.inner2(x).relu()
            return x

        def forward(self, x, use_ckpt=False):
            if use_ckpt:
                x = checkpoint(self.inner_fn, x, use_reentrant=False)
            else:
                x = self.inner_fn(x)
            x = self.outer(x).relu()
            return x

    model = NestedBlock()
    x = torch.randn(32, 128)

    # Outer checkpoint wrapping inner checkpoint
    def outer_fn(x):
        return model(x, use_ckpt=True)

    y_nested = checkpoint(outer_fn, x, use_reentrant=False)
    loss = y_nested.sum()
    loss.backward()

    # Reference: no checkpoint at all
    x2 = x.detach().clone().requires_grad_(True)
    y_ref = model(x2, use_ckpt=False)
    loss2 = y_ref.sum()
    loss2.backward()

    print(f"  Output match:    {torch.allclose(y_nested, y_ref, atol=1e-5)}")
    print(f"  Gradient match:  {torch.allclose(x.grad, x2.grad, atol=1e-5)}")
    print(
        "  -> nested checkpoint: outer saves inner fn input, inner saves its own inputs"
    )
    print("  -> each layer independently discards/reactivates activations")
    print()


# ============ 2. Selective checkpointing ============
def exp_selective():
    print("=" * 60)
    print("2. Selective checkpoint: only expensive layers")
    print("=" * 60)

    class SelectiveNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(64, 64, 3, padding=1)
            self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))

        def forward(self, x, use_ckpt=False):
            # Conv: cheap, don't checkpoint
            x = self.conv(x).relu()

            # Attention: expensive (O(S^2)), checkpoint this
            B, C, H, W = x.shape
            x_seq = x.view(B, C, H * W).transpose(1, 2)  # [B, S, C]
            if use_ckpt:

                def attn_fn(qkv):
                    return self.attn(qkv, qkv, qkv)[0] + qkv

                x_seq = checkpoint(attn_fn, x_seq, use_reentrant=False)
            else:
                x_seq = self.attn(x_seq, x_seq, x_seq)[0] + x_seq

            # FFN: moderate, don't checkpoint
            x_seq = self.ffn(x_seq) + x_seq

            return x_seq

    model = SelectiveNet()
    x = torch.randn(2, 64, 32, 32)

    y = model(x, use_ckpt=True)
    loss = y.sum()
    loss.backward()

    y2 = model(x, use_ckpt=False)
    loss2 = y2.sum()
    loss2.backward()

    print(f"  Output match: {torch.allclose(y, y2, atol=1e-5)}")
    print("  -> Only attention block is checkpointed (O(S^2) activation)")
    print("  -> Conv and FFN activations kept as-is (cheap)")
    print("  -> Balance: save memory on attention, avoid recompute cost on conv")
    print()


# ============ 3. Compile + checkpoint ============
def exp_compile_ckpt():
    print("=" * 60)
    print("3. torch.compile + checkpoint interaction")
    print("=" * 60)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)

        def forward(self, x):
            return self.fc(x).relu()

    model = nn.Sequential(*[Block() for _ in range(4)])

    @torch.compile
    def compiled_forward(x):
        for m in model:
            x = checkpoint(m, x, use_reentrant=False)
        return x

    x = torch.randn(16, 64)
    y = compiled_forward(x)
    loss = y.sum()
    loss.backward()

    # Reference without compile
    x2 = x.detach().clone().requires_grad_(True)
    for m in model:
        x2 = checkpoint(m, x2, use_reentrant=False)
    y2 = x2.sum()
    y2.backward()

    print(f"  Output match:   {torch.allclose(y, y2, atol=1e-5)}")
    print(f"  Grad match:     {torch.allclose(x.grad, x2.grad, atol=1e-5)}")
    print("  -> compile + checkpoint works (torch >= 2.1)")
    print("  -> ensure checkpoint is called INSIDE the compiled function")
    print()


# ============ 4. Memory saving quantification ============
def exp_memory():
    print("=" * 60)
    print("4. Quantify memory savings per layer")
    print("=" * 60)

    class DeepBlock(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.fc = nn.Linear(dim, dim)

        def forward(self, x):
            return self.fc(x).relu()

    num_layers = 8
    model = nn.Sequential(*[DeepBlock(256) for _ in range(num_layers)])

    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(64, 256, device="cuda", requires_grad=True)

        # No checkpoint
        torch.cuda.reset_peak_memory_stats()
        y = model(x)
        loss = y.sum()
        loss.backward()
        mem_no = torch.cuda.max_memory_allocated() / 1e6

        # All layers checkpointed
        torch.cuda.reset_peak_memory_stats()
        x2 = x.detach().clone().requires_grad_(True)

        def run_all(x):
            for m in model:
                x = checkpoint(m, x, use_reentrant=False)
            return x

        y2 = run_all(x2)
        loss2 = y2.sum()
        loss2.backward()
        mem_yes = torch.cuda.max_memory_allocated() / 1e6

        print(f"  Layers: {num_layers} x DeepBlock(256)")
        print(f"  Memory (no ckpt):      {mem_no:.1f} MB")
        print(f"  Memory (all ckpt):     {mem_yes:.1f} MB")
        print(
            f"  Saved:                 {mem_no - mem_yes:.1f} MB ({mem_yes / mem_no * 100:.0f}%)"
        )
        print("  -> each checkpointed layer discards activations during forward")
    else:
        print("  [SKIP] CUDA not available")
    print()


EXPERIMENTS = {
    "nested": exp_nested,
    "selective": exp_selective,
    "compile_ckpt": exp_compile_ckpt,
    "memory": exp_memory,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[checkpoint test2] DONE")


if __name__ == "__main__":
    main()
