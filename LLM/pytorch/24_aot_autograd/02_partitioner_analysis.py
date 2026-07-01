"""AOTAutograd case study 2: partitioner and memory trade-off.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. min_cut partitioner: save vs recompute tradeoff
  2. Activation checkpointing via partitioner
  3. Compare compiled vs eager memory usage

Run:
    python 02_partitioner_analysis.py
"""

import sys

import torch


# A small residual-like model that produces large activations
class SmallResBlock(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden, hidden * 4)
        self.linear2 = torch.nn.Linear(hidden * 4, hidden)

    def forward(self, x):
        y = self.linear1(x)
        y = y.relu()
        y = self.linear2(y)
        return x + y


def exp_compile_modes():
    print("=" * 60)
    print("1. torch.compile modes and partitioner behavior")
    print("=" * 60)

    model = SmallResBlock()

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available, cannot measure memory")
        return

    model = model.cuda()
    x = torch.randn(8, 128, device="cuda", requires_grad=True)

    print(f"  Model: SmallResBlock (hidden=128, expand=512)")
    print(f"  Input: shape={list(x.shape)}")

    # Compare different modes
    modes = ["default", "reduce-overhead"]
    for mode_name in modes:
        try:
            compiled_model = torch.compile(model, mode=mode_name)

            # Reset memory tracking
            torch.cuda.reset_peak_memory_stats()

            y = compiled_model(x)
            loss = y.sum()
            loss.backward()

            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"\n  mode='{mode_name}':")
            print(f"    Peak memory: {peak_mem:.1f} MB")
        except Exception as e:
            print(f"\n  mode='{mode_name}': ERROR {str(e)[:80]}")


def exp_recompute_manual():
    print("\n" + "=" * 60)
    print("2. Manual recompute via torch.utils.checkpoint")
    print("=" * 60)

    from torch.utils.checkpoint import checkpoint

    if not torch.cuda.is_available():
        return

    # Without checkpoint
    model = SmallResBlock().cuda()
    x = torch.randn(32, 128, device="cuda", requires_grad=True)
    x2 = torch.randn(32, 128, device="cuda", requires_grad=True)  # Second input for fairness

    # Option A: no checkpoint
    torch.cuda.reset_peak_memory_stats()
    y1 = model(x)
    loss1 = y1.sum()
    loss1.backward()
    mem_no_ckpt = torch.cuda.max_memory_allocated() / (1024**2)

    # Option B: with checkpoint (recompute in backward)
    torch.cuda.reset_peak_memory_stats()
    y2 = checkpoint(model, x2, use_reentrant=False)
    loss2 = y2.sum()
    loss2.backward()
    mem_with_ckpt = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"  Without checkpoint: peak {mem_no_ckpt:.1f} MB")
    print(f"  With checkpoint:    peak {mem_with_ckpt:.1f} MB")
    if mem_no_ckpt > 0 and mem_with_ckpt > 0:
        print(f"  Memory saved:       {mem_no_ckpt - mem_with_ckpt:.1f} MB")
        print(f"  -> checkpoint saves memory by recomputing activations")


def exp_compiled_vs_checkpoint():
    print("\n" + "=" * 60)
    print("3. torch.compile with built-in checkpoint")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    model = SmallResBlock().cuda()
    x = torch.randn(16, 128, device="cuda", requires_grad=True)

    # torch.compile handles the forward graph
    # Partitioner decides save/recompute internally
    compiled = torch.compile(model, mode="default")

    torch.cuda.reset_peak_memory_stats()
    y = compiled(x)
    loss = y.sum()
    loss.backward()
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  torch.compile default: peak {mem:.1f} MB")
    print(f"  -> compile reuses the fw graph for bw (AOTAutograd)")
    print(f"  -> partitioner internally decides save vs recompute")


EXPERIMENTS = {
    "modes": exp_compile_modes,
    "checkpoint": exp_recompute_manual,
    "compile": exp_compiled_vs_checkpoint,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[aot_autograd case 2] DONE")


if __name__ == "__main__":
    main()
