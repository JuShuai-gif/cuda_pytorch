"""RNG case study 3: torch.compile + dropout random semantics.

Companion script for rng/rng.md. Covers:
  1. Compile with dropout: verify randomness
  2. Dropout mask stability under compile
  3. Reproducibility under compile

Run:
    python 03_compile_dropout.py
"""

import sys

import torch


def exp_compile_dropout_randomness():
    print("=" * 60)
    print("1. Verify compile does not cache dropout mask")
    print("=" * 60)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, x):
            return self.dropout(self.linear(x))

    model = Model()
    compiled = torch.compile(model)

    x = torch.randn(4, 16)

    # Multiple calls with same input should produce different outputs
    results = []
    for i in range(5):
        out = compiled(x)
        results.append(out)

    # Check if results differ
    all_same = all(torch.equal(results[0], r) for r in results[1:])
    print(f"  Same input, 5 calls: all same? {all_same}")
    if not all_same:
        # They should NOT be all same (should be some randomness)
        unique = len(set(tuple(r.flatten()[:4].tolist()) for r in results))
        print(f"  Unique first-4 patterns: {unique}/5")
        print(f"  -> Compile correctly pushes RNG forward each call")
    else:
        print(f"  WARNING: dropout produces same mask each call")
        print(f"  -> May indicate compile cached the random op")

    # With manual seeds: should be reproducible
    torch.manual_seed(0)
    r1 = compiled(x.clone())

    torch.manual_seed(0)
    r2 = compiled(x.clone())

    print(f"\n  With same seed: match = {torch.allclose(r1, r2)}")
    print()


def exp_dropout_training():
    print("=" * 60)
    print("2. Training loop with compiled dropout")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )

    compiled_model = torch.compile(model)

    # Training mode: dropout is active
    compiled_model.train()
    x = torch.randn(4, 8)

    # Verify dropout is active (outputs occasionally zero)
    out = compiled_model(x)
    num_zeros = (out == 0).sum().item()
    print(f"  Training mode: {num_zeros} zeros in output / {out.numel()}")
    print(f"  -> Dropout is active in training mode under compile")

    # Eval mode: dropout is inactive
    compiled_model.eval()
    out_eval = compiled_model(x)
    num_zeros_eval = (out_eval == 0).sum().item()
    print(f"  Eval mode: {num_zeros_eval} zeros / {out_eval.numel()}")
    print()


def exp_compile_checkpoint_dropout():
    print("=" * 60)
    print("3. Compile + Checkpoint + Dropout: combined behavior")
    print("=" * 60)

    # This is a known tricky combination
    from torch.utils.checkpoint import checkpoint

    class TrickyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, x):
            return self.dropout(torch.nn.functional.relu(self.linear(x)))

    block = TrickyBlock()
    compiled_block = torch.compile(block)

    x = torch.randn(4, 16, requires_grad=True)

    # Option 1: Compile + checkpoint
    def f(x):
        return compiled_block(x).sum()

    torch.manual_seed(0)
    y1 = f(x.clone().detach().requires_grad_(True))

    torch.manual_seed(0)
    y2 = checkpoint(f, x.clone().detach().requires_grad_(True), use_reentrant=False)

    print(f"  Without checkpoint: {y1.item():.6f}")
    print(f"  With checkpoint: {y2.item():.6f}")
    print(f"  Match: {torch.allclose(y1, y2)}")
    print()


EXPERIMENTS = {
    "randomness": exp_compile_dropout_randomness,
    "training": exp_dropout_training,
    "checkpoint": exp_compile_checkpoint_dropout,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 3] DONE")


if __name__ == "__main__":
    main()
