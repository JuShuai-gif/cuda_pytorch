"""Module advanced: compile interaction, parameter freezing, dtype casting.

Companion script for module/module.md.
  1. compile + module:       torch.compile with hooks
  2. parameter freezing:     requires_grad=False partial training
  3. dtype casting:          mixed precision per-layer
  4. module introspection:   recursive parameter inspection
  5. weight initialization:  custom init per module type

Run:
    python test2.py                # full demo
    python test2.py compile        # compile + module interaction
    python test2.py freeze         # parameter freezing
    python test2.py dtype          # per-layer dtype casting
    python test2.py inspect        # recursive introspection
    python test2.py init           # custom initialization
"""

import sys
import torch
import torch.nn as nn


# ============ 1. torch.compile + Module hooks ============
def exp_compile():
    print("=" * 60)
    print("1. torch.compile + Module hooks")
    print("=" * 60)

    class HookedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return self.fc(x).relu()

    model = HookedModel()
    hooks_fired = []

    model.register_forward_pre_hook(lambda m, a: hooks_fired.append("pre"))
    model.register_forward_hook(lambda m, a, o: hooks_fired.append("fwd"))

    # Before compile: hooks fire
    model(torch.randn(2, 4))
    print(f"  Without compile: {hooks_fired}")

    # After compile: hooks may or may not fire (depends on torch version)
    hooks_fired.clear()
    compiled = torch.compile(model)
    compiled(torch.randn(2, 4))
    print(f"  With compile:    {hooks_fired}")
    print("  -> torch >= 2.3: hooks work with compile")
    print("  -> check _compiled_call_impl to see if compile path is active")

    # Fullgraph mode
    try:
        compiled_full = torch.compile(model, fullgraph=True)
        compiled_full(torch.randn(2, 4))
        print("  fullgraph=True: OK — no graph breaks")
    except Exception as e:
        print(f"  fullgraph=True: {type(e).__name__}")
    print()


# ============ 2. Parameter freezing ============
def exp_freeze():
    print("=" * 60)
    print("2. Parameter freezing: partial training")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )

    # Freeze first 2 layers
    for name, param in model.named_parameters():
        if "0" in name or "2" in name:
            param.requires_grad_(False)

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    print(f"  Trainable params: {trainable}/{total}")

    # Only pass trainable params to optimizer
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.01
    )
    print(
        f"  Optimizer param_groups[0]['params']: {len(opt.param_groups[0]['params'])}"
    )

    # Verify: trainable params get grad, frozen don't
    x = torch.randn(4, 4)
    loss = model(x).sum()
    loss.backward()

    for name, p in model.named_parameters():
        has_grad = p.grad is not None
        print(f"    {name:20s} requires_grad={p.requires_grad} has_grad={has_grad}")
    print()


# ============ 3. Per-layer dtype casting ============
def exp_dtype():
    print("=" * 60)
    print("3. Per-layer dtype: mixed precision per module")
    print("=" * 60)

    class MixedDtypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 64)
            self.body = nn.Linear(64, 64)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            x = self.embed(x).to(torch.float32)
            x = self.body(x)
            x = self.head(x.to(dtype=self.head.weight.dtype))
            return x

    model = MixedDtypeModel()

    # Set specific dtypes
    model.embed = model.embed.to(dtype=torch.float16)
    model.body = model.body.to(dtype=torch.float32)
    model.head = model.head.to(dtype=torch.bfloat16)

    for name, mod in model.named_children():
        dtype = next(mod.parameters()).dtype
        print(f"  {name:10s} dtype={dtype}")

    # Forward: handles dtype conversion in forward()
    x = torch.randint(0, 100, (4, 8))
    y = model(x)
    print(f"  Output dtype: {y.dtype}")
    print("  -> forward() handles cross-dtype with explicit .to() calls")
    print()


# ============ 4. Recursive module introspection ============
def exp_inspect():
    print("=" * 60)
    print("4. Recursive introspection: params/buffers/hooks dump")
    print("=" * 60)

    model = nn.Sequential(
        nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8)),
        nn.Linear(8, 4),
    )

    def inspect_module(mod, prefix=""):
        info = []
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            n_params = sum(1 for _ in child.parameters(recurse=False))
            n_bufs = sum(1 for _ in child.buffers(recurse=False))
            n_hooks = (
                len(child._forward_hooks)
                + len(child._forward_pre_hooks)
                + len(child._backward_hooks)
            )
            info.append(f"{full:30s} params={n_params} bufs={n_bufs} hooks={n_hooks}")
            info.extend(inspect_module(child, full))
        return info

    for line in inspect_module(model):
        print(f"    {line}")

    # Check frozen params recursively
    model[0][0].weight.requires_grad_(False)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(f"\n  Frozen: {name}")
    print()


# ============ 5. Custom weight initialization ============
def exp_init():
    print("=" * 60)
    print("5. Custom weight initialization by module type")
    print("=" * 60)

    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)

    class InitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvBlock()
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 10)

    model = InitModel()

    # Apply different init per module type
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    for name, p in model.named_parameters():
        if "weight" in name:
            print(f"  {name:25s} mean={p.mean().item():+.3f} std={p.std().item():.3f}")
    print()


EXPERIMENTS = {
    "compile": exp_compile,
    "freeze": exp_freeze,
    "dtype": exp_dtype,
    "inspect": exp_inspect,
    "init": exp_init,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[module test2] DONE")


if __name__ == "__main__":
    main()
