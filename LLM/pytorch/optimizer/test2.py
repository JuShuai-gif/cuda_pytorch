"""Optimizer advanced scenarios: lr scheduling, gradient clipping, resume.

Companion script for optimizer/optimizer.md.
  1. lr_scheduler:    StepLR, CosineAnnealingLR, Warmup
  2. grad_clipping:   clip_grad_norm_ vs clip_grad_value_
  3. checkpoint resume: save/restore optimizer + scheduler state
  4. weight_decay tuning: L2 vs decoupled in practice

Run:
    python test2.py               # full demo
    python test2.py lr            # learning rate scheduling
    python test2.py clip          # gradient clipping
    python test2.py resume        # checkpoint save/restore
    python test2.py wd            # weight decay comparison
"""

import sys
import tempfile
import os

import torch
import torch.nn as nn
import torch.optim as optim


# ============ 1. LR scheduling ============
def exp_lr():
    print("=" * 60)
    print("1. Learning rate scheduling")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(8, 4)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)

    # Test 3 scheduler types
    schedulers = {
        "StepLR(step=5, gamma=0.5)": optim.lr_scheduler.StepLR(
            opt, step_size=5, gamma=0.5
        ),
        "CosineAnnealing(T_max=20)": optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=20, eta_min=0.001
        ),
        "OneCycleLR(max=0.1)": optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, steps_per_epoch=1, epochs=20
        ),
    }

    for name, sched in schedulers.items():
        opt.param_groups[0]["lr"] = 0.1
        lrs = []
        for _ in range(20):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        print(f"  {name:30s}: lr={lrs[0]:.4f} -> {lrs[-1]:.4f}")

    # Warmup + cosine
    opt.param_groups[0]["lr"] = 0.001
    warmup_steps = 5
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=0.0001)
    lrs = []
    for step in range(20):
        if step < warmup_steps:
            opt.param_groups[0]["lr"] = 0.001 * (step + 1) / warmup_steps
        else:
            cosine.step()
        lrs.append(opt.param_groups[0]["lr"])
    print(
        f"\n  Warmup(5) + Cosine: lr={lrs[0]:.4f} -> peak={max(lrs):.4f} -> {lrs[-1]:.6f}"
    )
    print()


# ============ 2. Gradient clipping ============
def exp_clip():
    print("=" * 60)
    print("2. Gradient clipping: norm vs value")
    print("=" * 60)

    class SpikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            return self.fc(x) * 100  # amplify gradients

    torch.manual_seed(42)
    model = SpikeModel()
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)

    # Before clipping
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    grad_norm_before = sum(p.grad.norm().item() for p in model.parameters())
    model.zero_grad()

    # clip_grad_norm_ — scale all gradients to have total norm <= max_norm
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm_clipped = sum(p.grad.norm().item() for p in model.parameters())

    print(f"  grad norm before:  {grad_norm_before:.2f}")
    print(f"  after clip(1.0):   {grad_norm_clipped:.4f}  (<= 1.0)")
    model.zero_grad()

    # clip_grad_value_ — clamp each grad element to [-v, v]
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    original_max = max(p.grad.abs().max().item() for p in model.parameters())
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    after_max = max(p.grad.abs().max().item() for p in model.parameters())

    print(f"\n  grad max before:   {original_max:.2f}")
    print(f"  after clip(0.5):   {after_max:.4f}  (<= 0.5)")
    print(
        "  -> clip_grad_norm_ for global decay, clip_grad_value_ for individual element cap"
    )
    print()


# ============ 3. Checkpoint resume ============
def exp_resume():
    print("=" * 60)
    print("3. Checkpoint save/restore: model + optimizer + scheduler")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    opt = optim.AdamW(model.parameters(), lr=0.01)
    sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    # Train 5 steps
    for step in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        sched.step()

    lr_before = opt.param_groups[0]["lr"]
    step_before = list(opt.state.values())[0].get("step", 0)
    print(f"  Before save: lr={lr_before}, step={step_before}")

    # Save
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": 5,
        },
        tmp.name,
    )

    # Create new model + optimizer + scheduler, then restore
    model2 = nn.Linear(4, 2)
    opt2 = optim.AdamW(model2.parameters(), lr=0.1)  # different initial lr
    sched2 = optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.1)
    ckpt = torch.load(tmp.name, weights_only=False)
    model2.load_state_dict(ckpt["model"])
    opt2.load_state_dict(ckpt["opt"])
    sched2.load_state_dict(ckpt["sched"])

    lr_after = opt2.param_groups[0]["lr"]
    step_after = list(opt2.state.values())[0].get("step", 0)
    print(f"  After restore: lr={lr_after}, step={step_after}")
    print(f"  LR match:   {lr_before == lr_after}")
    print(f"  Step match: {step_before == step_after}")

    os.unlink(tmp.name)
    print()


# ============ 4. Weight decay comparison ============
def exp_wd():
    print("=" * 60)
    print("4. Weight decay: AdamW decoupled vs Adam L2")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(128, 8)
    y = torch.randn(128, 4)

    def train_and_get_norm(opt_class, wd):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        opt = opt_class(model.parameters(), lr=0.01, weight_decay=wd)
        for _ in range(100):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()
        return sum(p.data.norm().item() for p in model.parameters())

    norm_no_wd = train_and_get_norm(optim.AdamW, 0.0)
    norm_wd_01 = train_and_get_norm(optim.AdamW, 0.1)
    norm_wd_001 = train_and_get_norm(optim.AdamW, 0.01)

    print(f"  AdamW wd=0.0:  param norm = {norm_no_wd:.2f}")
    print(f"  AdamW wd=0.1:  param norm = {norm_wd_01:.2f}  (strongly regularized)")
    print(f"  AdamW wd=0.01: param norm = {norm_wd_001:.2f}")
    print("  -> Larger wd = smaller weights (decoupled: p *= 1 - lr*wd)")
    print()


EXPERIMENTS = {
    "lr": exp_lr,
    "clip": exp_clip,
    "resume": exp_resume,
    "wd": exp_wd,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[optimizer test2] DONE")


if __name__ == "__main__":
    main()
