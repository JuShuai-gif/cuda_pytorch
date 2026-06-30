"""AMP GradScaler demo: scale loss, unscale, overflow detection.

Companion script for grad_scaler/grad_scaler.md.
  1. basic scaler:          scale → backward → step → update
  2. overflow detection:    create NaN grad, see scaler skip
  3. dynamic scaling:       observe scale grow/shrink
  4. manual unscale:        pitfall demo (double unscale)

Run:
    python test1.py                # full demo (needs CUDA)
    python test1.py basic           # basic scaler workflow
    python test1.py overflow        # overflow detection
    python test1.py dynamic         # dynamic scale tracking
    python test1.py pitfall         # manual unscale pitfall
"""

import sys
import torch
import torch.nn as nn


def _cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return False
    return True


# ============ 1. Basic scaler workflow ============
def exp_basic():
    if not _cuda():
        return
    print("=" * 60)
    print("1. Basic GradScaler: scale → backward → step")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(8, 4).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(init_scale=128)
    x = torch.randn(32, 8, device="cuda", dtype=torch.float16)
    y = torch.randn(32, 4, device="cuda", dtype=torch.float16)

    model = model.half()  # model in fp16

    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)

    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    scaler.step(opt)
    scaler.update()

    print(f"  loss:        {loss.item():.4f}  (fp16)")
    print(f"  scaled_loss: {scaled_loss.item():.1f}  (×{scaler.get_scale():.0f})")
    print(f"  scale after: {scaler.get_scale():.0f}")
    print(f"  model.weight.grad norm: {model.weight.grad.norm().item():.4f}")

    # Verify: grad already unscaled by scaler.step()
    expected_grad_norm = model.weight.grad.norm().item()
    print(
        f"  → scaler.step(opt) does: unscale → check overflow → opt.step → update scale"
    )
    print()


# ============ 2. Overflow detection ============
def exp_overflow():
    if not _cuda():
        return
    print("=" * 60)
    print("2. Overflow detection: NaN grad → skip step")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(8, 4).cuda().half()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(init_scale=65536)
    x = torch.randn(32, 8, device="cuda", dtype=torch.float16)

    weight_before = model.weight.data.clone()

    # Step 1: normal — no overflow
    y = torch.randn(32, 4, device="cuda", dtype=torch.float16)
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    scale_after_normal = scaler.get_scale()
    print(f"  Normal step:   scale={scale_after_normal:.0f}, weight changed=True")

    # Step 2: create overflow by scaling loss HUGE
    weight_before2 = model.weight.data.clone()
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
    mega_loss = loss * 1e10  # force overflow
    scaler.scale(mega_loss).backward()

    scaler.step(opt)
    scaler.update()
    scale_after_overflow = scaler.get_scale()
    weight_changed = not torch.allclose(model.weight.data, weight_before2)

    print(
        f"  Overflow step: scale={scale_after_overflow:.0f}, weight changed={weight_changed}"
    )
    print(f"  → overflow detected → scale reduced ×0.5, step skipped")
    print()


# ============ 3. Dynamic scaling ============
def exp_dynamic():
    if not _cuda():
        return
    print("=" * 60)
    print("3. Dynamic scaling: growth_factor vs backoff_factor")
    print("=" * 60)

    model = nn.Linear(8, 4).cuda().half()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(
        init_scale=128,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=3,
    )
    x = torch.randn(32, 8, device="cuda", dtype=torch.float16)
    y = torch.randn(32, 4, device="cuda", dtype=torch.float16)

    scales = []
    for step in range(6):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scales.append(scaler.get_scale())

    print(f"  Scales across 6 steps: {[int(s) for s in scales]}")
    print(f"  → scale grows ×2 every 3 steps (growth_interval=3)")
    print()


# ============ 4. Manual unscale pitfall ============
def exp_pitfall():
    if not _cuda():
        return
    print("=" * 60)
    print("4. Pitfall: manual unscale after scaler.step → zero grad")
    print("=" * 60)

    model = nn.Linear(8, 4).cuda().half()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(init_scale=1024)
    x = torch.randn(32, 8, device="cuda", dtype=torch.float16)
    y = torch.randn(32, 4, device="cuda", dtype=torch.float16)

    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
    scaler.scale(loss).backward()

    grad_before = model.weight.grad.norm().item()
    print(f"  Grad norm before scaler.step: {grad_before:.4f}")

    scaler.step(opt)  # ← 已 unscale

    # WRONG: manually unscale again!
    scaler.unscale_(opt)
    grad_after_double = model.weight.grad.norm().item()
    print(f"  After scaler.step + manual unscale: {grad_after_double:.4f}")
    print(f"  → gradient divided by scale TWICE → effectively zero!")
    print(f"  → NEVER call scaler.unscale_() after scaler.step()")
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "overflow": exp_overflow,
    "dynamic": exp_dynamic,
    "pitfall": exp_pitfall,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[grad_scaler demo] DONE")


if __name__ == "__main__":
    main()
