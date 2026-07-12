"""Optimizer demo: SGD, AdamW from scratch + torch.optim comparison.

Companion script for optimizer/optimizer.md. Covers:
  1. SGD from scratch:     plain, momentum, nesterov, weight decay
  2. Adam from scratch:    moment estimates + bias correction
  3. AdamW:                decoupled weight decay
  4. state_dict / load:    optimizer serialization
  5. param_groups:         different learning rates per layer

Run:
    python test1.py              # full demo
    python test1.py sgd          # SGD implementations
    python test1.py adam         # Adam & AdamW
    python test1.py state        # state_dict serialization
    python test1.py groups       # param_groups demo
"""

import sys
import copy

import torch
import torch.nn as nn


# ============ 1. SGD from scratch ============
def exp_sgd():
    print("=" * 60)
    print("1. SGD: plain vs momentum vs nesterov")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = x.pow(3) + 0.3 * torch.randn(100, 1)

    def train_sgd(
        model_class,
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        epochs=100,
        label="",
    ):
        model = model_class()
        params = list(model.parameters())

        # SGD state (from scratch)
        buffers = [torch.zeros_like(p) for p in params]  # momentum buffers

        loss_history = []
        for epoch in range(epochs):
            y_pred = model(x)
            loss = torch.mean((y_pred - y) ** 2)

            # Weight decay before grad calc
            grad_params = torch.autograd.grad(loss, params, create_graph=False)

            for i, (p, g) in enumerate(zip(params, grad_params)):
                if weight_decay != 0:
                    g = g + weight_decay * p.data  # L2 weight decay

                if momentum > 0:
                    buffer = buffers[i]
                    buffer.mul_(momentum).add_(g)  # buf = momentum*buf + grad
                    if nesterov:
                        g = g + momentum * buffer  # Nesterov lookahead
                    else:
                        g = buffer

                with torch.no_grad():
                    p -= lr * g

            loss_history.append(loss.item())

        print(f"  [{label}] final loss: {loss_history[-1]:.6f}")
        return loss_history

    # Test plain SGD
    train_sgd(lambda: nn.Linear(1, 8), lr=0.05, label="plain SGD")

    # Test momentum SGD
    train_sgd(lambda: nn.Linear(1, 8), lr=0.05, momentum=0.9, label="momentum(0.9)")

    # Test nesterov
    train_sgd(
        lambda: nn.Linear(1, 8),
        lr=0.05,
        momentum=0.9,
        nesterov=True,
        label="nesterov(0.9)",
    )

    # Test with weight decay
    train_sgd(
        lambda: nn.Linear(1, 8),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.01,
        label="momentum+wd",
    )

    # Compare with torch.optim.SGD
    model = nn.Linear(1, 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(100):
        y_pred = model(x)
        loss = torch.mean((y_pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  [torch.optim.SGD]     final loss: {losses[-1]:.6f}")
    print()


# ============ 2. Adam from scratch ============
def exp_adam():
    print("=" * 60)
    print("2. Adam & AdamW from scratch")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = x.pow(3) + 0.3 * torch.randn(100, 1)

    def train_adam(
        decoupled_wd=False,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        epochs=200,
        label="",
    ):
        model = nn.Linear(1, 8)
        params = list(model.parameters())

        # Adam state
        m_buffers = [torch.zeros_like(p) for p in params]
        v_buffers = [torch.zeros_like(p) for p in params]
        t = 0

        losses = []
        for epoch in range(epochs):
            t += 1
            y_pred = model(x)
            loss = torch.mean((y_pred - y) ** 2)
            grad_params = torch.autograd.grad(loss, params, create_graph=False)

            beta1, beta2 = betas
            for i, (p, g) in enumerate(zip(params, grad_params)):
                m_buffers[i] = beta1 * m_buffers[i] + (1 - beta1) * g
                v_buffers[i] = beta2 * v_buffers[i] + (1 - beta2) * g * g

                m_hat = m_buffers[i] / (1 - beta1**t)
                v_hat = v_buffers[i] / (1 - beta2**t)

                with torch.no_grad():
                    if decoupled_wd:
                        # AdamW: weight decay decoupled
                        p.mul_(1 - lr * weight_decay)
                        p -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                    else:
                        if weight_decay > 0:
                            g = g + weight_decay * p.data
                        p -= lr * m_hat / (torch.sqrt(v_hat) + eps)

            losses.append(loss.item())

        print(f"  [{label}] final loss: {losses[-1]:.6f}")
        return losses

    train_adam(lr=0.01, label="Adam")
    train_adam(decoupled_wd=True, weight_decay=0.01, label="AdamW(wd=0.01)")

    # Compare with torch.optim
    model = nn.Linear(1, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    losses_ref = []
    for _ in range(200):
        y_pred = model(x)
        loss = torch.mean((y_pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_ref.append(loss.item())
    print(f"  [torch.optim.AdamW]   final loss: {losses_ref[-1]:.6f}")
    print()


# ============ 3. state_dict / load_state_dict ============
def exp_state():
    print("=" * 60)
    print("3. Optimizer state_dict serialization")
    print("=" * 60)

    model = nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Do one step to create momentum buffers
    x = torch.randn(2, 3)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    sd = opt.state_dict()
    print(f"  state_dict keys: {list(sd.keys())}")
    print(f"  param_groups keys: {list(sd['param_groups'][0].keys())}")
    print(f"  lr: {sd['param_groups'][0]['lr']}")
    print(f"  momentum: {sd['param_groups'][0]['momentum']}")
    print(f"  state has {len(sd['state'])} entries")

    # Serialize + restore
    import tempfile, os

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    torch.save(sd, tmp.name)

    model2 = nn.Linear(3, 3)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    sd2 = torch.load(tmp.name, weights_only=True)
    opt2.load_state_dict(sd2)
    os.unlink(tmp.name)

    print(f"\n  After load_state_dict:")
    print(f"    opt2 lr: {opt2.param_groups[0]['lr']}  (restored from 0.01)")
    print(f"    opt2 momentum: {opt2.param_groups[0]['momentum']} (restored)")

    # Check state buffers restored
    for group in opt2.param_groups:
        for p in group["params"]:
            if p in opt2.state:
                st = opt2.state[p]
                print(f"    p.shape={list(p.shape)}, state keys={list(st.keys())}")

    print()


# ============ 4. param_groups ============
def exp_groups():
    print("=" * 60)
    print("4. param_groups: different LR per layer")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = x.pow(3) + 0.3 * torch.randn(100, 1)

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    # Different LRs for different layers
    opt = torch.optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 0.1},  # first layer: higher LR
            {"params": model[2].parameters(), "lr": 0.01},  # middle: medium LR
            {"params": model[4].parameters(), "lr": 0.001},  # last: lower LR
        ]
    )

    print(f"  param_groups: {len(opt.param_groups)}")
    for i, pg in enumerate(opt.param_groups):
        names = []
        for p in pg["params"]:
            for name, param in model.named_parameters():
                if param is p:
                    names.append(name)
        print(f"    group {i}: lr={pg['lr']}, params={names}")

    # Train briefly
    for _ in range(50):
        y_pred = model(x)
        loss = torch.mean((y_pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"\n  Final loss: {loss.item():.6f}")
    print("  -> param_groups enable per-layer hyperparameters")
    print()


EXPERIMENTS = {
    "sgd": exp_sgd,
    "adam": exp_adam,
    "state": exp_state,
    "groups": exp_groups,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[optimizer demo] DONE")


if __name__ == "__main__":
    main()
