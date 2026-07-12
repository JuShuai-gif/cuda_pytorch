"""
Python-level debugging example.

Goal: step from your own code INTO PyTorch's Python source
(torch/nn/..., torch/overrides.py, autograd, etc).

How to run (see README.md for the full walkthrough):
  1. Open this file in VSCode.
  2. Set a breakpoint on the line marked `# <-- BREAKPOINT`.
  3. Run debug config "Python: Debug Current File".
  4. When paused, use "Step Into (F11)" on `model(x)` and `loss.backward()`.
     Because launch.json sets "justMyCode": false, F11 dives into the
     torch/nn/modules/*.py and autograd Python sources.
"""

import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step Into here (F11) -> torch/nn/modules/linear.py -> F.linear
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h)


def main() -> None:
    torch.manual_seed(0)
    model = TinyNet()
    x = torch.randn(2, 4)
    target = torch.randn(2, 1)

    y = model(x)  # <-- BREAKPOINT (then F11 to step in)
    loss = ((y - target) ** 2).mean()

    # Step Into backward() to explore torch/autograd/__init__.py.
    # The actual graph execution then crosses into C++ (see 02_cpp_debug.py).
    loss.backward()

    print("output:", y.detach().flatten().tolist())
    print("loss:", loss.item())
    print("fc1.weight.grad norm:", model.fc1.weight.grad.norm().item())


if __name__ == "__main__":
    main()
