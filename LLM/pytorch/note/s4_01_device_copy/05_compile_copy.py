"""Device Copy case study 5: torch.compile device copy optimization.

Companion script for device_copy/device_copy.md. Covers:
  1. Compile with cross-device ops
  2. Copy elimination under compile
  3. Device placement in compiled graph

Run:
    python 05_compile_copy.py
"""

import sys

import torch


def exp_compile_cross_device():
    print("=" * 60)
    print("1. torch.compile with cross-device copy")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    @torch.compile
    def cross_device_fn(x_cpu):
        x_gpu = x_cpu.cuda()
        y = x_gpu.relu()
        return y.cpu().sum()

    x = torch.randn(4, 8)  # CPU
    result = cross_device_fn(x)
    print(f"  cross_device_fn: {result:.4f}")
    print(f"  Compile handles CPU->CUDA->CPU in graph")
    print(f"  D2H/H2D copy is a graph node -> can be reordered by inductor")
    print()


def exp_compile_eliminates_copy():
    print("=" * 60)
    print("2. Compile eliminates redundant copy")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    # Redundant copy pattern
    def fn_with_redundant_copy(x_cuda):
        y = x_cuda + 1       # CUDA
        z = y.cpu()          # CUDA -> CPU (copy 1)
        w = z.cuda()         # CPU -> CUDA (copy 2, redundant!)
        return (w * 2).sum()

    # Eager: two copies happen
    x = torch.randn(4096, 4096, device="cuda")

    compiled = torch.compile(fn_with_redundant_copy)
    result = compiled(x)
    print(f"  Result: {result:.4f}")

    # Dynamo + Inductor may fuse the redundant copy
    # (z.cuda() after y.cpu() is effectively a no-op)
    from torch._dynamo import explain
    try:
        explanation = explain(fn_with_redundant_copy, x)
        gm = explanation.graphs[0]
        for node in gm.graph.nodes:
            if "copy" in str(node.target).lower() or "cpu" in str(node.target).lower():
                print(f"  Graph node: {node.target}")
    except Exception:
        pass
    print()


def exp_compile_device_placement():
    print("=" * 60)
    print("3. Device placement hints for compile")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    # Model components on different devices via compile
    class SplitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(128, 256)

        def forward(self, x):
            # Encoder on GPU 0 (default)
            h = self.encoder(x)
            # Move intermediate to GPU 1
            if torch.cuda.device_count() > 1:
                h = h.to(1)
            return h.sum()

    model = SplitModel()
    compiled = torch.compile(model)

    x = torch.randn(8, 128, device="cuda")
    result = compiled(x)
    print(f"  Split model result: {result:.4f}")
    print(f"\n  Under compile:")
    print(f"    .to(device) becomes a graph node")
    print(f"    Device placement is explicit in the FX graph")
    print()


EXPERIMENTS = {
    "cross": exp_compile_cross_device,
    "eliminate": exp_compile_eliminates_copy,
    "placement": exp_compile_device_placement,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 5] DONE")


if __name__ == "__main__":
    main()
