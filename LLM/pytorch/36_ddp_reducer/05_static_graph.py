"""DDP Reducer case study 5: static graph and find_unused deep dive.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. static_graph=True optimization
  2. find_unused deep dive with conditionals
  3. DDP vs compile interaction

Run:
    python 05_static_graph.py
"""

import sys

import torch


def exp_static_graph():
    print("=" * 60)
    print("1. static_graph=True optimization")
    print("=" * 60)

    class StaticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
            )

        def forward(self, x):
            return self.net(x)

    model = StaticModel()

    print(f"  static_graph=True: optimizer eliminates:")
    print(f"    1. For-loop to find unused parameters")
    print(f"    2. First-iteration allreduce for unused sync")
    print(f"    3. Extra hook registration for unused detection")
    print(f"")
    print(f"  usage:")
    print(f"    ddp = DDP(model, static_graph=True)")
    print(f"")
    print(f"  Requirement: same set of parameters used EVERY iteration")
    print(f"  If the computation graph changes -> errors")
    print()


def exp_conditional_forward():
    print("=" * 60)
    print("2. Conditional forward: find unused needed")
    print("=" * 60)

    class Conditional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.main = torch.nn.Linear(64, 64)
            self.aux = torch.nn.Linear(64, 64)
            self.head = torch.nn.Linear(64, 10)

        def forward(self, x, use_aux=False):
            h = self.main(x)
            if use_aux:
                h = h + self.aux(x)
            return self.head(h)

    model = Conditional()

    print(f"  With conditional forward:")
    print(f"    f(x, use_aux=False): main and head params used")
    print(f"    f(x, use_aux=True):  all params used")
    print(f"")
    print(f"  DDP options:")
    print(f"    find_unused_parameters=True: handles varying param usage")
    print(f"    static_graph=True: cannot handle (graph changes)")
    print(f"")

    # Show DDP can handle this
    model.use_aux = False
    x = torch.randn(8, 64)
    y1 = model(x)
    y2 = model(x, use_aux=True)
    print(f"  Without aux: output shape={list(y1.shape)}")
    print(f"  With aux:    output shape={list(y2.shape)}")
    print()


def exp_static_vs_dynamic():
    print("=" * 60)
    print("3. DDP static_graph vs dynamic pattern benchmarks")
    print("=" * 60)

    if torch.cuda.is_available():
        model = torch.nn.Sequential(*[torch.nn.Linear(512, 512) for _ in range(10)]).cuda()

        # Measure simple iteration time
        import time
        x = torch.randn(16, 512, device="cuda")

        t0 = time.perf_counter()
        for _ in range(50):
            y = model(x)
            loss = y.sum()
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        print(f"  Single-GPU iteration: {elapsed / 50 * 1000:.2f} ms")
        print(f"  With DDP static_graph=True:")
        print(f"    - Eliminates per-iteration unused param check")
        print(f"    - All parameters used every step -> no re-graph detection")
    else:
        print(f"  [SKIP] CUDA not available")


EXPERIMENTS = {
    "static": exp_static_graph,
    "conditional": exp_conditional_forward,
    "benchmark": exp_static_vs_dynamic,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 5] DONE")


if __name__ == "__main__":
    main()
