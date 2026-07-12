"""
Performance tuning demo: apply optimizations step by step and print the
speedup of each, on a small Transformer-ish workload.

Run:
    conda activate torch_env
    python perf_demo.py

Each technique maps to a section in 03_调试与调优手册.md:
    TF32       -> P3   AMP (bf16) -> P1   torch.compile -> P2
"""

import torch
import torch.nn as nn
import torch.utils.benchmark as bench

assert torch.cuda.is_available(), "this demo needs a GPU"
dev = "cuda"


class Block(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(x + self.fc2(torch.relu(self.fc1(x))))


class Net(nn.Module):
    def __init__(self, d=1024, n=6):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(d) for _ in range(n)])

    def forward(self, x):
        return self.blocks(x)


def measure(fn, x, label):
    # benchmark.Timer does warmup + cuda synchronize + robust stats (Playbook rule)
    t = bench.Timer(stmt="fn(x)", globals={"fn": fn, "x": x})
    ms = t.blocked_autorange(min_run_time=2.0).median * 1e3
    print(f"{label:<28} {ms:8.3f} ms")
    return ms


@torch.no_grad()
def main():
    torch.manual_seed(0)
    x = torch.randn(64, 512, 1024, device=dev)
    model = Net().to(dev).eval()

    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")
    print(f"{'variant':<28} {'time':>8}")
    print("-" * 40)

    # 1) baseline fp32
    base = measure(lambda t: model(t), x, "baseline (fp32)")

    # 2) P3: allow TF32 for fp32 matmuls (free speedup on Ampere+)
    torch.set_float32_matmul_precision("high")
    tf32 = measure(lambda t: model(t), x, "+ TF32 matmul")

    # 3) P1: bf16 autocast (Tensor Core)
    def amp_fn(t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(t)

    amp = measure(amp_fn, x, "+ bf16 autocast")

    # 4) P2: torch.compile on top of AMP (GPU backend needs Triton)
    comp = None
    cmodel = torch.compile(model)

    def comp_fn(t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return cmodel(t)

    try:
        comp_fn(x)  # trigger compile before timing
        comp = measure(comp_fn, x, "+ torch.compile")
    except Exception as e:
        print(
            f"+ torch.compile        SKIPPED ({type(e).__name__}: "
            f"pip install triton to enable GPU inductor)"
        )

    print("-" * 40)
    print("speedup vs baseline:")
    results = [("TF32", tf32), ("bf16", amp)]
    if comp is not None:
        results.append(("compile", comp))
    for label, ms in results:
        print(f"  {label:<10} {base / ms:5.2f}x")


if __name__ == "__main__":
    main()
