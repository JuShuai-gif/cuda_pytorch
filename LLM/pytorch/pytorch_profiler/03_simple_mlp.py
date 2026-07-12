import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleGeGLUMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = F.gelu(g, approximate="tanh")
        m = h * u
        y = self.down_proj(m)
        return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--hidden", type=int, default=3072)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--trace_dir", default="./traces/03_simple_mlp")
    args = p.parse_args()

    device = "cuda"
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=torch.bfloat16)

    mlp = SimpleGeGLUMLP(args.dim, args.hidden).to(device, dtype=torch.bfloat16)
    mlp.eval()

    fwd = torch.compile(mlp) if args.compile else mlp

    def step():
        with torch.profiler.record_function("mlp_fwd"), torch.no_grad():
            return fwd(x)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    compile_tag = "compile" if args.compile else "eager"
    tag = f"{args.batch}_{args.seq}_{args.dim}_{args.hidden}_{compile_tag}"

    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,  # adds CPU overhead
        profile_memory=False,  # adds CPU overhead
        with_stack=False,
    ) as prof:
        for _ in range(5):
            step()
            prof.step()
    torch.cuda.synchronize()

    print(f"saving traces ... {trace_path}")
    prof.export_chrome_trace(trace_path)

    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()