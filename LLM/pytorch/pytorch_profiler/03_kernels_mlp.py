import argparse
import os
import torch
from kernels import get_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--hidden", type=int, default=3072)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--trace_dir", default="./traces/03_kernels_mlp")
    args = p.parse_args()

    device = "cuda"
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=torch.bfloat16)

    class Config:
        hidden_size = args.dim
        intermediate_size = args.hidden
        hidden_act = "gelu_pytorch_tanh"

    kernels_layers = get_kernel("kernels-community/liger-kernels", version=1).layers
    kernels_geglu_mlp = kernels_layers.LigerGEGLUMLP
    kernels_geglu_mlp = kernels_geglu_mlp(Config()).to(device=device, dtype=torch.bfloat16).eval()

    fwd = torch.compile(kernels_geglu_mlp) if args.compile else kernels_geglu_mlp

    def step():
        with torch.profiler.record_function("kernels_mlp_fwd"), torch.no_grad():
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