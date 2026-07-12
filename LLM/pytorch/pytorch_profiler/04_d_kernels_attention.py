import argparse
import os
import torch
from kernels import get_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--trace_dir", default="./traces/04_d_flash_kernels")
    args = p.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    # flash-attn expects [batch, seq, heads, head_dim] (seq and heads are
    # swapped compared to SDPA's [batch, heads, seq, head_dim]).
    shape = (args.batch, args.seq, args.heads, args.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    # pre-built, version-pinned FlashAttention kernel from the Hugging Face Hub
    flash = get_kernel("kernels-community/flash-attn", version=1)

    def attn(q, k, v):
        return flash.flash_attn_func(q, k, v, causal=True)

    def step():
        with torch.profiler.record_function("flash_kernel_fwd"), torch.no_grad():
            return attn(q, k, v)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    tag = f"{args.batch}_{args.heads}_{args.seq}_{args.head_dim}_flashattn"

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