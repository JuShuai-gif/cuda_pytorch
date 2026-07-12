import argparse
import os
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# the backends torch.nn.functional.scaled_dot_product_attention can dispatch to
BACKENDS = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--head_dim", type=int, default=64)
    # "auto" lets SDPA pick the backend; the rest force a single backend
    p.add_argument(
        "--backend",
        choices=["auto", "math", "flash", "efficient", "cudnn"],
        default="auto",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--trace_dir", default="./traces/04_c_sdpa_attention")
    args = p.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    shape = (args.batch, args.heads, args.seq, args.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    def attn(q, k, v):
        # is_causal=True asks SDPA to apply the causal mask internally,
        # so we never build or materialize a [seq, seq] mask ourselves.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    fwd = torch.compile(attn) if args.compile else attn

    def step():
        with torch.profiler.record_function("sdpa_fwd"), torch.no_grad():
            if args.backend == "auto":
                return fwd(q, k, v)
            with sdpa_kernel(BACKENDS[args.backend]):
                return fwd(q, k, v)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    compile_tag = "compile" if args.compile else "eager"
    tag = f"{args.batch}_{args.heads}_{args.seq}_{args.head_dim}_{args.backend}_{compile_tag}"

    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,   # adds CPU overhead
        profile_memory=False,  # adds CPU overhead
        with_stack=False,      # adds CPU overhead
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