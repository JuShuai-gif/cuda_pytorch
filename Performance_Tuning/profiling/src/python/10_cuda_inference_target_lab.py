#!/usr/bin/env python3
"""目标GPU实验：PyTorch CUDA timing、operator、memory、NVTX和长期稳定性。

当前CPU版PyTorch会安全跳过。目标机可直接用torch.profiler、nsys和ncu采集。
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.profiler import ProfilerActivity, profile, record_function
except ImportError:
    print("SKIP: PyTorch未安装")
    raise SystemExit(0)


class TransformerLikeBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.up = nn.Linear(hidden, hidden * 4, bias=False)
        self.down = nn.Linear(hidden * 4, hidden, bias=False)

    def forward(self, x):
        residual = x
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        shape = (x.shape[0], x.shape[1], self.heads, self.head_dim)
        q, k, v = [t.view(shape).transpose(1, 2) for t in (q, k, v)]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = residual + self.out(attn.transpose(1, 2).reshape_as(x))
        h = self.up(self.norm(x))
        return x + self.down(torch.nn.functional.silu(h))


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--sequence", type=int, default=512)
parser.add_argument("--hidden", type=int, default=768)
parser.add_argument("--heads", type=int, default=12)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
parser.add_argument("--csv", default="cuda_inference_samples.csv")
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

if not torch.cuda.is_available():
    print("SKIP: 当前PyTorch没有CUDA支持")
    raise SystemExit(0)
if args.hidden % args.heads:
    raise SystemExit("hidden必须能被heads整除")

dtype = {"fp32": torch.float32, "fp16": torch.float16,
         "bf16": torch.bfloat16}[args.dtype]
device = torch.device("cuda")
torch.manual_seed(7)
model = TransformerLikeBlock(args.hidden, args.heads).to(device=device, dtype=dtype).eval()
x = torch.randn(args.batch, args.sequence, args.hidden, device=device, dtype=dtype)

with torch.inference_mode():
    reference = model(x).float().cpu()
    for _ in range(args.warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for iteration in range(args.iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(f"inference_step/{iteration}")
        start.record()
        output = model(x)
        end.record()
        end.synchronize()
        torch.cuda.nvtx.range_pop()
        samples.append(start.elapsed_time(end))
    torch.testing.assert_close(output.float().cpu(), reference, rtol=3e-3, atol=3e-3)

allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
peak = torch.cuda.max_memory_allocated()
mean = statistics.mean(samples)
print(f"device={torch.cuda.get_device_name()} dtype={args.dtype} shape={tuple(x.shape)}")
print(f"mean={mean:.3f} median={statistics.median(samples):.3f} "
      f"P90={percentile(samples,90):.3f} P95={percentile(samples,95):.3f} "
      f"P99={percentile(samples,99):.3f} min={min(samples):.3f} "
      f"max={max(samples):.3f} stddev={statistics.pstdev(samples):.3f} ms")
print(f"throughput={args.batch*1000/mean:.2f} samples/s "
      f"allocated={allocated} reserved={reserved} peak={peak} correctness=PASS")

with Path(args.csv).open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "latency_ms"])
    writer.writerows(enumerate(samples))

if args.profile:
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True,
                 profile_memory=True, with_stack=True) as prof:
        with torch.inference_mode():
            for _ in range(10):
                with record_function("transformer_like_block"):
                    model(x)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
