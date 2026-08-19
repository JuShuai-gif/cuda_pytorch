"""Final Project A: GPU inference optimization staircase.

Takes one model through the optimization ladder and records, at each step, the
full metric set (latency p50/p95/p99, GPU memory, kernel count), so the report
is a Before/After table, not a single "faster" claim.

Ladder:
  eager fp32 -> torch.compile -> eager fp16 -> Triton fused RMSNorm
  -> CUDA Graph -> (TensorRT FP16, from Stage 7, reported separately)

The model is a residual MLP (LayerNorm + Linear + GELU), the same family used
throughout the repo, so every stage's techniques apply.
"""

from __future__ import annotations

import torch
from torch import nn

import kernel.triton  # noqa: F401  (TRITON_PTXAS_BLACKWELL_PATH)
from common.env import collect_environment, resolve_device
from common.measure import cuda_event_latency
from kernel.triton.operators.rmsnorm import triton_rmsnorm


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, use_rmsnorm: bool = False):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.norm = None
            self.g = nn.Parameter(torch.ones(hidden))
        else:
            self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        if self.use_rmsnorm:
            b, s, h = x.shape
            h_norm = triton_rmsnorm(x.reshape(-1, h), self.g, 1e-5).reshape(b, s, h)
        else:
            h_norm = self.norm(x)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(h_norm)))


def make_model(hidden: int, layers: int, use_rmsnorm: bool = False) -> nn.Module:
    return nn.Sequential(*[ResidualBlock(hidden, use_rmsnorm) for _ in range(layers)])


def measure(model, x, device, warmup=20, iterations=100, repeats=3) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize(device)

    # Multiple repeats to damp clock/thermal jitter; report the median p50.
    p50s, p95s, p99s = [], [], []
    for _ in range(repeats):
        with torch.no_grad():
            t = cuda_event_latency(lambda: model(x), device=device, warmup=0,
                                   iterations=iterations)
        p50s.append(t.p50)
        p95s.append(t.p95)
        p99s.append(t.p99)
    p50s.sort(); p95s.sort(); p99s.sort()
    p50 = p50s[len(p50s) // 2]
    p95 = p95s[len(p95s) // 2]
    p99 = p99s[len(p99s) // 2]

    # Kernel count via torch profiler (one pass).
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize(device)
    n_kernels = sum(e.count for e in prof.key_averages()
                    if e.device_type == torch.autograd.DeviceType.CUDA)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize(device)

    return {
        "latency_us_p50": p50,
        "latency_us_p95": p95,
        "latency_us_p99": p99,
        "gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6,
        "kernel_count": n_kernels,
    }


def build_cuda_graph(model, x):
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.no_grad():
            model(x)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(g):
        with torch.no_grad():
            model(x)
    return g


def run_staircase(device, hidden=1024, layers=4, batch=1, seq=16) -> list[dict]:
    results = []

    # 1. eager fp32
    m = make_model(hidden, layers).to(device)
    x = torch.randn(batch, seq, hidden, device=device)
    results.append({"stage": "eager_fp32", **measure(m, x, device)})

    # 2. torch.compile
    mc = torch.compile(m)
    with torch.no_grad():
        mc(x)
    results.append({"stage": "torch_compile_fp32", **measure(mc, x, device)})

    # 3. eager fp16
    m16 = m.half()
    x16 = x.half()
    results.append({"stage": "eager_fp16", **measure(m16, x16, device)})

    # 4. Triton fused RMSNorm (fp16)
    mf = make_model(hidden, layers, use_rmsnorm=True).to(device).half()
    results.append({"stage": "triton_rmsnorm_fp16", **measure(mf, x16, device)})

    # 5. CUDA Graph (fp16 + triton)
    g = build_cuda_graph(mf, x16)
    with torch.no_grad():
        for _ in range(20):
            g.replay()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        t = cuda_event_latency(lambda: g.replay(), device=device, warmup=0,
                               iterations=100)
    results.append({
        "stage": "cuda_graph_fp16",
        "latency_us_p50": t.p50,
        "latency_us_p95": t.p95,
        "latency_us_p99": t.p99,
        "gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6,
        "kernel_count": 1,  # graph replay = single launch
    })

    return results


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("requires CUDA")

    results = run_staircase(device)
    from common.report import write_report
    write_report(args.output, {"kind": "inference_optimization",
                               "environment": collect_environment(device),
                               "stages": results})

    print(f"{'stage':22s} {'p50_us':>9s} {'p95_us':>9s} {'p99_us':>9s} "
          f"{'mem_mb':>8s} {'kernels':>8s}")
    for r in results:
        print(f"{r['stage']:22s} {r['latency_us_p50']:9.1f} {r['latency_us_p95']:9.1f} "
              f"{r['latency_us_p99']:9.1f} {r['gpu_memory_mb']:8.1f} "
              f"{r['kernel_count']:8d}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
