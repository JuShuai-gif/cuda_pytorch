"""Export a residual-MLP model to ONNX and save reference I/O.

This is the only Python step in the TensorRT lab: PyTorch lives in the flashrt
env, so we use it to (1) export the model to ONNX and (2) capture a reference
input and the corresponding PyTorch output for correctness checking.  The
TensorRT build and inference are done in C++ (see build_engine.cpp /
run_engine.cpp).

Run from the repo root:

    PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python
    $PY Work/src/inference/tensorrt/python/export_onnx.py \
        --hidden 1024 --layers 4 --batch 1 --seq 16 --outdir /tmp/trt_model

Outputs (in --outdir):
    model.onnx       ONNX graph (dynamic batch and seq dims)
    input.bin        fp32 input tensor, shape [batch, seq, hidden]
    output_ref.bin   fp32 reference output from PyTorch eager
    torch_bench.json torch eager vs torch.compile latency (for the report)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

# torch.compile's Inductor backend uses Triton, whose bundled ptxas-blackwell
# (CUDA 12.9) rejects sm_110a on this Jetson/Thor.  Point it at CUDA 13 ptxas.
os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda-13.0/bin/ptxas")


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


def make_model(hidden: int, layers: int) -> nn.Module:
    return nn.Sequential(*[ResidualBlock(hidden) for _ in range(layers)])


def bench(fn, warmup=20, iterations=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    n = len(samples)
    return {
        "mean_ms": sum(samples) / n,
        "median_ms": samples[n // 2],
        "p95_ms": samples[int(n * 0.95)],
        "p99_ms": samples[min(n - 1, int(n * 0.99))],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--outdir", default="/tmp/trt_model")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.hidden, args.layers).cuda().eval()
    x = torch.randn(args.batch, args.seq, args.hidden, device="cuda")

    onnx_path = outdir / "model.onnx"
    torch.onnx.export(
        model,
        x,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
        opset_version=17,
    )

    with torch.no_grad():
        y = model(x)
    (outdir / "input.bin").write_bytes(x.cpu().numpy().tobytes())
    (outdir / "output_ref.bin").write_bytes(y.cpu().numpy().tobytes())

    eager = bench(lambda: model(x))
    compiled = torch.compile(model)
    with torch.no_grad():
        compiled(x)  # warm compile
    comp = bench(lambda: compiled(x))

    report = {
        "hidden": args.hidden,
        "layers": args.layers,
        "batch": args.batch,
        "seq": args.seq,
        "torch_eager_ms": eager,
        "torch_compile_ms": comp,
    }
    (outdir / "torch_bench.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"wrote {onnx_path}, input.bin, output_ref.bin, torch_bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
