"""Generic profiling target for nsys / ncu.

Runs a configurable inference workload with NVTX ranges around each logical
stage, so the timeline can attribute time to named regions rather than opaque
kernel names.  NVTX names use underscores (not ``/``) because ``/`` is
range-stack syntax in NCU filters.

Stages marked: ``h2d``, ``preprocess``, ``block_N`` (one per residual block),
``postprocess``.  A single pass runs all stages once; ``--steps`` repeats the
whole pass.  This is intentionally small so nsys/ncu stay fast.
"""

from __future__ import annotations

import argparse

import torch

from common.env import resolve_device, resolve_dtype
from inference.workloads import InferenceConfig, make_input, make_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    config = InferenceConfig(args.hidden, args.layers, args.batch, args.seq_len)
    model = make_model(config, device=device, dtype=dtype)

    for step in range(args.steps):
        x_cpu = torch.randn(
            config.batch, config.seq_len, config.hidden, dtype=dtype
        )

        torch.cuda.nvtx.range_push("h2d")
        x = x_cpu.to(device, non_blocking=True)
        torch.cuda.nvtx.range_pop()

        with torch.no_grad():
            for i, block in enumerate(model):
                torch.cuda.nvtx.range_push(f"block_{i}")
                x = block(x)
                torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("postprocess")
        y = x.float().sum()
        torch.cuda.nvtx.range_pop()

        del x, y, x_cpu

    torch.cuda.synchronize(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
