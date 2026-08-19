"""DDP workload with rank/step NVTX ranges for PyTorch Profiler or nsys."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from .distributed import cleanup, initialize, resolve_dtype, synchronize_device, wrap_ddp
from .options import options_for_variant
from .workload import TinyDDPModel, WorkloadConfig, loss_fn, make_local_input, require_torch, seed_model


def _range(torch: Any, device: str, name: str) -> Any:
    return torch.cuda.nvtx.range(name) if device == "cuda" else nullcontext()


def _step(torch: Any, model: Any, optimizer: Any, inputs: Any, context: Any, index: int) -> Any:
    with ExitStack() as stack:
        stack.enter_context(torch.profiler.record_function(f"ddp_step_{index}"))
        stack.enter_context(_range(torch, context.device, f"ddp_step_rank_{context.rank}_{index}"))
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("ddp_forward"):
            with _range(torch, context.device, f"ddp_forward_rank_{context.rank}_step_{index}"):
                loss = loss_fn(model(inputs))
        with torch.profiler.record_function("ddp_backward"):
            with _range(torch, context.device, f"ddp_backward_rank_{context.rank}_step_{index}"):
                loss.backward()
        with torch.profiler.record_function("ddp_optimizer"):
            with _range(torch, context.device, f"ddp_optimizer_rank_{context.rank}_step_{index}"):
                optimizer.step()
        return loss.detach()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler", choices=("nvtx", "torch"), default="nvtx")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--bucket-cap-mb", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.steps <= 0:
        parser.error("--warmup must be >= 0 and --steps must be > 0")
    context = initialize(args.device, args.backend, args.timeout_s)
    torch = require_torch()
    try:
        dtype = resolve_dtype(args.dtype, context.device)
        config = WorkloadConfig(
            local_batch_size=args.local_batch_size,
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            layers=args.layers,
            expansion=args.expansion,
        )
        seed_model(config.model_seed, context.device)
        model = TinyDDPModel(config).to(device=context.device, dtype=dtype)
        ddp = wrap_ddp(model, context, options_for_variant(args.variant, bucket_cap_mb=args.bucket_cap_mb))
        optimizer = torch.optim.SGD(ddp.parameters(), lr=config.learning_rate)
        inputs = make_local_input(config, context.rank, device=context.device, dtype=dtype)
        for index in range(args.warmup):
            _step(torch, ddp, optimizer, inputs, context, -index - 1)
        synchronize_device(context)
        torch.distributed.barrier()

        if args.profiler == "nvtx":
            with _range(torch, context.device, f"ddp_profile_steady_state_rank_{context.rank}"):
                for index in range(args.steps):
                    _step(torch, ddp, optimizer, inputs, context, index)
            synchronize_device(context)
        else:
            if args.output_dir is None:
                raise ValueError("--output-dir is required for --profiler torch")
            if context.is_rank_zero:
                args.output_dir.mkdir(parents=True, exist_ok=False)
            torch.distributed.barrier()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if context.device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                acc_events=True,
            ) as profiler:
                for index in range(args.steps):
                    _step(torch, ddp, optimizer, inputs, context, index)
                    profiler.step()
            synchronize_device(context)
            trace = args.output_dir / f"rank_{context.rank}.json"
            profiler.export_chrome_trace(str(trace))
            sort_key = "self_cuda_time_total" if context.device == "cuda" else "self_cpu_time_total"
            (args.output_dir / f"rank_{context.rank}_key_averages.txt").write_text(
                profiler.key_averages().table(sort_by=sort_key, row_limit=40) + "\n",
                encoding="utf-8",
            )
        torch.distributed.barrier()
        if context.is_rank_zero:
            print(
                f"profile complete: world_size={context.world_size} backend={context.backend} "
                f"variant={args.variant}; inspect per-rank backward and NCCL kernels"
            )
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
