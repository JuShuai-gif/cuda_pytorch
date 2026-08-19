"""Synchronized DDP training-step benchmark with per-rank raw samples."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
from typing import Any

from metrics.metrics import summarize_latencies

from .distributed import cleanup, initialize, resolve_dtype, synchronize_device, wrap_ddp
from .options import options_for_variant
from .workload import (
    TinyDDPModel,
    WorkloadConfig,
    linear_training_flops,
    loss_fn,
    make_local_input,
    parameter_count,
    require_torch,
    seed_model,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return result


def _train_step(model: Any, optimizer: Any, inputs: Any) -> Any:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs))
    loss.backward()
    optimizer.step()
    return loss.detach()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--local-batch-size", type=_positive_int, default=4)
    parser.add_argument("--sequence-length", type=_positive_int, default=64)
    parser.add_argument("--hidden-size", type=_positive_int, default=256)
    parser.add_argument("--layers", type=_positive_int, default=8)
    parser.add_argument("--expansion", type=_positive_int, default=4)
    parser.add_argument("--bucket-cap-mb", type=float)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--iterations", type=_positive_int, default=30)
    parser.add_argument("--peak-tflops-per-device", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-s", type=_positive_int, default=300)
    args = parser.parse_args(argv)
    if args.bucket_cap_mb is not None and args.bucket_cap_mb <= 0:
        parser.error("--bucket-cap-mb must be > 0")
    if args.peak_tflops_per_device is not None and args.peak_tflops_per_device <= 0:
        parser.error("--peak-tflops-per-device must be > 0")

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
        options = options_for_variant(args.variant, bucket_cap_mb=args.bucket_cap_mb)
        seed_model(config.model_seed, context.device)
        model = TinyDDPModel(config).to(device=context.device, dtype=dtype)
        ddp = wrap_ddp(model, context, options)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=config.learning_rate)
        inputs = make_local_input(
            config, context.rank, device=context.device, dtype=dtype
        )
        for _ in range(args.warmup):
            _train_step(ddp, optimizer, inputs)
        synchronize_device(context)
        torch.distributed.barrier()
        if context.device == "cuda":
            torch.cuda.reset_peak_memory_stats(context.local_rank)

        samples_s: list[float] = []
        last_loss = None
        for _ in range(args.iterations):
            synchronize_device(context)
            started = time.perf_counter()
            last_loss = _train_step(ddp, optimizer, inputs)
            synchronize_device(context)
            samples_s.append(time.perf_counter() - started)

        local_payload = {
            "rank": context.rank,
            "step_times_s": samples_s,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(context.local_rank)
                if context.device == "cuda"
                else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(context.local_rank)
                if context.device == "cuda"
                else None
            ),
            "last_loss": float(last_loss.item()) if last_loss is not None else None,
        }
        gathered: list[Any] = [None for _ in range(context.world_size)]
        torch.distributed.all_gather_object(gathered, local_payload)
        if context.is_rank_zero:
            critical_samples_s = [
                max(rank_payload["step_times_s"][index] for rank_payload in gathered)
                for index in range(args.iterations)
            ]
            latency = summarize_latencies(critical_samples_s)
            mean_step_s = latency.mean_ms / 1000.0
            global_batch = config.local_batch_size * context.world_size
            total_flops = linear_training_flops(config, context.world_size)
            peak = (
                args.peak_tflops_per_device * 1.0e12 * context.world_size
                if args.peak_tflops_per_device is not None
                else None
            )
            mfu = total_flops / (mean_step_s * peak) if peak else None
            params = parameter_count(ddp.module)
            gradient_bytes = params * torch.tensor([], dtype=dtype).element_size()
            ring_bytes_one_direction = (
                2.0 * (context.world_size - 1) / context.world_size * gradient_bytes
                if context.world_size > 1
                else 0.0
            )
            report = {
                "schema_version": 1,
                "benchmark": "tiny_residual_mlp_ddp_training_step",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "torch_version": torch.__version__,
                "context": context.to_dict(),
                "variant": args.variant,
                "ddp_options": options.to_dict(),
                "configuration": config.to_dict(),
                "dtype": str(dtype),
                "measurement_boundary": "zero_grad through optimizer.step; synchronized device wall time; per-step critical sample is max rank latency",
                "latency": latency.to_dict(),
                "critical_raw_step_times_ms": [value * 1000.0 for value in critical_samples_s],
                "per_rank": gathered,
                "throughput": {
                    "global_samples_per_second": global_batch / mean_step_s,
                    "global_tokens_per_second": global_batch * config.sequence_length / mean_step_s,
                },
                "flops": {
                    "global_linear_training_flops_per_step": total_flops,
                    "convention": "FMA=2; residual-block Linear ops only; backward=2*forward; norm/GELU/loss/SGD excluded",
                },
                "mfu": mfu,
                "parameter_count": params,
                "communication_model": {
                    "gradient_payload_bytes_per_rank": gradient_bytes,
                    "ring_allreduce_bytes_sent_per_rank_estimate": ring_bytes_one_direction,
                    "ring_allreduce_bytes_received_per_rank_estimate": ring_bytes_one_direction,
                    "ring_allreduce_bytes_sent_plus_received_per_rank_estimate": 2.0 * ring_bytes_one_direction,
                    "warning": "analytical ring payload, not a measured NCCL algorithm or exposed time",
                },
                "warnings": [
                    "isolated synchronized latency does not reveal overlap; use the Nsight Systems timeline",
                    "small teaching workload and short runs are not production performance claims",
                ],
            }
            if peak is None:
                report["warnings"].append(
                    "MFU is null because no verified matching peak TFLOP/s was supplied"
                )
            if context.world_size == 1:
                report["warnings"].append(
                    "world_size=1 is a CUDA/NCCL control path with no inter-rank gradient communication"
                )
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            print(rendered, end="")
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as handle:
                    handle.write(rendered)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
