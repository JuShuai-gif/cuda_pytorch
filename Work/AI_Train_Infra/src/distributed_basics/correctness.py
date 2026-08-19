"""Compare one DDP update with a single-process global-batch reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .distributed import cleanup, initialize, resolve_dtype, wrap_ddp
from .options import options_for_variant
from .workload import (
    TinyDDPModel,
    WorkloadConfig,
    build_global_reference_input,
    loss_fn,
    make_local_input,
    require_torch,
    seed_model,
)


def _step(model: Any, optimizer: Any, inputs: Any) -> Any:
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
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-s", type=int, default=120)
    args = parser.parse_args(argv)
    context = initialize(args.device, args.backend, args.timeout_s)
    torch = require_torch()
    try:
        dtype = resolve_dtype(args.dtype, context.device)
        config = WorkloadConfig(
            local_batch_size=2,
            sequence_length=8,
            hidden_size=16,
            layers=3,
            expansion=2,
            learning_rate=1.0e-2,
        )
        reference_state = None
        reference_loss = None
        if context.is_rank_zero:
            seed_model(config.model_seed, context.device)
            reference = TinyDDPModel(config).to(device=context.device, dtype=dtype)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=config.learning_rate)
            global_inputs = build_global_reference_input(
                config,
                context.world_size,
                device=context.device,
                dtype=dtype,
            )
            reference_loss = float(_step(reference, reference_optimizer, global_inputs).item())
            reference_state = {
                name: tensor.detach().clone()
                for name, tensor in reference.state_dict().items()
            }
        torch.distributed.barrier()

        seed_model(config.model_seed, context.device)
        local_model = TinyDDPModel(config).to(device=context.device, dtype=dtype)
        ddp = wrap_ddp(local_model, context, options_for_variant(args.variant))
        optimizer = torch.optim.SGD(ddp.parameters(), lr=config.learning_rate)
        local_inputs = make_local_input(
            config,
            context.rank,
            device=context.device,
            dtype=dtype,
        )
        local_loss = float(_step(ddp, optimizer, local_inputs).item())

        cross_rank_max = torch.zeros((), device=context.device, dtype=torch.float32)
        for parameter in ddp.module.parameters():
            rank_zero_value = parameter.detach().clone()
            torch.distributed.broadcast(rank_zero_value, src=0)
            cross_rank_max = torch.maximum(
                cross_rank_max,
                (parameter.detach().float() - rank_zero_value.float()).abs().max(),
            )
        torch.distributed.all_reduce(cross_rank_max, op=torch.distributed.ReduceOp.MAX)

        reference_max = torch.zeros((), device=context.device, dtype=torch.float32)
        if context.is_rank_zero:
            assert reference_state is not None
            for name, tensor in ddp.module.state_dict().items():
                reference_max = torch.maximum(
                    reference_max,
                    (tensor.detach().float() - reference_state[name].float()).abs().max(),
                )
        torch.distributed.broadcast(reference_max, src=0)
        passed = bool(reference_max.item() <= args.atol and cross_rank_max.item() <= args.atol)
        pass_tensor = torch.tensor(int(passed), device=context.device)
        torch.distributed.all_reduce(pass_tensor, op=torch.distributed.ReduceOp.MIN)
        passed = bool(pass_tensor.item())

        if context.is_rank_zero:
            report = {
                "schema_version": 1,
                "test": "ddp_one_step_matches_global_batch_reference",
                "context": context.to_dict(),
                "variant": args.variant,
                "dtype": str(dtype),
                "configuration": config.to_dict(),
                "reference_loss": reference_loss,
                "rank_zero_local_loss": local_loss,
                "reference_parameter_max_abs_error": float(reference_max.item()),
                "cross_rank_parameter_max_abs_error": float(cross_rank_max.item()),
                "atol": args.atol,
                "passed": passed,
            }
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            print(rendered, end="")
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as handle:
                    handle.write(rendered)
        return 0 if passed else 1
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
