"""Numerically compare the eager baseline with the optimization candidate."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .baseline import build_baseline
from .optimized import build_compiled
from .workload import (
    WorkloadConfig,
    make_input,
    require_torch,
    resolve_device,
    resolve_dtype,
    synchronize,
    train_step,
)


def _max_abs(first: Any, second: Any) -> float:
    return float((first.float() - second.float()).abs().max().item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("compiled", "eager-clone"), default="compiled")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    args = parser.parse_args(argv)

    torch = require_torch()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    config = WorkloadConfig(batch_size=2, sequence_length=8, hidden_size=32, mlp_size=64)
    baseline = build_baseline(config, device=device, dtype=dtype)
    if args.candidate == "compiled":
        candidate = build_compiled(
            config, device=device, dtype=dtype, mode=args.compile_mode
        )
    else:
        candidate = build_baseline(config, device=device, dtype=dtype)
    inputs = make_input(config, device=device, dtype=dtype)

    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        expected_output = baseline(inputs)
        candidate_output = candidate(inputs)
    synchronize(device)

    tolerance = {torch.float32: (2.0e-5, 2.0e-5), torch.bfloat16: (2.0e-2, 2.0e-2)}[dtype]
    rtol, atol = tolerance
    torch.testing.assert_close(candidate_output, expected_output, rtol=rtol, atol=atol)

    baseline.train()
    candidate.train()
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=config.learning_rate)
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=config.learning_rate)
    expected_loss = train_step(baseline, baseline_optimizer, inputs)
    candidate_loss = train_step(candidate, candidate_optimizer, inputs)
    synchronize(device)
    torch.testing.assert_close(candidate_loss, expected_loss, rtol=rtol, atol=atol)

    expected_parameters = list(baseline.parameters())
    candidate_parameters = list(candidate.parameters())
    if len(expected_parameters) != len(candidate_parameters):
        raise AssertionError("candidate parameter structure differs from baseline")
    maximum_parameter_error = 0.0
    for expected, actual in zip(expected_parameters, candidate_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        maximum_parameter_error = max(maximum_parameter_error, _max_abs(actual, expected))

    report = {
        "status": "pass",
        "candidate": args.candidate,
        "device": device,
        "dtype": str(dtype),
        "rtol": rtol,
        "atol": atol,
        "forward_max_abs_error": _max_abs(candidate_output, expected_output),
        "loss_abs_error": abs(float(candidate_loss.item()) - float(expected_loss.item())),
        "parameter_max_abs_error_after_one_step": maximum_parameter_error,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
