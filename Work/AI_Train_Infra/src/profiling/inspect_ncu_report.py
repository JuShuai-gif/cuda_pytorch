#!/usr/bin/env python3
"""Extract auditable NCU metrics; unavailable counters remain null, never zero."""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CANDIDATE_METRICS = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "device__attribute_multiprocessor_count",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
)


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def load_action(path: Path, range_index: int, action_index: int) -> Any:
    try:
        import ncu_report  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "cannot import ncu_report. Locate the Nsight Compute extras/python "
            "directory for this installed version and add it to PYTHONPATH; do "
            "not install an unrelated PyPI package."
        ) from exc
    report = ncu_report.load_report(str(path))
    return report.range_by_idx(range_index).action_by_idx(action_index)


def metric_record(action: Any, name: str) -> Dict[str, Any]:
    names = set(action.metric_names())
    if name not in names:
        return {"status": "unavailable", "value": None, "unit": None}
    try:
        metric = action[name]
        return {
            "status": "available",
            "value": json_value(metric.value()),
            "unit": metric.unit() if hasattr(metric, "unit") else None,
        }
    except Exception as exc:  # The report can expose a name without a scalar rollup.
        return {"status": "unavailable", "value": None, "unit": None, "reason": str(exc)}


def inspect(path: Path, range_index: int, action_index: int, dump_all: bool) -> Dict[str, Any]:
    action = load_action(path, range_index, action_index)
    available_names = sorted(action.metric_names())
    payload: Dict[str, Any] = {
        "report": str(path.resolve()),
        "kernel": action.name(),
        "range_index": range_index,
        "action_index": action_index,
        "metric_count": len(available_names),
        "selected_metrics": {name: metric_record(action, name) for name in CANDIDATE_METRICS},
        "availability_policy": "missing metric means unavailable/null, never numeric zero",
    }
    if dump_all:
        payload["metric_names"] = available_names
        payload["all_metrics"] = {name: metric_record(action, name) for name in available_names}
    return payload


def comparison(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for name in CANDIDATE_METRICS:
        before = baseline["selected_metrics"][name]
        after = optimized["selected_metrics"][name]
        row: Dict[str, Any] = {"baseline": before, "optimized": after, "optimized_over_baseline": None}
        bval, aval = before.get("value"), after.get("value")
        if (
            before.get("status") == "available"
            and after.get("status") == "available"
            and isinstance(bval, numbers.Number)
            and isinstance(aval, numbers.Number)
            and bval != 0
        ):
            row["optimized_over_baseline"] = aval / bval
        rows[name] = row
    return {
        "baseline_kernel": baseline["kernel"],
        "optimized_kernel": optimized["kernel"],
        "metrics": rows,
        "warning": "A counter ratio is evidence, not an end-to-end speedup claim.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="baseline .ncu-rep")
    parser.add_argument("--compare", type=Path, help="optional optimized .ncu-rep")
    parser.add_argument("--range-index", type=int, default=0)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--dump-all", action="store_true", help="include every available metric and its value")
    parser.add_argument("--output", type=Path, help="created exclusively; existing files are rejected")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.range_index < 0 or args.action_index < 0:
        raise SystemExit("indices must be non-negative")
    baseline = inspect(args.report, args.range_index, args.action_index, args.dump_all)
    payload: Dict[str, Any] = {"baseline": baseline}
    if args.compare:
        optimized = inspect(args.compare, args.range_index, args.action_index, args.dump_all)
        payload["optimized"] = optimized
        payload["comparison"] = comparison(baseline, optimized)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
