"""Compute DDP backward/NCCL overlap from an exported Nsight Systems SQLite DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from .timeline import Interval, summarize_overlap


GLOBAL_ID_PROCESS_MASK = -16777216  # clear the low 24-bit thread-id field


def _ranges(connection: sqlite3.Connection, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        WHERE n.end IS NOT NULL
        """
    )
    result = []
    for start, end, global_tid, name in rows:
        if name is not None and pattern.search(str(name)):
            result.append(
                {"start": start, "end": end, "global_tid": global_tid, "name": str(name)}
            )
    return result


def _kernels_for_range(connection: sqlite3.Connection, item: dict[str, Any]) -> list[dict[str, Any]]:
    global_pid = item["global_tid"] & GLOBAL_ID_PROCESS_MASK
    rows = connection.execute(
        """
        SELECT k.start, k.end, names.value, k.deviceId, k.streamId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
          ON r.correlationId = k.correlationId
         AND (r.globalTid & ?) = k.globalPid
        JOIN StringIds AS names ON k.demangledName = names.id
        WHERE (r.globalTid & ?) = ? AND r.start >= ? AND r.start <= ?
        """,
        (
            GLOBAL_ID_PROCESS_MASK,
            GLOBAL_ID_PROCESS_MASK,
            global_pid,
            item["start"],
            item["end"],
        ),
    )
    return [
        {"start": start, "end": end, "name": name, "device": device, "stream": stream}
        for start, end, name, device, stream in rows
    ]


def analyze(database: Path, backward_regex: str, communication_regex: str) -> dict[str, Any]:
    if not database.is_file():
        raise FileNotFoundError(database)
    backward_pattern = re.compile(backward_regex)
    communication_pattern = re.compile(communication_regex, re.IGNORECASE)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        ranges = _ranges(connection, backward_pattern)
        analyses = []
        for item in ranges:
            kernels = _kernels_for_range(connection, item)
            communication = [
                Interval(kernel["start"], kernel["end"])
                for kernel in kernels
                if communication_pattern.search(kernel["name"])
            ]
            compute = [
                Interval(kernel["start"], kernel["end"])
                for kernel in kernels
                if not communication_pattern.search(kernel["name"])
            ]
            summary = summarize_overlap(compute, communication)
            analyses.append(
                {
                    "range": item["name"],
                    "kernel_count": len(kernels),
                    "compute_kernel_count": len(compute),
                    "communication_kernel_count": len(communication),
                    "timeline": summary.to_dict(),
                    "communication_kernel_names": sorted(
                        {kernel["name"] for kernel in kernels if communication_pattern.search(kernel["name"])}
                    ),
                }
            )
        return {
            "schema_version": 1,
            "database": str(database.resolve()),
            "backward_regex": backward_regex,
            "communication_regex": communication_regex,
            "ranges": analyses,
            "warnings": [
                "exposed_communication is the union of communication intervals not overlapped by backward kernels; it is not automatically the causal step-time penalty",
                "kernel attribution follows all CUDA launches from the same process during each CPU NVTX backward range, including autograd worker threads; inspect the raw timeline before accepting the classification",
            ],
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--backward-regex", default=r"^ddp_backward_rank_[0-9]+_step_[0-9]+$")
    parser.add_argument("--communication-regex", default=r"nccl|allreduce|all_reduce|reduce_scatter|all_gather")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = analyze(args.database, args.backward_regex, args.communication_regex)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
