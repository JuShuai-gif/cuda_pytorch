"""Benchmark continuous vs static batching and paged vs contiguous KV cache.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m serving.llm_scheduler.benchmark --output /tmp/serving.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from serving.llm_scheduler.paged_kv import simulate_contiguous, simulate_paged
from serving.llm_scheduler.scheduler import (
    gen_requests,
    simulate_continuous,
    simulate_static,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    requests = gen_requests(args.n, args.rate, args.seed)

    static = simulate_static(list(requests), batch_size=8)
    continuous = simulate_continuous(list(requests), max_batch=32)

    # KV cache memory: block_size=16, budget = 4096 blocks (illustrative).
    contiguous = simulate_contiguous(requests, 16, 4096, max_len=512)
    paged = simulate_paged(requests, 16, 4096)

    report = {
        "kind": "llm_scheduler",
        "static_batching": static,
        "continuous_batching": continuous,
        "kv_contiguous": contiguous,
        "kv_paged": paged,
    }
    write_report(args.output, report)

    print("== static vs continuous batching ==")
    for name, r in [("static", static), ("continuous", continuous)]:
        print(f"  {name:12s} makespan={r['makespan_s']:.1f}s  "
              f"ttft_p50={r['ttft_p50']:.3f}s  ttft_p95={r['ttft_p95']:.3f}s  "
              f"throughput={r['throughput_tokens_per_s']:.1f} tok/s")

    print("== KV cache: contiguous vs paged ==")
    print(f"  contiguous: served={contiguous['served']}  waste_ratio={contiguous['waste_ratio']:.2f}")
    print(f"  paged     : served={paged['served']}  waste_ratio={paged['waste_ratio']:.2f}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
