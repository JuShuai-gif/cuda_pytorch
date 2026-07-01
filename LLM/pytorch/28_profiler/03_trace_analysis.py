"""Profiler case study 1: trace analysis for performance bottleneck.

Companion script for profiler/ directory. Covers:
  1. torch.profiler for kernel-level analysis
  2. Trace export for Chrome/systrace
  3. Memory timeline analysis

Run:
    python 03_trace_analysis.py
"""

import sys

import torch


def exp_profiler_basics():
    print("=" * 60)
    print("1. torch.profiler for performance analysis")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
    )

    x = torch.randn(32, 512)

    # Profile with trace
    activities = []
    if torch.cuda.is_available():
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("forward_pass"):
            y = model(x)

        with torch.profiler.record_function("loss"):
            loss = y.sum()

        with torch.profiler.record_function("backward"):
            loss.backward()

    # Export trace
    prof.export_chrome_trace("/tmp/trace.json")
    print(f"  Profile saved to /tmp/trace.json")
    print(f"  Open in: chrome://tracing")

    # Print summary
    print(f"\n  Profile summary:")
    key_avgs = prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=5)
    print(key_avgs)

    import os; os.remove("/tmp/trace.json")
    print()


EXPERIMENTS = {
    "basics": exp_profiler_basics,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[profiler case 1] DONE")


if __name__ == "__main__":
    main()
