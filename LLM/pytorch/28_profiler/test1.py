"""Profiler demo: basic profiling, chrome trace, memory analysis.

Companion script for profiler/profiler.md. Covers:
  1. basic profiler:       torch.autograd.profiler (legacy)
  2. torch.profiler:       profile + schedule + tensorboard trace
  3. operator table:       top operators by CPU/GPU time
  4. memory profiling:     track alloc/dealloc events
  5. FLOPs estimation:     using record_shapes

Run:
    python test1.py                  # full demo
    python test1.py basic            # legacy profiler
    python test1.py trace            # torch.profiler with chrome trace
    python test1.py operators        # operator timing table
    python test1.py memory           # memory profiling
"""

import sys
import os

import torch
import torch.nn as nn


# ============ 1. Legacy profiler ============
def exp_basic():
    print("=" * 60)
    print("1. Legacy profiler: torch.autograd.profiler")
    print("=" * 60)

    def compute(x):
        for _ in range(3):
            x = x * 2 + 1
            x = x.relu()
            x = x.sum(dim=-1)
        return x

    x = torch.randn(128, 128)

    with torch.autograd.profiler.profile() as prof:
        y = compute(x)

    # Print top 5 operators by CPU time
    print(f"  {'Key':<40s} {'CPU time (us)':>14s} {'Calls':>6s}")
    print(f"  {'-' * 40} {'-' * 14} {'-' * 6}")

    events = sorted(prof.key_averages(), key=lambda e: e.cpu_time_total, reverse=True)[
        :5
    ]
    for e in events:
        print(f"  {e.key:<40s} {e.cpu_time_total:>14.1f} {e.count:>6d}")
    print()

    if torch.cuda.is_available():
        x_cuda = torch.randn(128, 128, device="cuda")
        with torch.autograd.profiler.profile(use_cuda=True) as prof_cuda:
            y = compute(x_cuda)

        events_cuda = sorted(
            prof_cuda.key_averages(), key=lambda e: e.cuda_time_total, reverse=True
        )[:5]
        print(f"  GPU kernel timing:")
        print(f"  {'Key':<40s} {'CUDA time (us)':>14s} {'Calls':>6s}")
        print(f"  {'-' * 40} {'-' * 14} {'-' * 6}")
        for e in events_cuda:
            if e.cuda_time_total > 0:
                print(f"  {e.key:<40s} {e.cuda_time_total:>14.1f} {e.count:>6d}")
    print()


# ============ 2. torch.profiler with chrome trace ============
def exp_trace():
    print("=" * 60)
    print("2. torch.profiler: chrome trace & schedule")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    x = torch.randn(64, 128)

    out_dir = "/tmp/pytorch_profiler_demo"
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if torch.cuda.is_available()
        else [
            torch.profiler.ProfilerActivity.CPU,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for step in range(5):  # total: wait(1) + warmup(1) + active(2) + 1 extra
            y = model(x).sum()
            prof.step()

    # Find generated trace files
    trace_files = sorted(
        [
            f
            for f in os.listdir(out_dir)
            if f.endswith(".json") or f.endswith(".pt.trace.json")
        ]
    )
    print(f"  Trace files generated in {out_dir}:")
    for f in trace_files:
        fsize = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"    {f} ({fsize:.1f} KB)")

    print(f"\n  To view: open chrome://tracing and load the JSON file")
    print(f"    or: tensorboard --logdir={out_dir}")
    print()


# ============ 3. Operator timing table ============
def exp_operators():
    print("=" * 60)
    print("3. Operator timing analysis")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    def run_model(x):
        for m in model:
            x = m(x)
            x = x.relu() if not isinstance(m, nn.ReLU) else x
        return x

    x = torch.randn(256, 512)

    with torch.autograd.profiler.profile(
        record_shapes=True, use_cuda=torch.cuda.is_available()
    ) as prof:
        y = run_model(x)
        y.sum().backward()

    # Group by operator type
    from collections import defaultdict

    op_stats = defaultdict(lambda: {"cpu": 0, "cuda": 0, "count": 0})
    for e in prof.key_averages():
        name = e.key
        op_stats[name]["cpu"] += e.cpu_time_total
        op_stats[name]["cuda"] += e.cuda_time_total
        op_stats[name]["count"] += e.count

    cpu_sorted = sorted(op_stats.items(), key=lambda kv: kv[1]["cpu"], reverse=True)[:6]

    print(f"  {'Operator':<40s} {'CPU (us)':>12s} {'Calls':>6s}")
    print(f"  {'-' * 40} {'-' * 12} {'-' * 6}")
    for name, stats in cpu_sorted:
        print(f"  {name:<40s} {stats['cpu']:>12.0f} {stats['count']:>6d}")

    if torch.cuda.is_available():
        cuda_sorted = sorted(
            [(n, s) for n, s in op_stats.items() if s["cuda"] > 0],
            key=lambda kv: kv[1]["cuda"],
            reverse=True,
        )[:6]
        if cuda_sorted:
            print(f"\n  {'Operator':<40s} {'CUDA (us)':>12s} {'Calls':>6s}")
            print(f"  {'-' * 40} {'-' * 12} {'-' * 6}")
            for name, stats in cuda_sorted:
                print(f"  {name:<40s} {stats['cuda']:>12.0f} {stats['count']:>6d}")

    # Self vs total time
    print(f"\n  Self CPU time (excludes children):")
    print(f"  {'Key':<40s} {'Self CPU (us)':>14s}")
    print(f"  {'-' * 40} {'-' * 14}")
    for e in sorted(
        prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True
    )[:5]:
        print(f"  {e.key:<40s} {e.self_cpu_time_total:>14.0f}")
    print()


# ============ 4. Memory profiling ============
def exp_memory():
    print("=" * 60)
    print("4. Memory profiling")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    out_dir = "/tmp/pytorch_profiler_memory"
    os.makedirs(out_dir, exist_ok=True)

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    ).cuda()

    x = torch.randn(128, 512, device="cuda")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            y = model(x).sum()
            y.backward()
            prof.step()

    print(f"  Memory profiling events collected")
    print(f"  Key memory info available in chrome trace:")
    print(f"    - Alloc/Dealloc events with timestamps")
    print(f"    - Peak memory usage timeline")
    print(f"    - Memory allocated per operator")
    print()

    # Quick memory snapshot
    print(f"  Current GPU memory:")
    print(f"    allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"    reserved:  {torch.cuda.memory_reserved() / 1e6:.1f} MB")

    torch.cuda.empty_cache()
    print(
        f"    after empty_cache: {torch.cuda.memory_allocated() / 1e6:.1f} MB allocated"
    )
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "trace": exp_trace,
    "operators": exp_operators,
    "memory": exp_memory,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[profiler demo] DONE")


if __name__ == "__main__":
    main()
