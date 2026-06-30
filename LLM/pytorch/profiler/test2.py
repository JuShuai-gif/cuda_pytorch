"""Profiler advanced: custom events, memory timeline, distributed profiling.

Companion script for profiler/profiler.md.
  1. custom events:            mark user-defined events
  2. memory timeline:          track alloc/free over time
  3. distributed profiling:    multi-rank profiler
  4. FLOPs estimation:         manual FLOPs count per op

Run:
    python test2.py                  # full demo
    python test2.py custom_events    # custom profiling events
    python test2.py memory_timeline  # memory timeline analysis
    python test2.py flops            # FLOPs estimation
"""

import sys
import os
import torch
import torch.nn as nn


# ============ 1. Custom events ============
def exp_custom_events():
    print("=" * 60)
    print("1. Custom profiling events")
    print("=" * 60)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        # Mark a custom event
        torch.profiler.record_function("data_prep").__enter__()
        x = torch.randn(4096, 4096)
        torch.profiler.record_function("data_prep").__exit__(None, None, None)

        # Another custom span
        with torch.profiler.record_function("matmul_compute"):
            y = x @ x.T

        prof.step()

    # Find our custom events
    events = list(prof.key_averages())
    custom_events = [e for e in events if e.key in ("data_prep", "matmul_compute")]
    print(f"  Custom events found:")
    for e in custom_events:
        print(f"    {e.key:20s} cpu_time={e.cpu_time_total:.0f}us  count={e.count}")
    print("  -> record_function marks custom regions in the trace")
    print()


# ============ 2. Memory timeline ============
def exp_memory_timeline():
    print("=" * 60)
    print("2. Memory timeline analysis")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    out = "/tmp/profiler_mem_demo"
    os.makedirs(out, exist_ok=True)

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).cuda()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        for step in range(3):
            x = torch.randn(256, 1024, device="cuda")
            y = model(x).sum()
            y.backward()
            prof.step()

    # Summarize memory usage per operator
    print(f"  Memory events captured")
    mem_stats = {}
    for evt in prof.key_averages():
        if evt.cuda_memory_usage > 0:
            mem_stats[evt.key] = evt.cuda_memory_usage

    for op, mem in sorted(mem_stats.items(), key=lambda x: -x[1])[:5]:
        print(f"    {op:35s} cuda_mem={mem / 1e6:.1f} MB")

    print("  -> profile_memory=True tracks per-operator memory allocation")
    print()


# ============ 3. Manual FLOPs estimation ============
def exp_flops():
    print("=" * 60)
    print("3. Manual FLOPs estimation")
    print("=" * 60)

    # FLOPs formulas for common ops
    def estimate_flops(node, shapes):
        """Rough FLOP estimation from op type + shapes."""
        target = str(getattr(node, "target", ""))
        if not shapes:
            return 0

        # MatMul: 2 * M * N * K
        if "linear" in target:
            in_shape, w_shape = shapes[0], shapes[1]
            if len(in_shape) >= 2 and len(w_shape) >= 2:
                # [B, ..., D] @ [D, F] -> 2 * B * ... * D * F
                return (
                    2
                    * in_shape[-1]
                    * w_shape[-1]
                    * torch.tensor(in_shape[:-1]).prod().item()
                )

        # Conv2d: 2 * out_h * out_w * in_c * out_c * k_h * k_w
        if "conv" in target:
            inp, weight = shapes[0], shapes[1]
            if len(inp) == 4 and len(weight) == 4:
                B, C, H, W = inp
                OC, IC, KH, KW = weight
                return 2 * H * W * IC * OC * KH * KW * B

        # Element-wise: input_size
        if target in ("add", "mul", "relu"):
            return torch.tensor(shapes[0]).prod().item()

        return 0

    # Test: trace a model and estimate FLOPs
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Linear(8 * 8 * 8, 64),
    )

    gm = torch.fx.symbolic_trace(model)

    # Propagate shapes
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(torch.randn(1, 3, 8, 8))

    total_flops = 0
    for node in gm.graph.nodes:
        if "tensor_meta" in node.meta:
            # Get shapes of all args
            shapes = []
            for arg in node.all_input_nodes:
                if "tensor_meta" in arg.meta:
                    shapes.append(arg.meta["tensor_meta"].shape)
            if shapes:
                flops = estimate_flops(node, shapes)
                if flops > 0:
                    total_flops += flops
                    print(f"    {node.name:10s}: ~{flops / 1e6:.1f} MFLOPs")

    print(f"\n  Total estimated: {total_flops / 1e6:.1f} MFLOPs")
    print("  -> FLOPs estimated from op type + input shapes (approximate)")
    print()


EXPERIMENTS = {
    "custom_events": exp_custom_events,
    "memory_timeline": exp_memory_timeline,
    "flops": exp_flops,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[profiler test2] DONE")


if __name__ == "__main__":
    main()
