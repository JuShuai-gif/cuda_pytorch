#!/usr/bin/env python3
"""run_benchmarks.py - edge_ai_compiler_pro 基准测试.

测量并输出 Markdown 报告:
  - compilation time : edge-opt --edge-lower-to-llvm 的墙钟时间
  - optimization time: edge-opt (shape-inference + fusion) 的墙钟时间
  - runtime latency  : edge-run 内置 Profiler 的纯计算延迟
  - throughput       : 1000 / latency_ms (推理/秒)
  - memory usage     : edge-memplan 规划峰值字节

用法: python3 benchmarks/run_benchmarks.py [--build-dir DIR] [--iters N]
"""

import argparse
import os
import re
import statistics
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def timeit(cmd, iters):
    """返回 (mean_ms, min_ms) 墙钟时间."""
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run(cmd)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(samples), min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.path.join(REPO, "build"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--model", default=os.path.join(REPO, "examples/end_to_end/mlp.mlir")
    )
    ap.add_argument("--out", default=os.path.join(REPO, "benchmarks/results.md"))
    args = ap.parse_args()

    bin_dir = os.path.join(args.build_dir, "bin")
    edge_opt = os.path.join(bin_dir, "edge-opt")
    edge_run = os.path.join(bin_dir, "edge-run")
    edge_memplan = os.path.join(bin_dir, "edge-memplan")
    for t in (edge_opt, edge_run, edge_memplan):
        if not os.path.exists(t):
            raise SystemExit(f"missing tool: {t} (build first)")

    model = args.model
    iters = args.iters

    # 编译时间 (Edge -> LLVM 方言)
    comp_mean, comp_min = timeit([edge_opt, model, "--edge-lower-to-llvm"], iters)
    # 优化时间 (形状推断 + 融合)
    opt_mean, opt_min = timeit(
        [edge_opt, model, "--edge-shape-inference", "--edge-fuse-conv-bn-relu"], iters
    )

    # runtime latency (内置 profiler 纯计算延迟)
    lat_samples = []
    for _ in range(iters):
        _, out, _ = run([edge_run, model, "--edge-fill=1.0"])
        m = re.search(r"Total latency:\s*([0-9.]+)\s*ms", out)
        if m:
            lat_samples.append(float(m.group(1)))
    lat_mean = statistics.mean(lat_samples) if lat_samples else 0.0
    throughput = (1000.0 / lat_mean) if lat_mean > 0 else 0.0

    # memory usage (规划峰值)
    _, mem_out, _ = run([edge_memplan, model, "--edge-align=64"])
    mm = re.search(r"Planned peak \(reuse\)\s*:\s*(\d+)", mem_out)
    naive = re.search(r"Naive peak \(no reuse\)\s*:\s*(\d+)", mem_out)
    planned_bytes = int(mm.group(1)) if mm else 0
    naive_bytes = int(naive.group(1)) if naive else 0

    # 写报告
    lines = [
        "# Benchmark Results",
        "",
        f"- Model: `{os.path.relpath(model, REPO)}`",
        f"- Iterations: {iters}",
        "",
        "| metric | value |",
        "|--------|-------|",
        f"| Compilation time (Edge→LLVM, wall) | {comp_mean:.2f} ms (min {comp_min:.2f}) |",
        f"| Optimization time (shape+fusion, wall) | {opt_mean:.2f} ms (min {opt_min:.2f}) |",
        f"| Runtime latency (pure compute) | {lat_mean:.4f} ms |",
        f"| Throughput | {throughput:.1f} inferences/s |",
        f"| Memory peak (naive→planned) | {naive_bytes} → {planned_bytes} bytes |",
        "",
        "> Wall-clock numbers include process startup; runtime latency is the "
        "in-process Profiler measurement (pure kernel time).",
        "",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[benchmark] written to {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
