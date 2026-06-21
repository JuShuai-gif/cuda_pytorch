#!/usr/bin/env python3
"""edge_compile.py - edge_ai_compiler_pro 端到端编译驱动 (Module 16).

串联工具链:  edge-opt(优化) -> edge-statistics -> edge-memplan -> edge-lower-to-llvm -> edge-run
并产出四份报告: fusion / compilation / latency / memory.

用法:
  python3 scripts/edge_compile.py [model.mlir] [--build-dir DIR] [--out DIR]
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    """运行命令, 返回 (returncode, stdout, stderr)."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def write_report(out_dir, name, title, body):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(f"# {title}\n\n{body}\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="edge end-to-end compiler driver")
    ap.add_argument(
        "model", nargs="?", default=os.path.join(REPO, "examples/end_to_end/mlp.mlir")
    )
    ap.add_argument("--build-dir", default=os.path.join(REPO, "build"))
    ap.add_argument("--out", default=os.path.join(REPO, "reports"))
    args = ap.parse_args()

    bin_dir = os.path.join(args.build_dir, "bin")
    edge_opt = os.path.join(bin_dir, "edge-opt")
    edge_memplan = os.path.join(bin_dir, "edge-memplan")
    edge_run = os.path.join(bin_dir, "edge-run")
    for tool in (edge_opt, edge_memplan, edge_run):
        if not os.path.exists(tool):
            sys.exit(f"error: tool not found: {tool} (build first)")

    os.makedirs(args.out, exist_ok=True)
    model = args.model
    print(f"[edge-compile] model = {model}")

    # ---- Stage 1: 优化 (shape inference + fusion) ----
    rc, optimized_ir, err = run(
        [edge_opt, model, "--edge-shape-inference", "--edge-fuse-conv-bn-relu"]
    )
    if rc != 0:
        sys.exit(f"optimize failed:\n{err}")
    opt_path = os.path.join(args.out, "optimized.mlir")
    with open(opt_path, "w") as f:
        f.write(optimized_ir)

    # ---- Stage 2: fusion 报告 (优化前后算子统计) ----
    _, stats_before, _ = run([edge_opt, model, "--edge-statistics"])
    _, stats_after, _ = run([edge_opt, opt_path, "--edge-statistics"])
    fusion_body = (
        "## Before optimization\n\n```\n"
        + _stats_only(stats_before)
        + "```\n\n## After (shape-inference + conv-bn-relu fusion)\n\n```\n"
        + _stats_only(stats_after)
        + "```\n"
    )
    p_fusion = write_report(args.out, "fusion_report.md", "Fusion Report", fusion_body)

    # ---- Stage 3: compilation 报告 (能否一路降到 LLVM 方言) ----
    rc, llvm_ir, err = run([edge_opt, model, "--edge-lower-to-llvm"])
    llvm_funcs = llvm_ir.count("llvm.func")
    ok = rc == 0 and llvm_funcs > 0
    comp_body = (
        f"- Lower to LLVM dialect: {'OK' if ok else 'FAILED'}\n"
        f"- llvm.func count: {llvm_funcs}\n"
        f"- llvm.* lines: {sum(1 for l in llvm_ir.splitlines() if 'llvm.' in l)}\n\n"
        "Pipeline: Edge -> Linalg -> bufferize -> loops -> LLVM dialect.\n"
    )
    p_comp = write_report(
        args.out, "compilation_report.md", "Compilation Report", comp_body
    )

    # ---- Stage 4: memory 报告 ----
    _, mem_out, _ = run([edge_memplan, model, "--edge-align=64"])
    p_mem = write_report(
        args.out, "memory_report.md", "Memory Report", "```\n" + mem_out + "```\n"
    )

    # ---- Stage 5: latency 报告 (执行 + profiling) ----
    rc, run_out, err = run([edge_run, model, "--edge-fill=1.0"])
    p_lat = write_report(
        args.out,
        "latency_report.md",
        "Latency Report",
        "```\n" + (run_out if rc == 0 else err) + "```\n",
    )

    # ---- 汇总 ----
    print("[edge-compile] reports written:")
    for p in (p_fusion, p_comp, p_mem, p_lat):
        print("  -", os.path.relpath(p, REPO))
    print(f"[edge-compile] compilation to LLVM: {'OK' if ok else 'FAILED'}")


def _stats_only(text):
    """只保留 edge-statistics 报告部分 (去掉打印的 IR)."""
    lines = []
    keep = False
    for l in text.splitlines():
        if l.startswith("# Edge Graph Statistics"):
            keep = True
        if l.startswith("module"):
            break
        if keep:
            lines.append(l)
    return ("\n".join(lines).rstrip() + "\n") if lines else text


if __name__ == "__main__":
    main()
