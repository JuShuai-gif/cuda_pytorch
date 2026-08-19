"""Generate a Markdown Before/After optimization report from the staircase JSON.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m projects.inference_optimization.report --input /tmp/opt.json --output /tmp/report.md
"""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    data = json.load(open(args.input))
    stages = data["stages"]
    base = stages[0]

    lines = ["# GPU Inference Optimization Report", ""]
    lines.append(f"模型：残差 MLP（hidden=1024, layers=4, batch=1, seq=16）")
    lines.append(f"硬件：{data['environment'].get('device_properties', {}).get('name', 'CUDA')}")
    lines.append("")
    lines.append("## 优化阶梯（Before → After）")
    lines.append("")
    lines.append("| stage | p50(us) | p95(us) | p99(us) | 显存(MB) | kernel 数 | 加速比 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in stages:
        speedup = base["latency_us_p50"] / r["latency_us_p50"]
        lines.append(f"| {r['stage']} | {r['latency_us_p50']:.1f} | {r['latency_us_p95']:.1f} | "
                     f"{r['latency_us_p99']:.1f} | {r['gpu_memory_mb']:.1f} | "
                     f"{r['kernel_count']} | {speedup:.2f}x |")
    lines.append("")
    lines.append("## 每个优化为什么有效")
    lines.append("")
    lines.append("- eager fp32 → torch.compile：Inductor 融合 elementwise 算子，减少 kernel 数")
    lines.append("- → fp16：Tensor Core 吞吐翻倍，显存减半")
    lines.append("- → Triton RMSNorm：fused 归一化，省中间张量（Stage 6 fusion）")
    lines.append("- → CUDA Graph：把 N 次 launch 折叠成 1 次，消除 launch 开销（Stage 2）")
    lines.append("- TensorRT FP16（Stage 7）：自动 layer fusion + tactic selection，见 TensorRT 模块")
    lines.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
