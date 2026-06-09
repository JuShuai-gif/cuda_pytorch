#!/usr/bin/env python3
"""
机器人感知流水线性能分析入口点。

运行一个 5 阶段感知流水线（图像采集、图像预处理、
激光雷达预处理、检测、后处理），共运行 N 帧，
使用实际的 numpy 计算，并输出 latency_report.json。
"""

import sys
import time
import numpy as np

from timer import TimerContext
from tracker import LatencyTracker
from pipeline_sim import run_pipeline_frame


def main() -> int:
    num_frames = 200
    pipeline_name = "perception_v1"
    seed = 42

    # 解析命令行参数
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--frames" and i + 1 < len(args):
            num_frames = int(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif args[i] == "--help":
            print("用法: python main.py [--frames N] [--seed N] [--help]")
            return 0
        else:
            i += 1

    rng = np.random.default_rng(seed)
    tracker = LatencyTracker()

    wall_start = time.perf_counter()

    for frame_idx in range(num_frames):
        with TimerContext("end_to_end") as t:
            run_pipeline_frame(rng, tracker)
        tracker.record("end_to_end", t.elapsed_us)

        if (frame_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - wall_start
            fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{frame_idx + 1:4d}/{num_frames}] "
                f"{fps:.1f} FPS, 最近端到端: {t.elapsed_us:.0f} us",
                flush=True,
            )

    wall_end = time.perf_counter()
    total_time = wall_end - wall_start

    report_path = tracker.write_json_report(
        "latency_report.json",
        pipeline_name=pipeline_name,
        total_frames=num_frames,
    )

    st = tracker.stats()
    e2e = st.get("end_to_end", {})
    print(f"\n流水线完成: {num_frames} 帧, {total_time:.2f}s")
    print(f"平均 FPS: {num_frames / total_time:.1f}")
    print(f"端到端均值: {e2e.get('mean_us', 0):.0f} us")
    print(f"报告已写入: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
