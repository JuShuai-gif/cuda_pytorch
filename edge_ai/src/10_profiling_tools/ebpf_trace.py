#!/usr/bin/env python3
"""
ebpf_trace.py - 模拟 eBPF 风格的函数延迟追踪器。

本脚本提供两种模式：

1. 模拟模式（默认，无需 root 权限）：
   使用 Python 的 cProfile 和自定义追踪装饰器来测量函数
   执行时间。打印延迟分布直方图。

2. BPFTRACE 示例（注释示例）：
   用于生产环境的真实 bpftrace 单行命令。这些需要 bpftrace 和
   root 权限。

用法：
  python3 ebpf_trace.py --simulate                          # 运行模拟
  python3 ebpf_trace.py --simulate --target <函数名>         # 追踪特定函数
  python3 ebpf_trace.py --bpftrace-examples                 # 显示 bpftrace 示例

模拟模式前置条件：Python 3.7+
真实 eBPF 前置条件：bpftrace、bcc-tools、内核 4.15+
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import math
import os
import pstats
import random
import sys
import textwrap
import time
import io
from collections import defaultdict
from typing import Callable, Dict, List, Tuple


# ============================================================================
# 延迟直方图工具
# ============================================================================


class LatencyHistogram:
    """构建记录延迟的对数尺度直方图。"""

    def __init__(self, name: str = "latency"):
        self.name = name
        self.buckets: List[int] = []
        self.count = 0
        self.total = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def record(self, duration_us: float) -> None:
        self.buckets.append(int(duration_us))
        self.count += 1
        self.total += duration_us
        if duration_us < self.min_val:
            self.min_val = duration_us
        if duration_us > self.max_val:
            self.max_val = duration_us

    def print_histogram(self, unit: str = "us") -> None:
        if self.count == 0:
            print(f"  [{self.name}] 没有记录到样本。")
            return

        # 定义对数刻度桶边界
        edges = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
        ]
        bucket_names = [
            "0-1",
            "1-2",
            "2-4",
            "4-8",
            "8-16",
            "16-32",
            "32-64",
            "64-128",
            "128-256",
            "256-512",
            "512-1K",
            "1K-2K",
            "2K-4K",
            "4K-8K",
            "8K-16K",
            "16K-32K",
            "32K-64K",
            ">64K",
        ]

        counts = [0] * len(edges)
        for val in self.buckets:
            for i, edge in enumerate(edges):
                if val <= edge:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1

        print(f"\n{'=' * 60}")
        print(f"  延迟分布: {self.name}")
        print(f"{'=' * 60}")
        print(f"  样本数:   {self.count:>8}")
        print(f"  平均值:   {self.total / max(self.count, 1):>8.2f} {unit}")
        print(f"  P50:      {self.percentile(50):>8.0f} {unit}")
        print(f"  P95:      {self.percentile(95):>8.0f} {unit}")
        print(f"  P99:      {self.percentile(99):>8.0f} {unit}")
        print(f"  最小值:   {self.min_val:>8.0f} {unit}")
        print(f"  最大值:   {self.max_val:>8.0f} {unit}")
        print()
        print(f"  {'范围':<14} {'数量':>8}  分布")
        print(f"  {'-' * 14} {'-' * 8}  {'-' * 30}")

        max_count = max(counts) if counts else 1
        for i, (name, cnt) in enumerate(zip(bucket_names, counts)):
            if cnt == 0:
                continue
            bar_len = int(cnt / max_count * 30)
            bar = "#" * bar_len
            print(f"  {name:<14} {cnt:>8}  {bar}")

    def percentile(self, pct: float) -> float:
        if not self.buckets:
            return 0.0
        sorted_vals = sorted(self.buckets)
        idx = int(math.ceil(pct / 100.0 * len(sorted_vals))) - 1
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return float(sorted_vals[idx])


# ============================================================================
# 函数追踪器（使用 cProfile 模拟 eBPF）
# ============================================================================


class FunctionTracer:
    """使用 cProfile 和装饰器追踪函数调用延迟。"""

    def __init__(self) -> None:
        self.histograms: Dict[str, LatencyHistogram] = {}
        self._profilers: Dict[str, cProfile.Profile] = {}

    def trace(self, func: Callable) -> Callable:
        """装饰器：追踪函数执行时间。"""
        name = func.__qualname__
        hist = self.histograms.setdefault(name, LatencyHistogram(name))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ns = time.perf_counter_ns() - start
                hist.record(float(elapsed_ns) / 1000.0)  # 将纳秒转换为微秒

        return wrapper

    def print_all(self) -> None:
        for hist in sorted(
            self.histograms.values(), key=lambda h: h.total, reverse=True
        ):
            hist.print_histogram()


# ============================================================================
# 模拟工作负载：模拟机器人流水线用于追踪演示
# ============================================================================

tracer = FunctionTracer()


@tracer.trace
def sensor_read():
    """模拟传感器数据采集。"""
    time.sleep(random.uniform(0.0001, 0.002))  # 100us ~ 2ms
    return {"frame_id": random.randint(0, 100000)}


@tracer.trace
def image_preprocess(data):
    """模拟图像预处理：缩放、归一化。"""
    time.sleep(random.uniform(0.0005, 0.003))
    for _ in range(random.randint(100, 1000)):
        _ = math.sqrt(float(random.randint(1, 1000)))
    return data


@tracer.trace
def object_detection(data):
    """模拟 GPU 推理（模型前向传播）。"""
    # 模拟 GPU 内核启动开销 + 推理时间
    time.sleep(random.uniform(0.008, 0.025))  # 8ms ~ 25ms
    return {"detections": random.randint(1, 15)}


@tracer.trace
def target_tracking(detections):
    """模拟卡尔曼滤波跟踪更新。"""
    time.sleep(random.uniform(0.0002, 0.001))
    return detections


@tracer.trace
def path_planning(tracks):
    """模拟 A* / RRT 路径规划。"""
    time.sleep(random.uniform(0.002, 0.012))  # 2ms ~ 12ms
    return {"trajectory": [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]}


@tracer.trace
def control_output(trajectory):
    """模拟 MPC 控制输出计算。"""
    time.sleep(random.uniform(0.0001, 0.0005))  # 100us ~ 500us
    return {"throttle": 0.3, "steering": 0.01}


@tracer.trace
def process_frame():
    """端到端帧处理流水线。"""
    data = sensor_read()
    data = image_preprocess(data)
    detections = object_detection(data)
    tracks = target_tracking(detections)
    traj = path_planning(tracks)
    cmd = control_output(traj)
    return cmd


# ============================================================================
# 纯 Python 直方图生成（无 cProfile），用于轻量级追踪
# ============================================================================


def _raw_sensor_read() -> dict:
    time.sleep(random.uniform(0.0001, 0.002))
    return {"frame_id": random.randint(0, 100000)}


def _raw_image_preprocess(data: dict) -> dict:
    time.sleep(random.uniform(0.0005, 0.003))
    for _ in range(random.randint(100, 1000)):
        _ = math.sqrt(float(random.randint(1, 1000)))
    return data


def _raw_object_detection(data: dict) -> dict:
    time.sleep(random.uniform(0.008, 0.025))
    return {"detections": random.randint(1, 15)}


def _raw_target_tracking(detections: dict) -> dict:
    time.sleep(random.uniform(0.0002, 0.001))
    return detections


def _raw_path_planning(tracks: dict) -> dict:
    time.sleep(random.uniform(0.002, 0.012))
    return {"trajectory": [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]}


def _raw_control_output(trajectory: dict) -> dict:
    time.sleep(random.uniform(0.0001, 0.0005))
    return {"throttle": 0.3, "steering": 0.01}


def run_lightweight_trace(num_frames: int = 100) -> Dict[str, List[float]]:
    """运行流水线并收集每个函数的时间（不使用 cProfile）。"""
    timings: Dict[str, List[float]] = defaultdict(list)

    for frame_idx in range(num_frames):
        t_start = time.perf_counter_ns()

        t0 = time.perf_counter_ns()
        data = _raw_sensor_read()
        timings["sensor_read"].append((time.perf_counter_ns() - t0) / 1e3)

        t0 = time.perf_counter_ns()
        data = _raw_image_preprocess(data)
        timings["image_preprocess"].append((time.perf_counter_ns() - t0) / 1e3)

        t0 = time.perf_counter_ns()
        detections = _raw_object_detection(data)
        timings["object_detection"].append((time.perf_counter_ns() - t0) / 1e3)

        t0 = time.perf_counter_ns()
        tracks = _raw_target_tracking(detections)
        timings["target_tracking"].append((time.perf_counter_ns() - t0) / 1e3)

        t0 = time.perf_counter_ns()
        traj = _raw_path_planning(tracks)
        timings["path_planning"].append((time.perf_counter_ns() - t0) / 1e3)

        t0 = time.perf_counter_ns()
        _ = _raw_control_output(traj)
        timings["control_output"].append((time.perf_counter_ns() - t0) / 1e3)

        e2e_us = (time.perf_counter_ns() - t_start) / 1e3
        timings["end_to_end"].append(e2e_us)

        if (frame_idx + 1) % 20 == 0:
            print(
                f"  已处理 {frame_idx + 1}/{num_frames} 帧... 端到端: {e2e_us:.0f} us"
            )

    return dict(timings)


def print_lightweight_summary(timings: Dict[str, List[float]]) -> None:
    """打印每个被追踪函数的直方图摘要。"""
    print(f"\n{'=' * 70}")
    print(f"  轻量级追踪摘要（模拟 eBPF 风格）")
    print(f"{'=' * 70}")

    for name, vals in sorted(timings.items(), key=lambda kv: sum(kv[1]), reverse=True):
        hist = LatencyHistogram(name)
        for v in vals:
            hist.record(v)
        hist.print_histogram()


# ============================================================================
# bpftrace 示例（供生产环境使用的注释示例）
# ============================================================================

BPFTRACE_EXAMPLES = """
======================================================================
  真实 BPFTRACE 示例（需要 bpftrace + root 权限）
======================================================================

1. 函数入口/出口延迟直方图（内核函数）：
   $ sudo bpftrace -e 'kprobe:vfs_read { @start[tid]=nsecs; }
                        kretprobe:vfs_read /@start[tid]/ {
                            @lat_us = hist((nsecs - @start[tid]) / 1000);
                            delete(@start[tid]); }'

2. 通过 uprobe 追踪用户空间函数延迟（需要调试符号）：
   $ sudo bpftrace -e 'uprobe:/path/to/binary:func_name { @start[tid]=nsecs; }
                        uretprobe:/path/to/binary:func_name /@start[tid]/ {
                            @lat_us = hist((nsecs - @start[tid]) / 1000);
                            delete(@start[tid]); }'

3. 追踪每个任务的调度延迟：
   $ sudo bpftrace -e 'tracepoint:sched:sched_switch {
                            @sched_lat_ns = hist(nsecs -
                                args->prev_state ? 0 : args->prev->sched_info.last_queued);
                         }'

4. 按进程统计系统调用数量：
   $ sudo bpftrace -e 'tracepoint:syscalls:sys_enter_* {
                            @syscalls[comm] = count(); }'

5. 块 I/O 延迟分布：
   $ sudo bpftrace -e 'kprobe:blk_mq_start_request { @start[arg0]=nsecs; }
                        kprobe:blk_mq_complete_request /@start[arg0]/ {
                            @bio_lat_us = hist((nsecs - @start[arg0]) / 1000);
                            delete(@start[arg0]); }'

6. 内存分配追踪（分配大小直方图）：
   $ sudo bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc {
                            @alloc_bytes = hist(arg0); }'

7. 监控特定 PID 的某个函数：
   $ sudo bpftrace -e 'uretprobe:/proc/PID/exe:process_frame {
                            printf("process_frame 已返回\\n"); }'

======================================================================
  BCC 工具集示例
======================================================================

  $ sudo /usr/share/bcc/tools/funclatency -u ./my_app:process_frame    # 用户函数延迟
  $ sudo /usr/share/bcc/tools/biolatency                              # 块 I/O 延迟
  $ sudo /usr/share/bcc/tools/cachestat                               # 页面缓存命中率
  $ sudo /usr/share/bcc/tools/runqlat                                 # 调度器运行队列延迟
  $ sudo /usr/share/bcc/tools/tcptop                                   # TCP 流量排行

======================================================================
"""


# ============================================================================
# 命令行界面
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟 eBPF 风格的函数延迟追踪器")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="运行模拟追踪（默认）",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="要模拟的帧数（默认：100）",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="使用 cProfile 进行追踪（替代轻量级计时器）",
    )
    parser.add_argument(
        "--bpftrace-examples",
        action="store_true",
        help="显示真实 bpftrace 和 BCC 示例",
    )
    args = parser.parse_args()

    if args.bpftrace_examples:
        print(BPFTRACE_EXAMPLES)
        return

    print("=" * 60)
    print("  eBPF 风格函数延迟追踪器（模拟）")
    print("=" * 60)
    print(f"  模式:       {'cProfile' if args.cprofile else '轻量级计时器'}")
    print(f"  帧数:       {args.frames}")
    print(f"  PID:        {os.getpid()}")
    print()

    if args.cprofile:
        print("[信息] 正在使用 cProfile 运行...")
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.frames):
            process_frame()
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        print("\n--- cProfile 累计时间前 30 函数 ---")
        print(s.getvalue())

        print("\n--- 每个函数的延迟直方图 ---")
        tracer.print_all()
    else:
        print("[信息] 正在运行轻量级追踪...")
        timings = run_lightweight_trace(args.frames)
        print_lightweight_summary(timings)

    print("\n[完成] 查看真实 bpftrace 示例：python3 ebpf_trace.py --bpftrace-examples")


if __name__ == "__main__":
    main()
