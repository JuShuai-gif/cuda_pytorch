#!/usr/bin/env python3
"""
LatencyTracker: 针对机器人感知流水线的各阶段计时，
包含直方图、百分位数统计及结构化 JSON 报告输出。
"""

import json
import math
import threading
from collections import defaultdict
from typing import Dict, List


class LatencyTracker:
    """
    记录每个阶段的延迟样本（微秒），并计算统计信息。
    线程安全，支持并发使用。输出结构化的
    latency_report.json 而非打印到 stdout。
    """

    def __init__(self):
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, stage_name: str, elapsed_us: float) -> None:
        """记录某个阶段的单次延迟样本（微秒）。"""
        with self._lock:
            self._samples[stage_name].append(elapsed_us)

    def stats(self) -> Dict[str, Dict[str, float]]:
        """
        返回各阶段统计信息: count, mean_us, p50_us, p95_us, p99_us,
        max_us, stddev_us, min_us。
        所有数值单位为微秒。
        """
        result = {}
        with self._lock:
            for name, samples in self._samples.items():
                if not samples:
                    continue
                sorted_samples = sorted(samples)
                n = len(sorted_samples)
                mean = sum(sorted_samples) / n
                p50 = sorted_samples[int(n * 0.50)]
                p95 = sorted_samples[min(int(n * 0.95), n - 1)]
                p99 = sorted_samples[min(int(n * 0.99), n - 1)]
                max_val = sorted_samples[-1]
                min_val = sorted_samples[0]
                stddev = math.sqrt(sum((x - mean) ** 2 for x in sorted_samples) / n)
                result[name] = {
                    "count": n,
                    "mean_us": mean,
                    "p50_us": p50,
                    "p95_us": p95,
                    "p99_us": p99,
                    "max_us": max_val,
                    "min_us": min_val,
                    "stddev_us": stddev,
                }
        return result

    def _histogram_bins(self, samples: List[float], num_bins: int = 20) -> List[int]:
        """计算样本列表的直方图柱计数值。"""
        if not samples or len(samples) < 2:
            return [len(samples)] if samples else []
        min_val = min(samples)
        max_val = max(samples)
        if min_val == max_val:
            return [len(samples)] + [0] * (num_bins - 1)
        bin_width = (max_val - min_val) / num_bins
        counts = [0] * num_bins
        for s in samples:
            idx = min(int((s - min_val) / bin_width), num_bins - 1)
            counts[idx] += 1
        return counts

    def write_json_report(
        self,
        filepath: str,
        pipeline_name: str = "perception_v1",
        total_frames: int = 0,
    ) -> str:
        """
        生成并写入包含结构化指标的 latency_report.json。

        成功时返回文件路径。
        """
        st = self.stats()
        stages_json = {}
        for name, s in st.items():
            if name == "end_to_end":
                continue
            with self._lock:
                samples = self._samples.get(name, [])
            stages_json[name] = {
                "mean_us": round(s["mean_us"], 2),
                "p50_us": round(s["p50_us"], 2),
                "p99_us": round(s["p99_us"], 2),
                "p95_us": round(s["p95_us"], 2),
                "max_us": round(s["max_us"], 2),
                "min_us": round(s["min_us"], 2),
                "stddev_us": round(s["stddev_us"], 2),
                "count": int(s["count"]),
                "histogram": self._histogram_bins(samples),
            }

        e2e_json = {}
        if "end_to_end" in st:
            e2e = st["end_to_end"]
            with self._lock:
                e2e_samples = self._samples.get("end_to_end", [])
            e2e_json = {
                "mean_us": round(e2e["mean_us"], 2),
                "p50_us": round(e2e["p50_us"], 2),
                "p99_us": round(e2e["p99_us"], 2),
                "max_us": round(e2e["max_us"], 2),
                "min_us": round(e2e["min_us"], 2),
                "stddev_us": round(e2e["stddev_us"], 2),
                "histogram": self._histogram_bins(e2e_samples),
            }

        # 识别瓶颈: 均值最高（排除 e2e）的阶段
        stage_stats = {k: v for k, v in st.items() if k != "end_to_end"}
        bottleneck_name = "unknown"
        bottleneck_pct = 0.0
        if stage_stats:
            bottleneck_name = max(stage_stats, key=lambda k: stage_stats[k]["mean_us"])
            bottleneck_mean = stage_stats[bottleneck_name]["mean_us"]
            total_mean = sum(v["mean_us"] for v in stage_stats.values())
            bottleneck_pct = (
                round(bottleneck_mean / total_mean * 100.0, 1)
                if total_mean > 0
                else 0.0
            )

        report = {
            "pipeline": pipeline_name,
            "total_frames": total_frames,
            "stages": stages_json,
            "e2e": e2e_json,
            "bottleneck": bottleneck_name,
            "bottleneck_pct": bottleneck_pct,
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        return filepath

    def clear(self) -> None:
        """清空所有已记录的样本。"""
        with self._lock:
            self._samples.clear()
