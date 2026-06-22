#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块 —— 自动生成 Markdown 格式的模型压缩对比报告。

本模块实现：
1. 对比报告生成（Markdown 表格 + Mermaid 图表）
2. 每个阶段的精度/参数量/FLOPs/延迟/内存汇总
3. 流水线架构 Mermaid 图
4. JSON 格式的原始实验数据导出

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert report payloads to JSON-safe values without losing simple structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


# ============================================================
# 报告生成器
# ============================================================


class ReportGenerator:
    """
    模型压缩流水线报告生成器 —— 汇总各阶段的性能数据并生成可视化报告。

    输出：
    - comparison_report.md：Markdown 格式的对比报告
    - experiment_results.json：原始实验数据
    """

    def __init__(self, output_dir: str = "./reports") -> None:
        """
        参数：
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 各阶段的结果收集
        self.stage_results: dict[str, dict[str, Any]] = {}

    def add_stage_result(
        self,
        stage_name: str,
        model_name: str,
        accuracy: float | None = None,
        benchmark: dict[str, Any] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> None:
        """
        添加一个阶段的性能数据。

        参数：
            stage_name: 阶段名称（如 "baseline", "pruned", "quantized"）
            model_name: 模型名称
            accuracy: Top-1 精度
            benchmark: 基准测试结果字典
            extra_info: 额外信息
        """
        entry: dict[str, Any] = {
            "stage": stage_name,
            "model": model_name,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
        }

        if benchmark is not None:
            summary = benchmark.get("summary", {})
            entry["params_M"] = summary.get("params_M", "N/A")
            entry["flops_M"] = summary.get("flops_M", "N/A")
            entry["size_MB"] = summary.get("size_MB", "N/A")
            entry["latency_ms"] = summary.get("latency_ms", "N/A")
            entry["memory_MB"] = summary.get("memory_MB", "N/A")

        if extra_info:
            entry["extra"] = _json_safe(extra_info)

        self.stage_results[stage_name] = entry
        logger.info("添加阶段结果: %s (精度=%.2f%%)", stage_name, accuracy or 0)

    def generate_comparison_report(
        self,
        title: str = "模型压缩流水线对比报告",
        pipeline_stages: list[str] | None = None,
    ) -> str:
        """
        生成 Markdown 格式的完整对比报告。

        参数：
            title: 报告标题
            pipeline_stages: 流水线阶段顺序列表

        返回：
            报告的 Markdown 字符串
        """
        if not self.stage_results:
            logger.warning("没有阶段数据，跳过报告生成")
            return ""

        if pipeline_stages is None:
            pipeline_stages = list(self.stage_results.keys())

        lines: list[str] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # === 报告头 ===
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**生成时间**: {now}")
        lines.append("")
        lines.append(
            "> 本报告由基准测试模块自动生成，汇总了模型压缩流水线各阶段的性能数据。"
        )
        lines.append("")

        # === 流水线架构图 ===
        lines.append("## 流水线架构")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph LR")
        for i, stage in enumerate(pipeline_stages):
            stage_id = stage.replace(" ", "_")
            lines.append(f"    {stage_id}[{stage}]")
            if i < len(pipeline_stages) - 1:
                next_id = pipeline_stages[i + 1].replace(" ", "_")
                lines.append(f"    {stage_id} --> {next_id}")
        lines.append("```")
        lines.append("")

        # === 总览表格 ===
        lines.append("## 性能对比总览")
        lines.append("")
        lines.append(
            "| 阶段 | 模型 | 精度 (%) | 参数量 (M) | FLOPs (M) | 模型大小 (MB) | CPU 延迟 (ms) | 内存 (MB) |"
        )
        lines.append(
            "|------|------|----------|-----------|-----------|-------------|-------------|----------|"
        )

        for stage in pipeline_stages:
            r = self.stage_results.get(stage)
            if r is None:
                continue
            acc_str = f"{r['accuracy']:.2f}" if r.get("accuracy") is not None else "N/A"
            params_str = str(r.get("params_M", "N/A"))
            flops_str = str(r.get("flops_M", "N/A"))
            size_str = str(r.get("size_MB", "N/A"))
            lat_str = f"{r.get('latency_ms', 'N/A')}"
            mem_str = str(r.get("memory_MB", "N/A"))

            lines.append(
                f"| **{stage}** | {r['model']} | {acc_str} | {params_str} | "
                f"{flops_str} | {size_str} | {lat_str} | {mem_str} |"
            )

        lines.append("")

        # === 各阶段详细数据 ===
        lines.append("## 各阶段详细数据")
        lines.append("")

        for stage in pipeline_stages:
            r = self.stage_results.get(stage)
            if r is None:
                continue

            lines.append(f"### {stage}")
            lines.append("")

            lines.append(f"- **模型**: {r['model']}")
            if r.get("accuracy") is not None:
                lines.append(f"- **精度**: {r['accuracy']:.2f}%")
            if "params_M" in r and r["params_M"] != "N/A":
                lines.append(f"- **参数量**: {r['params_M']} M")
            if "flops_M" in r and r["flops_M"] != "N/A":
                lines.append(f"- **FLOPs**: {r['flops_M']} M")
            if "size_MB" in r and r["size_MB"] != "N/A":
                lines.append(f"- **模型大小**: {r['size_MB']} MB")
            if "latency_ms" in r and r["latency_ms"] != "N/A":
                lines.append(f"- **CPU 延迟**: {r['latency_ms']} ms")
            if "memory_MB" in r and r["memory_MB"] != "N/A":
                lines.append(f"- **内存使用**: {r['memory_MB']} MB")

            extra = r.get("extra", {})
            if extra:
                lines.append(f"- **额外信息**: {json.dumps(_json_safe(extra), ensure_ascii=False)}")

            lines.append("")

        # === 压缩效果分析 ===
        lines.append("## 压缩效果分析")
        lines.append("")

        if "baseline" in self.stage_results:
            baseline = self.stage_results["baseline"]
            lines.append("### 相对基线的变化")
            lines.append("")

            for stage in pipeline_stages:
                if stage == "baseline":
                    continue
                r = self.stage_results.get(stage)
                if r is None:
                    continue

                lines.append(f"#### **{stage}**")
                lines.append("")

                # 精度变化
                if (
                    baseline.get("accuracy") is not None
                    and r.get("accuracy") is not None
                ):
                    acc_delta = r["accuracy"] - baseline["accuracy"]
                    lines.append(
                        f"- 精度变化: {'+' if acc_delta >= 0 else ''}{acc_delta:.2f}%"
                    )

                # 参数量变化
                if isinstance(baseline.get("params_M"), (int, float)) and isinstance(
                    r.get("params_M"), (int, float)
                ):
                    params_ratio = (
                        r["params_M"] / baseline["params_M"] * 100
                        if baseline["params_M"] > 0
                        else 0
                    )
                    params_reduction = 100 - params_ratio
                    lines.append(
                        f"- 参数量: {params_ratio:.1f}% (压缩 {params_reduction:.1f}%)"
                    )

                # 模型大小变化
                if isinstance(baseline.get("size_MB"), (int, float)) and isinstance(
                    r.get("size_MB"), (int, float)
                ):
                    size_ratio = (
                        r["size_MB"] / baseline["size_MB"] * 100
                        if baseline["size_MB"] > 0
                        else 0
                    )
                    lines.append(
                        f"- 模型大小: {size_ratio:.1f}% (压缩 {100 - size_ratio:.1f}%)"
                    )

                # 延迟变化
                if isinstance(baseline.get("latency_ms"), (int, float)) and isinstance(
                    r.get("latency_ms"), (int, float)
                ):
                    speedup = (
                        baseline["latency_ms"] / r["latency_ms"]
                        if r["latency_ms"] > 0
                        else 0
                    )
                    lines.append(f"- 推理加速: {speedup:.2f}×")

                lines.append("")

        # === 技术方法说明 ===
        lines.append("## 技术方法说明")
        lines.append("")
        lines.append("| 技术 | 方法 | 参考论文 |")
        lines.append("|------|------|----------|")
        lines.append(
            "| 剪枝 | 幅度剪枝/通道剪枝/渐进剪枝 | Deep Compression (Han et al., ICLR 2016) |"
        )
        lines.append(
            "| 量化 | PTQ/QAT, INT8/INT4/INT2, per-tensor/per-channel | AWQ (Lin et al., MLSys 2024), SmoothQuant (Xiao et al., ICML 2023) |"
        )
        lines.append("| 蒸馏 | KD 蒸馏 + 特征蒸馏 | Hinton et al., 2015 |")
        lines.append("| 导出 | ONNX + ONNX Runtime | - |")
        lines.append(
            "| 基准测试 | 参数量/FLOPs/延迟/内存/吞吐量 | MCUNet (Lin et al., NeurIPS 2020) |"
        )
        lines.append("")

        # === 环境信息 ===
        lines.append("## 运行环境")
        lines.append("")

        import platform
        import torch

        lines.append(f"- **操作系统**: {platform.system()} {platform.release()}")
        lines.append(f"- **Python 版本**: {platform.python_version()}")
        lines.append(f"- **PyTorch 版本**: {torch.__version__}")
        lines.append(f"- **设备**: CPU (无 GPU)")
        lines.append("")

        # === 页脚 ===
        lines.append("---")
        lines.append("")
        lines.append(f"*报告由 ReportGenerator 自动生成于 {now}*")
        lines.append("")

        # 保存报告
        report_path = self.output_dir / "comparison_report.md"
        report_content = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("对比报告已保存: %s", report_path)

        return report_content

    def save_raw_results(self) -> str:
        """
        保存原始实验数据为 JSON 文件。

        返回：
            JSON 文件路径
        """
        json_path = self.output_dir / "experiment_results.json"

        data = {
            "generated_at": datetime.now().isoformat(),
            "stages": self.stage_results,
            "num_stages": len(self.stage_results),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("原始结果已保存: %s", json_path)

        return str(json_path)

    def generate_mermaid_pipeline_diagram(
        self,
        stages: list[str],
    ) -> str:
        """
        生成流水线的 Mermaid 图表代码。

        参数：
            stages: 阶段列表

        返回：
            Mermaid 图表代码字符串
        """
        lines = ["```mermaid", "graph TB"]

        # 设置样式
        styles = [
            ("baseline", "#4A90D9", "#fff"),
            ("prune", "#E74C3C", "#fff"),
            ("quantize", "#F39C12", "#fff"),
            ("distill", "#2ECC71", "#fff"),
            ("export", "#9B59B6", "#fff"),
            ("benchmark", "#1ABC9C", "#fff"),
        ]

        for i, stage in enumerate(stages):
            stage_id = stage.replace(" ", "_")
            # 匹配样式
            color, font_color = "#95A5A6", "#fff"
            for keyword, c, fc in styles:
                if keyword in stage.lower():
                    color, font_color = c, fc
                    break
            lines.append(f"    {stage_id}[{stage}]")
            if i < len(stages) - 1:
                next_id = stages[i + 1].replace(" ", "_")
                lines.append(f"    {stage_id} --> {next_id}")

        lines.append("```")

        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================


def create_report(
    stage_results: dict[str, dict[str, Any]],
    output_dir: str = "./reports",
    title: str = "模型压缩流水线对比报告",
) -> tuple[str, str]:
    """
    便捷函数：一键生成对比报告和原始数据 JSON。

    参数：
        stage_results: 各阶段的性能数据
        output_dir: 输出目录
        title: 报告标题

    返回：
        (报告路径, JSON 数据路径)
    """
    generator = ReportGenerator(output_dir)

    # 添加各阶段数据
    for stage_name, data in stage_results.items():
        generator.add_stage_result(
            stage_name=stage_name,
            model_name=data.get("model", stage_name),
            accuracy=data.get("accuracy"),
            benchmark=data.get("benchmark"),
            extra_info=data.get("extra"),
        )

    # 生成报告
    generator.generate_comparison_report(title=title)

    # 保存原始数据
    json_path = generator.save_raw_results()

    report_path = str(Path(output_dir) / "comparison_report.md")

    return report_path, json_path
