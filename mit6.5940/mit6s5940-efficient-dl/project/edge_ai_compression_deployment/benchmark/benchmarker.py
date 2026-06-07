#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试模块 —— 测量模型的关键性能指标。

本模块测量以下指标：
1. 参数量（总参数 + 可训练参数）
2. FLOPs（手动计算，基于各层尺寸）
3. 模型文件大小（磁盘占用 MB）
4. CPU 延迟（单次推理时间，含 warmup + 多次测量）
5. 内存使用（运行时峰值内存，基于 psutil）
6. 批量 CPU 吞吐量（不同 batch size 下的处理速度）

所有方法均在 CPU 上运行，无需 GPU。

参考论文：
- MCUNet (Lin et al., NeurIPS 2020) - MCU 性能度量
- Once-for-All (Cai et al., ICLR 2020) - 多平台基准测试

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# 性能基准测试器
# ============================================================


class Benchmarker:
    """
    模型性能基准测试器 —— 全面评估模型的各项性能指标。

    在纯 CPU 环境下运行，测量参数量、FLOPs、延迟、内存、吞吐量。
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (3, 32, 32),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        参数：
            model: 待测试的模型
            input_shape: 输入张量形状 (C, H, W)
            device: 运行设备
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def run_full_benchmark(
        self,
        warmup_runs: int = 10,
        measure_runs: int = 100,
        batch_sizes: list[int] | None = None,
        measure_memory: bool = True,
        calculate_flops: bool = True,
    ) -> dict[str, Any]:
        """
        执行完整的基准测试套件。

        参数：
            warmup_runs: 预热运行次数
            measure_runs: 正式测量运行次数
            batch_sizes: 测试的 batch size 列表
            measure_memory: 是否测量内存
            calculate_flops: 是否计算 FLOPs

        返回：
            包含所有性能指标的字典
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 32, 64]

        logger.info("=" * 60)
        logger.info("开始性能基准测试")
        logger.info("=" * 60)

        results: dict[str, Any] = {}

        # 1. 参数量统计
        results["parameters"] = self.measure_parameters()

        # 2. FLOPs 计算
        if calculate_flops:
            results["flops"] = self.calculate_flops()

        # 3. 模型大小
        results["model_size"] = self.measure_model_size()

        # 4. CPU 延迟
        results["latency"] = self.measure_cpu_latency(warmup_runs, measure_runs)

        # 5. 内存使用
        if measure_memory:
            results["memory"] = self.measure_memory_usage()

        # 6. 批量吞吐量
        results["throughput"] = self.measure_batch_throughput(batch_sizes)

        # 汇总
        results["summary"] = self._generate_summary(results)

        logger.info("=" * 60)
        logger.info("基准测试完成")
        self._log_summary(results["summary"])
        logger.info("=" * 60)

        return results

    def measure_parameters(self) -> dict[str, Any]:
        """
        统计模型参数量。

        返回：
            dict: 包含总数、可训练数、各类型参数的详细信息
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        per_type: dict[str, int] = {}
        per_layer: dict[str, int] = {}

        for name, param in self.model.named_parameters():
            num = param.numel()
            total_params += num

            if param.requires_grad:
                trainable_params += num
            else:
                non_trainable_params += num

            # 按类型统计
            param_type = type(param).__name__
            per_type[param_type] = per_type.get(param_type, 0) + num

            # 按层统计
            per_layer[name] = num

        result = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "params_millions": round(total_params / 1e6, 2),
            "fp32_size_mb": round(total_params * 4 / (1024 * 1024), 2),
            "int8_size_mb": round(total_params * 1 / (1024 * 1024), 2),
            "int4_size_mb": round(total_params * 0.5 / (1024 * 1024), 2),
            "per_type": per_type,
            "top_layers_by_params": sorted(
                per_layer.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

        logger.info(
            "参数量: 总计=%s (%.2fM), 可训练=%s",
            f"{total_params:,}",
            total_params / 1e6,
            f"{trainable_params:,}",
        )

        return result

    def calculate_flops(self) -> dict[str, Any]:
        """
        手动计算模型的 FLOPs（浮点运算次数）。

        虽然 PyTorch 有 profile 工具，但手动计算更精确、无需额外依赖。

        卷积层 FLOPs 公式：
            FLOPs = 2 * K_h * K_w * C_in * C_out * H_out * W_out

        全连接层 FLOPs 公式：
            FLOPs = 2 * in_features * out_features

        返回：
            dict: 包含总分和各层 FLOPs 的字典
        """
        # 获取特征图输出尺寸
        with torch.no_grad():
            dummy = torch.randn(1, *self.input_shape)
            _ = self.model(dummy)

        total_flops = 0
        per_layer: dict[str, int] = {}
        feature_maps: dict[str, tuple[int, ...]] = {}
        current_input = None

        # 为每个模块注册 hook 来获取特征图尺寸
        hooks: list[Any] = []

        def get_shape_hook(name: str):
            def hook(module: nn.Module, _inp: Any, outp: Any) -> None:
                if isinstance(outp, torch.Tensor):
                    feature_maps[name] = tuple(outp.shape)

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(get_shape_hook(name)))

        # 重新跑一次前向，收集特征图尺寸
        with torch.no_grad():
            self.model(dummy)

        # 移除 hooks
        for hook in hooks:
            hook.remove()

        # 计算各层 FLOPs
        for name, module in self.model.named_modules():
            sp = name  # short name for logging

            if isinstance(module, nn.Conv2d):
                # 获取该层输出尺寸
                out_shape = feature_maps.get(name)
                if out_shape is None:
                    continue

                # 卷积 FLOPs = 2 * K_h * K_w * C_in * C_out * H_out * W_out
                k_h, k_w = module.kernel_size
                c_in = module.in_channels
                c_out = module.out_channels
                h_out, w_out = out_shape[2], out_shape[3]

                # 考虑 groups（分组卷积）
                flops = 2 * k_h * k_w * (c_in // module.groups) * c_out * h_out * w_out

                # 如果有 bias，加上 bias 加法
                if module.bias is not None:
                    flops += c_out * h_out * w_out

                total_flops += flops
                per_layer[name] = flops

            elif isinstance(module, nn.Linear):
                c_in = module.in_features
                c_out = module.out_features

                # 全连接 FLOPs = 2 * in * out
                flops = 2 * c_in * c_out

                if module.bias is not None:
                    flops += c_out

                total_flops += flops
                per_layer[name] = flops

        result = {
            "total_flops": total_flops,
            "total_mflops": round(total_flops / 1e6, 2),
            "total_gflops": round(total_flops / 1e9, 4),
            "per_layer": per_layer,
            "input_shape": (1, *self.input_shape),
        }

        logger.info(
            "FLOPs: 总计=%s (%.2f MFLOPs)", f"{total_flops:,}", total_flops / 1e6
        )

        return result

    def measure_model_size(self, save_dir: str | None = None) -> dict[str, Any]:
        """
        测量模型文件的磁盘大小。

        参数：
            save_dir: 保存临时模型文件的目录（默认 None 使用内存）

        返回：
            dict: 模型大小信息
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(self.model.state_dict(), tmp.name)
            file_size = os.path.getsize(tmp.name)
            os.unlink(tmp.name)

        total_params = sum(p.numel() for p in self.model.parameters())

        result = {
            "disk_size_bytes": file_size,
            "disk_size_mb": round(file_size / (1024 * 1024), 2),
            "disk_size_kb": round(file_size / 1024, 2),
            "fp32_in_memory_mb": round(total_params * 4 / (1024 * 1024), 2),
        }

        logger.info(
            "模型大小: 磁盘=%.2f MB, FP32内存=%.2f MB",
            result["disk_size_mb"],
            result["fp32_in_memory_mb"],
        )

        return result

    def measure_cpu_latency(
        self,
        warmup_runs: int = 10,
        measure_runs: int = 100,
    ) -> dict[str, Any]:
        """
        测量 CPU 推理延迟。

        使用 batch_size=1 的模型进行多次推理，取平均/中位数/标准差。

        参数：
            warmup_runs: 预热运行次数（不计入测量）
            measure_runs: 正式测量运行次数

        返回：
            dict: 延迟统计信息
        """
        self.model.eval()
        dummy_input = torch.randn(1, *self.input_shape, device=self.device)

        # 预热阶段 —— 让 CPU cache 和调度器热起来
        logger.info("CPU 延迟测试: 预热 %d 次...", warmup_runs)
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)

        # 强制同步，确保所有操作完成
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 正式测量阶段
        logger.info("CPU 延迟测试: 测量 %d 次...", measure_runs)
        latencies: list[float] = []

        for _ in range(measure_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start_time) * 1000)  # 转为毫秒

        latencies_arr = np.array(latencies)

        result = {
            "mean_ms": round(float(latencies_arr.mean()), 3),
            "median_ms": round(float(np.median(latencies_arr)), 3),
            "std_ms": round(float(latencies_arr.std()), 3),
            "min_ms": round(float(latencies_arr.min()), 3),
            "max_ms": round(float(latencies_arr.max()), 3),
            "p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
        }

        logger.info(
            "CPU 延迟: 平均=%.3fms, 中位数=%.3fms, P95=%.3fms",
            result["mean_ms"],
            result["median_ms"],
            result["p95_ms"],
        )

        return result

    def measure_memory_usage(self) -> dict[str, Any]:
        """
        估算模型运行时的内存使用。

        使用 psutil 获取物理内存变化，或退还基于参数计算的估算值。

        返回：
            dict: 内存使用信息
        """
        try:
            import psutil  # type: ignore

            process = psutil.Process()

            # 记录基线内存
            _ = process.memory_info().rss
            mem_before = process.memory_info().rss

            # 运行一次推理
            dummy_input = torch.randn(1, *self.input_shape, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

            mem_after = process.memory_info().rss
            mem_used = max(0, mem_after - mem_before)

            result = {
                "method": "psutil",
                "peak_rss_bytes": mem_used,
                "peak_rss_mb": round(mem_used / (1024 * 1024), 2),
            }
        except ImportError:
            # 如果没有 psutil，使用参数 + 中间激活的估计值
            total_params = sum(p.numel() for p in self.model.parameters())
            # 估算中间激活：batch × C × H × W × 4bytes × 保守系数 2
            estimated_activations = (
                1
                * self.input_shape[0]
                * self.input_shape[1]
                * self.input_shape[2]
                * 4
                * 2
            )
            total_memory = total_params * 4 + estimated_activations

            result = {
                "method": "estimated",
                "estimated_bytes": total_memory,
                "estimated_mb": round(total_memory / (1024 * 1024), 2),
                "note": "psutil 未安装，使用参数估算。安装: pip install psutil",
            }

        logger.info(
            "内存使用: %s MB",
            result.get("peak_rss_mb", result.get("estimated_mb", "N/A")),
        )

        return result

    def measure_batch_throughput(
        self,
        batch_sizes: list[int],
        num_batches: int = 50,
    ) -> dict[str, Any]:
        """
        测量不同 batch size 下的 CPU 吞吐量。

        参数：
            batch_sizes: 测试的 batch size 列表
            num_batches: 每个 batch size 测试的批次数量

        返回：
            dict: 各 batch size 下的吞吐量信息
        """
        self.model.eval()
        throughputs: dict[str, Any] = {}

        for bs in batch_sizes:
            dummy_input = torch.randn(bs, *self.input_shape, device=self.device)

            # 预热
            with torch.no_grad():
                _ = self.model(dummy_input)

            # 测量
            start_time = time.perf_counter()
            total_samples = 0
            for _ in range(num_batches):
                with torch.no_grad():
                    _ = self.model(dummy_input)
                total_samples += bs
            elapsed = time.perf_counter() - start_time

            samples_per_sec = total_samples / elapsed
            ms_per_batch = (elapsed / num_batches) * 1000
            ms_per_sample = ms_per_batch / bs

            throughputs[f"batch_{bs}"] = {
                "batch_size": bs,
                "num_batches": num_batches,
                "total_samples": total_samples,
                "elapsed_seconds": round(elapsed, 3),
                "samples_per_second": round(samples_per_sec, 1),
                "ms_per_batch": round(ms_per_batch, 2),
                "ms_per_sample": round(ms_per_sample, 2),
            }

            logger.info(
                "吞吐量 bs=%d: %.1f samples/sec (%.2f ms/batch)",
                bs,
                samples_per_sec,
                ms_per_batch,
            )

        return throughputs

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """从详细结果中提取关键摘要。"""
        summary: dict[str, Any] = {}

        if "parameters" in results:
            p = results["parameters"]
            summary["params_M"] = p.get("params_millions", "N/A")

        if "flops" in results:
            f = results["flops"]
            summary["flops_M"] = f.get("total_mflops", "N/A")

        if "model_size" in results:
            s = results["model_size"]
            summary["size_MB"] = s.get("disk_size_mb", "N/A")

        if "latency" in results:
            l = results["latency"]
            summary["latency_ms"] = l.get("mean_ms", "N/A")

        if "memory" in results:
            m = results["memory"]
            summary["memory_MB"] = m.get("peak_rss_mb") or m.get("estimated_mb", "N/A")

        return summary

    def _log_summary(self, summary: dict[str, Any]) -> None:
        """输出基准测试摘要到日志。"""
        lines = [
            "┌─────────────────────────────────────┐",
            "│        基准测试摘要 (Benchmark Summary)       │",
            "├─────────────────────────────────────┤",
            f"│  参数量:     {summary.get('params_M', 'N/A'):>8} M  │",
            f"│  FLOPs:      {summary.get('flops_M', 'N/A'):>8} M  │",
            f"│  模型大小:   {summary.get('size_MB', 'N/A'):>8} MB │",
            f"│  推理延迟:   {summary.get('latency_ms', 'N/A'):>8} ms │",
            f"│  内存使用:   {summary.get('memory_MB', 'N/A'):>8} MB │",
            "└─────────────────────────────────────┘",
        ]
        for line in lines:
            logger.info(line)


# ============================================================
# 便捷函数
# ============================================================


def benchmark_model(
    model: nn.Module,
    input_shape: tuple[int, ...] = (3, 32, 32),
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    一键对模型执行完整基准测试。

    参数：
        model: 待测试的模型
        input_shape: 输入形状
        config: 基准测试配置字典（可选）

    返回：
        完整的基准测试结果
    """
    if config is None:
        config = {}

    benchmarker = Benchmarker(model, input_shape=input_shape)

    return benchmarker.run_full_benchmark(
        warmup_runs=config.get("warmup_runs", 10),
        measure_runs=config.get("measure_runs", 100),
        batch_sizes=config.get("batch_sizes", [1, 4, 16, 32, 64]),
        measure_memory=config.get("measure_memory", True),
        calculate_flops=config.get("calculate_flops", True),
    )
