#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出模块 —— 将 PyTorch 模型导出为 ONNX 格式。

本模块实现：
1. PyTorch → ONNX 导出（支持动态 batch）
2. ONNX Runtime 推理（验证导出正确性）
3. TensorRT 工作流模拟（无 GPU 环境下的概念验证）
4. 模型大小报告生成

参考论文：
- MCUNet (Lin et al., NeurIPS 2020) - TinyEngine 代码生成
- Once-for-All (Cai et al., ICLR 2020) - 多平台部署

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# ONNX 导出器
# ============================================================


class ONNXExporter:
    """
    ONNX 模型导出器 —— 将 PyTorch 模型导出为 ONNX 格式。

    ONNX (Open Neural Network Exchange) 是一种通用的模型表示格式，
    被 TensorRT、ONNX Runtime、OpenVINO 等多种推理引擎支持。
    """

    def __init__(
        self,
        model: nn.Module,
        export_path: str = "./output/model.onnx",
        input_shape: list[int] | None = None,
        opset_version: int = 13,
        dynamic_batch: bool = True,
        optimize: bool = False,
    ) -> None:
        """
        参数：
            model: 待导出的 PyTorch 模型
            export_path: ONNX 文件保存路径
            input_shape: 输入张量形状 [C, H, W]，默认 [3, 32, 32]
            opset_version: ONNX opset 版本
            dynamic_batch: 是否导出动态 batch size
            optimize: 是否使用 ONNX simplifier 优化（需要 pip install onnx-simplifier）
        """
        self.model = model
        self.export_path = export_path
        self.input_shape = input_shape or [3, 32, 32]
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
        self.optimize = optimize

        # 确保输出目录存在
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

    def export(self) -> dict[str, Any]:
        """
        执行 ONNX 导出。

        返回：
            dict: 导出结果信息（文件路径、大小、是否成功等）
        """
        logger.info("开始 ONNX 导出: %s", self.export_path)

        try:
            import onnx  # type: ignore
        except ImportError:
            logger.error("ONNX 库未安装。请执行: pip install onnx onnxruntime")
            return {"success": False, "error": "ONNX 库未安装"}

        self.model.eval()
        self.model.cpu()

        # 创建示例输入张量
        batch_size = 1 if not self.dynamic_batch else 1
        dummy_input = torch.randn(batch_size, *self.input_shape, device="cpu")

        # 配置动态轴
        dynamic_axes: dict[str, dict[int, str]] | None = None
        if self.dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        # ONNX 导出参数
        export_kwargs: dict[str, Any] = {
            "export_params": True,  # 导出模型参数
            "opset_version": self.opset_version,
            "do_constant_folding": True,  # 执行常量折叠优化
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": dynamic_axes,
        }

        try:
            # 执行 ONNX 导出
            torch.onnx.export(
                self.model,
                dummy_input,
                self.export_path,
                **export_kwargs,
            )
            logger.info("ONNX 导出成功: %s", self.export_path)

        except Exception as e:
            logger.error("ONNX 导出失败: %s", e)
            return {"success": False, "error": str(e)}

        # 验证导出的模型
        try:
            onnx_model = onnx.load(self.export_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX 模型验证通过")
        except Exception as e:
            logger.warning("ONNX 模型验证失败: %s", e)

        # 获取文件大小
        file_size = os.path.getsize(self.export_path)
        file_size_mb = file_size / (1024 * 1024)

        # 可选：使用 onnx-simplifier 优化
        if self.optimize:
            try:
                import onnxsim  # type: ignore

                logger.info("使用 onnx-simplifier 优化模型...")
                model_simp, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(model_simp, self.export_path)
                    logger.info("ONNX 优化完成")
                else:
                    logger.warning("ONNX 优化失败，保留原始模型")
            except ImportError:
                logger.warning(
                    "onnx-simplifier 未安装，跳过优化。"
                    "安装命令: pip install onnx-simplifier"
                )

        model_info = {
            "success": True,
            "path": self.export_path,
            "size_bytes": file_size,
            "size_mb": round(file_size_mb, 2),
            "opset_version": self.opset_version,
            "dynamic_batch": self.dynamic_batch,
            "input_shape": [batch_size, *self.input_shape],
        }

        logger.info(
            "ONNX 导出信息: 大小=%.2f MB, opset=%d", file_size_mb, self.opset_version
        )

        return model_info


# ============================================================
# ONNX Runtime 推理器
# ============================================================


class ONNXRuntimeInference:
    """
    ONNX Runtime 推理引擎 —— 使用 ONNX Runtime 加载并执行 ONNX 模型。

    用途：
    1. 验证 ONNX 导出后模型的输出与原始 PyTorch 模型一致
    2. 测量 ONNX 模型的 CPU 推理性能
    """

    def __init__(self, onnx_path: str) -> None:
        """
        参数：
            onnx_path: ONNX 模型文件路径
        """
        self.onnx_path = onnx_path
        self.session: Any = None

        try:
            import onnxruntime as ort  # type: ignore

            self.ort = ort
        except ImportError:
            raise ImportError("ONNX Runtime 未安装。请执行: pip install onnxruntime")

    def load(self) -> None:
        """加载 ONNX 模型到 ONNX Runtime 会话。"""
        logger.info("加载 ONNX 模型: %s", self.onnx_path)

        sess_options = self.ort.SessionOptions()
        sess_options.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # 使用 CPU 执行提供程序
        self.session = self.ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        input_info = self.session.get_inputs()
        output_info = self.session.get_outputs()

        logger.info("ONNX Runtime 会话已创建")
        for inp in input_info:
            logger.debug("  输入: %s (shape=%s)", inp.name, inp.shape)
        for out in output_info:
            logger.debug("  输出: %s (shape=%s)", out.name, out.shape)

    def infer(self, input_data: Any) -> Any:
        """
        执行 ONNX 推理。

        参数：
            input_data: NumPy 数组或 PyTorch 张量

        返回：
            推理结果（NumPy 数组）
        """
        if self.session is None:
            self.load()

        # 转换为 NumPy 数组
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        outputs = self.session.run(None, {"input": input_data})
        return outputs[0]


# ============================================================
# TensorRT 工作流模拟（无 GPU 环境的概念验证）
# ============================================================


class TensorRTSimulator:
    """
    TensorRT 工作流模拟器 —— 在无 GPU/无 TensorRT 的环境下模拟其工作流。

    TensorRT 是 NVIDIA 的高性能深度学习推理引擎，主要步骤：
    1. 解析 ONNX 模型
    2. 图优化（层融合、常量折叠、kernel 自动调优）
    3. 生成优化后的推理引擎

    由于环境限制（无 GPU），本模块模拟上述流程并生成概念报告。
    """

    def __init__(self, onnx_path: str) -> None:
        """
        参数：
            onnx_path: ONNX 模型文件路径
        """
        self.onnx_path = onnx_path
        self.model_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0

    def simulate(self) -> dict[str, Any]:
        """
        模拟 TensorRT 优化流程并生成报告。

        返回：
            dict: 包含模拟优化结果的字典
        """
        logger.info("TensorRT 工作流模拟开始（无 GPU 环境的概念验证）")

        # 模拟的优化步骤
        simulation_steps = [
            "1. 解析 ONNX 模型图",
            "2. 识别可融合的层（Conv+BN+ReLU → CBR 融合）",
            "3. 消除不必要的 reshape/transpose 操作",
            "4. 常量折叠（预计算常量表达式）",
            "5. 选择最优 kernel 实现（精度校准中）",
            "6. INT8 校准表生成（需要校准数据集）",
        ]

        # 模拟的优化效果（基于经验数据）
        original_size_mb = self.model_size / (1024 * 1024)
        # TensorRT 优化后预期大小：FP32 → ~1.0x, FP16 → ~0.5x, INT8 → ~0.25x
        estimated_fp16_size = original_size_mb * 0.5
        estimated_int8_size = original_size_mb * 0.25

        # 模拟的延迟提升
        estimated_latency_improvement_fp32 = 1.5  # 1.5x 加速
        estimated_latency_improvement_fp16 = 2.0  # 2x 加速
        estimated_latency_improvement_int8 = 3.0  # 3x 加速

        report = {
            "simulated": True,
            "note": "本报告为 TensorRT 工作流的概念验证模拟，无实际 GPU 环境。",
            "steps": simulation_steps,
            "original_size_mb": round(original_size_mb, 2),
            "estimated_optimizations": {
                "FP32": {
                    "size_mb": round(original_size_mb, 2),
                    "latency_improvement": f"{estimated_latency_improvement_fp32:.1f}x",
                },
                "FP16": {
                    "size_mb": round(estimated_fp16_size, 2),
                    "latency_improvement": f"{estimated_latency_improvement_fp16:.1f}x",
                },
                "INT8": {
                    "size_mb": round(estimated_int8_size, 2),
                    "latency_improvement": f"{estimated_latency_improvement_int8:.1f}x",
                },
            },
        }

        logger.info("TensorRT 模拟完成")
        return report


# ============================================================
# 模型大小报告生成器
# ============================================================


def generate_model_size_report(
    pytorch_model: nn.Module,
    onnx_path: str | None = None,
    export_dir: str = "./output",
) -> dict[str, Any]:
    """
    生成模型大小对比报告。

    对比 PyTorch 模型（.pt）与多种导出格式的模型大小。

    参数：
        pytorch_model: PyTorch 模型
        onnx_path: ONNX 文件路径
        export_dir: 导出目录

    返回：
        包含各种格式模型大小的字典
    """
    report: dict[str, Any] = {}

    # PyTorch 模型大小（保存为 .pt）
    pt_path = os.path.join(export_dir, "model.pt")
    os.makedirs(export_dir, exist_ok=True)

    torch.save(pytorch_model.state_dict(), pt_path)
    pt_size = os.path.getsize(pt_path)
    report["pytorch_pt"] = {
        "path": pt_path,
        "size_bytes": pt_size,
        "size_mb": round(pt_size / (1024 * 1024), 2),
    }

    # PyTorch 模型大小（保存为 TorchScript）
    try:
        ts_path = os.path.join(export_dir, "model_scripted.pt")
        scripted = torch.jit.script(pytorch_model)
        torch.jit.save(scripted, ts_path)
        ts_size = os.path.getsize(ts_path)
        report["torchscript"] = {
            "path": ts_path,
            "size_bytes": ts_size,
            "size_mb": round(ts_size / (1024 * 1024), 2),
        }
    except Exception as e:
        logger.warning("TorchScript 导出失败: %s", e)
        report["torchscript"] = {"error": str(e)}

    # ONNX 模型大小
    if onnx_path and os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path)
        report["onnx"] = {
            "path": onnx_path,
            "size_bytes": onnx_size,
            "size_mb": round(onnx_size / (1024 * 1024), 2),
        }

    # 参数量
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    report["parameters"] = {
        "total": total_params,
        "fp32_memory_mb": round(total_params * 4 / (1024 * 1024), 2),
        "int8_memory_mb": round(total_params * 1 / (1024 * 1024), 2),
        "int4_memory_mb": round(total_params * 0.5 / (1024 * 1024), 2),
    }

    logger.info(
        "模型大小报告: PyTorch=%.2f MB, 参数=%.2f M",
        pt_size / (1024 * 1024),
        total_params / 1e6,
    )

    return report


# ============================================================
# 导出器工厂函数
# ============================================================


def export_model(
    model: nn.Module,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    便捷函数：一键导出模型并生成报告。

    参数：
        model: PyTorch 模型
        config: 导出配置字典

    返回：
        包含导出结果和报告的字典
    """
    format_type = config.get("format", "onnx")
    export_path = config.get("onnx_path", "./output/model.onnx")
    input_shape = config.get("input_shape", [3, 32, 32])
    opset = config.get("opset_version", 13)
    dynamic_batch = config.get("dynamic_batch", True)
    optimize = config.get("optimize", False)
    simulate_trt = config.get("simulate_tensorrt", True)

    results: dict[str, Any] = {}

    if format_type == "onnx":
        # ONNX 导出
        exporter = ONNXExporter(
            model=model,
            export_path=export_path,
            input_shape=input_shape,
            opset_version=opset,
            dynamic_batch=dynamic_batch,
            optimize=optimize,
        )
        onnx_result = exporter.export()
        results["onnx_export"] = onnx_result

        # ONNX 推理验证
        if onnx_result.get("success"):
            try:
                inf_session = ONNXRuntimeInference(export_path)
                inf_session.load()

                # 用随机输入验证推理一致性
                dummy = torch.randn(1, *input_shape)
                model.eval()
                with torch.no_grad():
                    pt_output = model(dummy).numpy()
                ort_output = inf_session.infer(dummy)

                # 比较输出
                diff = float((abs(pt_output - ort_output)).mean())
                results["onnx_verification"] = {
                    "mean_absolute_diff": diff,
                    "consistent": diff < 0.01,
                }
                logger.info("ONNX 推理验证: 平均差异=%.6f", diff)
            except Exception as e:
                results["onnx_verification"] = {"error": str(e)}
                logger.warning("ONNX 推理验证失败: %s", e)

        # 模型大小报告
        size_report = generate_model_size_report(model, export_path)
        results["size_report"] = size_report

        # TensorRT 模拟
        if simulate_trt:
            trt_sim = TensorRTSimulator(export_path)
            results["tensorrt_simulation"] = trt_sim.simulate()

    return results
