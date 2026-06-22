#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口文件 —— 边缘 AI 模型压缩与部署全链路流水线。

MIT 6.5940 高效深度学习课程最终项目。

本脚本支持以下运行模式：
- train:        训练基线模型
- prune:        模型剪枝
- quantize:     模型量化（PTQ/QAT）
- distill:      知识蒸馏
- export:       导出 ONNX 模型
- benchmark:    性能基准测试
- full_pipeline: 执行完整压缩流水线（train → prune → quantize → distill → export → benchmark）

用法示例：
    python main.py --mode full_pipeline
    python main.py --mode train --model mobilenetv2 --epochs 50
    python main.py --mode prune --sparsity 0.5
    python main.py --mode quantize --bits 8 --scheme per_channel

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 设置项目根路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 导入项目模块
# ============================================================
from models.model_factory import create_model, get_model_info, list_available_models
from compression.pruner import create_pruner, SensitivityAnalyzer
from quantization.quantizer import create_quantizer
from distillation.distiller import Distiller
from export.exporter import export_model
from benchmark.benchmarker import Benchmarker
from reports.report_generator import ReportGenerator

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ============================================================
# 配置加载
# ============================================================


def load_config(config_path: str) -> dict[str, Any]:
    """
    从 YAML 文件加载配置。

    参数：
        config_path: YAML 配置文件路径

    返回：
        配置字典

    异常：
        FileNotFoundError: 配置文件不存在时抛出
        yaml.YAMLError: YAML 解析错误时抛出
    """
    if not os.path.exists(config_path):
        logger.error("配置文件不存在: %s", config_path)
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("配置文件已加载: %s", config_path)
    return config


# ============================================================
# 合成数据集（替代 CIFAR-10，因 torchvision 不可用）
# ============================================================


class SyntheticDataset(Dataset):
    """Synthetic image-classification dataset replacing torchvision.datasets.CIFAR10.

    Yields random float32 tensors of shape (C, H, W) and random integer
    labels.  Useful for benchmarking and algorithm validation when
    torchvision is unavailable.
    """

    def __init__(
        self,
        num_samples: int,
        img_size: int = 32,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = torch.randn(self.in_channels, self.img_size, self.img_size)
        label = int(torch.randint(0, self.num_classes, (1,)).item())
        return img, label


# ============================================================
# 数据加载
# ============================================================


def get_data_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """
    根据配置创建数据加载器（合成数据，替代 CIFAR-10）。

    参数：
        config: 配置字典

    返回：
        (train_loader, test_loader)
    """
    batch_size = config["training"]["batch_size"]
    model_cfg = config["model"]
    img_size = model_cfg.get("input_size", 32)
    num_classes = model_cfg.get("num_classes", 10)

    train_samples = config["training"].get("train_samples", 50000)
    test_samples = config["training"].get("test_samples", 10000)

    logger.info("创建合成数据集（替代 CIFAR-10，因 torchvision 不可用）...")
    train_dataset = SyntheticDataset(
        num_samples=train_samples,
        img_size=img_size,
        num_classes=num_classes,
    )
    test_dataset = SyntheticDataset(
        num_samples=test_samples,
        img_size=img_size,
        num_classes=num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(
        "数据加载完成: 训练样本=%d, 测试样本=%d, batch_size=%d",
        len(train_dataset),
        len(test_dataset),
        batch_size,
    )

    return train_loader, test_loader


# ============================================================
# 训练函数
# ============================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    返回：
        (平均损失, 训练精度)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            logger.debug("Batch %d: loss=%.4f", batch_idx, loss.item())

    return total_loss / len(dataloader), correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    在验证集上评估模型精度。

    返回：
        Top-1 准确率 (0.0 ~ 1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total


def run_training(
    config: dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, Any]:
    """
    执行基线模型训练。

    参数：
        config: 全局配置
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器

    返回：
        字典: {"model": 模型, "accuracy": 精度, "model_info": 模型信息}
    """
    logger.info("=" * 60)
    logger.info("阶段 1: 基线模型训练")
    logger.info("=" * 60)

    cfg = config["training"]
    model_cfg = config["model"]
    device = torch.device(config["general"]["device"])

    # 创建模型
    model = create_model(
        model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
    )
    model.to(device)

    # 获取模型信息
    info = get_model_info(model)
    logger.info("模型信息: %s", info)

    # 优化器和调度器
    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
    elif cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["lr_milestones"],
        gamma=cfg["lr_gamma"],
    )

    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    os.makedirs(config["general"]["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(
        config["general"]["checkpoint_dir"], "baseline_model.pth"
    )

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_acc = validate(model, test_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d: loss=%.4f, train_acc=%.2f%%, val_acc=%.2f%%, lr=%.6f",
            epoch,
            cfg["epochs"],
            train_loss,
            train_acc * 100,
            val_acc * 100,
            lr,
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  最佳模型已保存 (acc=%.2f%%)", best_acc * 100)

    # 加载最佳模型
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info("基线训练完成: 最佳精度=%.2f%%", best_acc * 100)

    return {
        "model": model,
        "accuracy": best_acc,
        "model_info": info,
    }


# ============================================================
# 剪枝阶段
# ============================================================


def run_pruning(
    config: dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, Any]:
    """
    执行模型剪枝。

    参数：
        config: 全局配置
        model: 待剪枝的模型
        train_loader: 训练数据加载器（用于 finetune）
        test_loader: 测试数据加载器

    返回：
        {"model": 剪枝后模型, "sparsity": 实际稀疏度, "accuracy": 精度}
    """
    logger.info("=" * 60)
    logger.info("阶段 2: 模型剪枝")
    logger.info("=" * 60)

    prune_cfg = config["pruning"]
    device = torch.device(config["general"]["device"])

    # 创建剪枝器
    pruner = create_pruner(model, method=prune_cfg["method"])

    # 执行剪枝
    pruned_model = pruner.prune(sparsity=prune_cfg["sparsity"])

    # 剪枝后微调
    logger.info("剪枝后微调 (%d epochs)...", prune_cfg["finetune_epochs"])
    optimizer = optim.SGD(
        pruned_model.parameters(),
        lr=prune_cfg["finetune_lr"],
        momentum=0.9,
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, prune_cfg["finetune_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            pruned_model, train_loader, optimizer, criterion, device
        )
        val_acc = validate(pruned_model, test_loader, device)
        logger.info(
            "Finetune Epoch %d/%d: loss=%.4f, val_acc=%.2f%%",
            epoch,
            prune_cfg["finetune_epochs"],
            train_loss,
            val_acc * 100,
        )

    # 获取最终精度
    final_accuracy = validate(pruned_model, test_loader, device)

    logger.info(
        "剪枝完成: 稀疏度=%.2f%%, 精度=%.2f%%",
        pruner.sparsity_applied * 100,
        final_accuracy * 100,
    )

    return {
        "model": pruned_model,
        "sparsity": pruner.sparsity_applied,
        "accuracy": final_accuracy,
    }


# ============================================================
# 量化阶段
# ============================================================


def run_quantization(
    config: dict[str, Any],
    model: nn.Module,
    test_loader: DataLoader,
) -> dict[str, Any]:
    """
    执行模型量化（PTQ 或 QAT）。

    参数：
        config: 全局配置
        model: 待量化的模型
        test_loader: 测试数据加载器

    返回：
        {"model": 量化后模型, "bits": 位宽, "accuracy": 精度}
    """
    logger.info("=" * 60)
    logger.info("阶段 3: 模型量化")
    logger.info("=" * 60)

    quant_cfg = config["quantization"]
    device = torch.device(config["general"]["device"])

    # 创建量化器
    quantizer = create_quantizer(
        bits=quant_cfg["bits"],
        scheme=quant_cfg["scheme"],
        method=quant_cfg["method"],
        symmetric=quant_cfg.get("symmetric", True),
    )

    # 如果是 PTQ，执行校准
    if quant_cfg["method"] == "ptq" and hasattr(quantizer, "calibrate"):
        # 创建小型校准集
        logger.info("PTQ 校准...")
        quantizer.calibrate(model, test_loader, num_batches=10)

    # 执行量化
    quantized_model = quantizer.quantize(model)

    # 评估量化后精度
    accuracy = validate(quantized_model, test_loader, device)

    logger.info("量化完成: %d-bit, 精度=%.2f%%", quant_cfg["bits"], accuracy * 100)

    return {
        "model": quantized_model,
        "bits": quant_cfg["bits"],
        "scheme": quant_cfg["scheme"],
        "accuracy": accuracy,
    }


# ============================================================
# 蒸馏阶段
# ============================================================


def run_distillation(
    config: dict[str, Any],
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, Any]:
    """
    执行知识蒸馏。

    参数：
        config: 全局配置
        student_model: 学生模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器

    返回：
        {"model": 蒸馏后模型, "accuracy": 精度}
    """
    logger.info("=" * 60)
    logger.info("阶段 4: 知识蒸馏")
    logger.info("=" * 60)

    dist_cfg = config["distillation"]
    train_cfg = config["training"]
    device = torch.device(config["general"]["device"])

    # 创建教师模型
    teacher_arch = dist_cfg["teacher_architecture"]
    logger.info("创建教师模型: %s", teacher_arch)
    teacher_model = create_model(
        teacher_arch, num_classes=config["model"]["num_classes"]
    )
    teacher_model.to(device)

    # 如果提供了教师模型检查点，加载它；否则快速训练一个
    teacher_ckpt = dist_cfg.get("teacher_checkpoint", "")
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        logger.info("教师模型权重已加载: %s", teacher_ckpt)
    else:
        logger.info("训练教师模型...")
        teacher_opt = optim.SGD(
            teacher_model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )
        teacher_sched = optim.lr_scheduler.CosineAnnealingLR(teacher_opt, T_max=10)
        criterion = nn.CrossEntropyLoss()

        teacher_epochs = dist_cfg.get("teacher_epochs", 10)
        for epoch in range(1, teacher_epochs + 1):
            train_loss, train_acc = train_epoch(
                teacher_model, train_loader, teacher_opt, criterion, device
            )
            teacher_sched.step()
            if epoch % 3 == 0:
                logger.info("教师 epoch %d: loss=%.4f", epoch, train_loss)

    teacher_acc = validate(teacher_model, test_loader, device)
    logger.info("教师模型精度: %.2f%%", teacher_acc * 100)

    # 创建蒸馏器
    distiller = Distiller(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=dist_cfg["temperature"],
        alpha=dist_cfg["alpha"],
        feature_distillation=dist_cfg.get("feature_distillation", False),
        beta=dist_cfg.get("beta", 0.1),
        device=device,
    )

    # 蒸馏训练
    optimizer = optim.Adam(student_model.parameters(), lr=dist_cfg["distill_lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=dist_cfg["distill_epochs"]
    )

    history = distiller.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=dist_cfg["distill_epochs"],
        scheduler=scheduler,
        val_loader=test_loader,
    )

    # 获取蒸馏后学生模型
    final_model = distiller.get_student_model()
    final_accuracy = validate(final_model, test_loader, device)

    logger.info("蒸馏完成: 学生精度=%.2f%%", final_accuracy * 100)

    return {
        "model": final_model,
        "accuracy": final_accuracy,
        "teacher_accuracy": teacher_acc,
        "history": history,
    }


# ============================================================
# 导出 + 基准测试阶段
# ============================================================


def run_export_and_benchmark(
    config: dict[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """
    执行模型导出和基准测试。

    参数：
        config: 全局配置
        model: 最终压缩后的模型

    返回：
        包含导出结果和基准测试结果的字典
    """
    logger.info("=" * 60)
    logger.info("阶段 5: 模型导出")
    logger.info("=" * 60)

    export_cfg = config["export"]
    model_cfg = config["model"]

    # ONNX 导出
    export_result = export_model(
        model,
        {
            **export_cfg,
            "input_shape": [
                model_cfg["input_channels"],
                model_cfg["input_size"],
                model_cfg["input_size"],
            ],
        },
    )

    # 基准测试
    logger.info("=" * 60)
    logger.info("阶段 6: 性能基准测试")
    logger.info("=" * 60)

    bench_cfg = config["benchmark"]
    benchmarker = Benchmarker(
        model,
        input_shape=(
            model_cfg["input_channels"],
            model_cfg["input_size"],
            model_cfg["input_size"],
        ),
    )

    bench_result = benchmarker.run_full_benchmark(
        warmup_runs=bench_cfg["warmup_runs"],
        measure_runs=bench_cfg["measure_runs"],
        batch_sizes=bench_cfg["batch_sizes"],
        measure_memory=bench_cfg["measure_memory"],
        calculate_flops=bench_cfg["calculate_flops"],
    )

    return {
        "export": export_result,
        "benchmark": bench_result,
    }


# ============================================================
# 完整流水线
# ============================================================


def run_full_pipeline(config: dict[str, Any]) -> None:
    """
    执行完整的模型压缩流水线：
    train → prune → quantize → distill → export → benchmark → report

    参数：
        config: 全局配置字典
    """
    logger.info("=" * 70)
    logger.info("边缘 AI 模型压缩与部署 - 完整流水线")
    logger.info("=" * 70)

    start_time = time.time()
    device = torch.device(config["general"]["device"])
    model_cfg = config["model"]

    # 加载数据
    train_loader, test_loader = get_data_loaders(config)

    # 用于收集各阶段结果的字典
    stage_results: dict[str, dict[str, Any]] = {}
    current_model: nn.Module | None = None
    current_accuracy: float = 0.0

    # === 阶段 1: 训练基线模型 ===
    if "train" in config["pipeline"]["stages"]:
        train_result = run_training(config, train_loader, test_loader)
        current_model = train_result["model"]
        current_accuracy = train_result["accuracy"]

        stage_results["baseline"] = {
            "model": model_cfg["architecture"],
            "accuracy": current_accuracy,
            "benchmark": Benchmarker(current_model).run_full_benchmark(
                warmup_runs=5,
                measure_runs=30,
                calculate_flops=True,
                measure_memory=False,
            ),
        }
        logger.info("基线阶段完成 ✓")

    else:
        # 如果跳过训练，直接创建模型（用于调试）
        current_model = create_model(
            model_cfg["architecture"], num_classes=model_cfg["num_classes"]
        )
        current_model.to(device)
        current_accuracy = validate(current_model, test_loader, device)
        logger.info("跳过训练，直接评估: accuracy=%.2f%%", current_accuracy * 100)

    # === 阶段 2: 剪枝 ===
    if "prune" in config["pipeline"]["stages"] and current_model is not None:
        prune_result = run_pruning(config, current_model, train_loader, test_loader)
        current_model = prune_result["model"]
        current_accuracy = prune_result["accuracy"]

        stage_results["pruned"] = {
            "model": f"{model_cfg['architecture']}-pruned",
            "accuracy": current_accuracy,
            "benchmark": Benchmarker(current_model).run_full_benchmark(
                warmup_runs=5,
                measure_runs=30,
                calculate_flops=True,
                measure_memory=False,
            ),
            "extra": {"sparsity": prune_result["sparsity"]},
        }
        logger.info("剪枝阶段完成 ✓")

    # === 阶段 3: 量化 ===
    if "quantize" in config["pipeline"]["stages"] and current_model is not None:
        quantize_result = run_quantization(config, current_model, test_loader)
        current_model = quantize_result["model"]
        current_accuracy = quantize_result["accuracy"]

        stage_results["quantized"] = {
            "model": f"{model_cfg['architecture']}-quantized-{quantize_result['bits']}bit",
            "accuracy": current_accuracy,
            "benchmark": Benchmarker(current_model).run_full_benchmark(
                warmup_runs=5,
                measure_runs=30,
                calculate_flops=True,
                measure_memory=False,
            ),
            "extra": quantize_result,
        }
        logger.info("量化阶段完成 ✓")

    # === 阶段 4: 蒸馏 ===
    if "distill" in config["pipeline"]["stages"] and current_model is not None:
        distill_result = run_distillation(
            config, current_model, train_loader, test_loader
        )
        current_model = distill_result["model"]
        current_accuracy = distill_result["accuracy"]

        stage_results["distilled"] = {
            "model": f"{model_cfg['architecture']}-distilled",
            "accuracy": current_accuracy,
            "benchmark": Benchmarker(current_model).run_full_benchmark(
                warmup_runs=5,
                measure_runs=30,
                calculate_flops=True,
                measure_memory=False,
            ),
        }
        logger.info("蒸馏阶段完成 ✓")

    # === 阶段 5+6: 导出 + 基准测试 ===
    if current_model is not None:
        eb_result = run_export_and_benchmark(config, current_model)

        stage_results["final"] = {
            "model": f"{model_cfg['architecture']}-final",
            "accuracy": current_accuracy,
            "benchmark": eb_result["benchmark"],
            "extra": {"export": eb_result["export"]},
        }
        logger.info("导出和基准测试阶段完成 ✓")

    # === 生成报告 ===
    if config["pipeline"].get("generate_report", True):
        logger.info("=" * 60)
        logger.info("生成对比报告")
        logger.info("=" * 60)

        report_gen = ReportGenerator(config["general"]["report_dir"])

        for stage_name, data in stage_results.items():
            report_gen.add_stage_result(
                stage_name=stage_name,
                model_name=data["model"],
                accuracy=data["accuracy"],
                benchmark=data.get("benchmark"),
                extra_info=data.get("extra"),
            )

        report_gen.generate_comparison_report(
            title="模型压缩流水线对比报告 - MIT 6.5940",
        )
        report_gen.save_raw_results()
        logger.info("报告生成完成 ✓")

    # 总耗时
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(
        "完整流水线执行完毕！总耗时: %.2f 秒 (%.2f 分钟)", elapsed, elapsed / 60
    )
    logger.info(
        "最终模型精度: %.2f%%", current_accuracy * 100 if current_accuracy else 0
    )
    logger.info("=" * 70)


# ============================================================
# 单模式入口
# ============================================================


def run_single_mode(mode: str, config: dict[str, Any]) -> None:
    """
    执行单个模式（非完整流水线）。

    参数：
        mode: 运行模式
        config: 全局配置
    """
    device = torch.device(config["general"]["device"])
    model_cfg = config["model"]

    # 如果有训练/蒸馏就加载数据
    needs_data = mode in ("train", "prune", "distill")
    train_loader = None
    test_loader = None

    if needs_data:
        train_loader, test_loader = get_data_loaders(config)

    if mode == "train":
        result = run_training(config, train_loader, test_loader)
        logger.info("模型精度: %.2f%%", result["accuracy"] * 100)

    elif mode == "prune":
        # 先创建并加载/训练基线模型
        model = create_model(
            model_cfg["architecture"], num_classes=model_cfg["num_classes"]
        )
        model.to(device)
        # 尝试加载已有检查点
        ckpt_path = os.path.join(
            config["general"]["checkpoint_dir"], "baseline_model.pth"
        )
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info("已加载基线模型: %s", ckpt_path)
        else:
            logger.warning("基线模型检查点不存在，先快速训练一个...")
            _ = run_training(config, train_loader, test_loader)
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))

        result = run_pruning(config, model, train_loader, test_loader)
        logger.info("剪枝后精度: %.2f%%", result["accuracy"] * 100)

    elif mode == "quantize":
        model = create_model(
            model_cfg["architecture"], num_classes=model_cfg["num_classes"]
        )
        model.to(device)
        ckpt_path = os.path.join(
            config["general"]["checkpoint_dir"], "baseline_model.pth"
        )
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            if test_loader is None:
                _, test_loader = get_data_loaders(config)
            logger.warning("基线模型检查点不存在，使用随机初始化的模型")

        # 用 test_loader 作为校准数据（在量化函数内部会用到）
        if test_loader is None:
            _, test_loader = get_data_loaders(config)

        result = run_quantization(config, model, test_loader)
        logger.info("量化后精度: %.2f%%", result["accuracy"] * 100)

    elif mode == "distill":
        # 创建学生模型
        student_model = create_model(
            model_cfg["architecture"], num_classes=model_cfg["num_classes"]
        )
        student_model.to(device)
        result = run_distillation(config, student_model, train_loader, test_loader)
        logger.info("蒸馏后精度: %.2f%%", result["accuracy"] * 100)

    elif mode == "export":
        model = create_model(
            model_cfg["architecture"], num_classes=model_cfg["num_classes"]
        )
        model.to(device)
        ckpt_path = os.path.join(
            config["general"]["checkpoint_dir"], "baseline_model.pth"
        )
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        result = run_export_and_benchmark(config, model)
        logger.info(
            "导出完成: %s", result["export"].get("onnx_export", {}).get("path", "N/A")
        )

    elif mode == "benchmark":
        model = create_model(
            model_cfg["architecture"], num_classes=model_cfg["num_classes"]
        )
        model.to(device)
        ckpt_path = os.path.join(
            config["general"]["checkpoint_dir"], "baseline_model.pth"
        )
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        result = run_export_and_benchmark(config, model)
        logger.info("基准测试结果: %s", result["benchmark"].get("summary", {}))

    else:
        logger.error("未知模式: %s", mode)
        print(
            f"错误: 未知的模式 '{mode}'。支持的模式: train, prune, quantize, distill, export, benchmark, full_pipeline"
        )


# ============================================================
# 命令行参数
# ============================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="边缘 AI 模型压缩与部署全链路流水线 - MIT 6.5940 最终项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode full_pipeline
  python main.py --mode train --model resnet18 --epochs 100
  python main.py --mode prune --sparsity 0.7
  python main.py --mode quantize --bits 8 --scheme per_channel
  python main.py --mode benchmark --model mobilenetv2
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train",
            "prune",
            "quantize",
            "distill",
            "export",
            "benchmark",
            "full_pipeline",
        ],
        default="full_pipeline",
        help="运行模式 (默认: full_pipeline)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="配置文件路径 (默认: configs/config.yaml)",
    )

    # 可选命令行覆盖参数
    parser.add_argument(
        "--model", type=str, default=None, help="模型架构 (覆盖配置文件)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="训练轮数 (覆盖配置文件)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="批处理大小 (覆盖配置文件)"
    )
    parser.add_argument("--lr", type=float, default=None, help="学习率 (覆盖配置文件)")
    parser.add_argument(
        "--sparsity", type=float, default=None, help="剪枝稀疏度 (覆盖配置文件)"
    )
    parser.add_argument(
        "--bits", type=int, default=None, help="量化位宽 (覆盖配置文件)"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        help="量化方案: per_tensor/per_channel (覆盖配置文件)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="启用 CPU 快速验收模式：tinycnn、少量 synthetic data、短训练和短 benchmark",
    )

    return parser


def apply_quick_mode(config: dict[str, Any]) -> None:
    """Apply a deterministic CPU smoke-test profile for CI and fast local checks."""
    config["model"]["architecture"] = "tinycnn"
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 16
    config["training"]["train_samples"] = 256
    config["training"]["test_samples"] = 128
    config["pruning"]["finetune_epochs"] = 1
    config["quantization"]["calibration_samples"] = 128
    config["distillation"]["teacher_architecture"] = "tinycnn"
    config["distillation"]["teacher_epochs"] = 1
    config["distillation"]["distill_epochs"] = 1
    config["benchmark"]["warmup_runs"] = 1
    config["benchmark"]["measure_runs"] = 3
    config["benchmark"]["batch_sizes"] = [1, 4]
    logger.info("已启用 quick 模式：tinycnn + short synthetic benchmark")


def merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """将命令行参数合并到配置字典中（覆盖原有值）。"""
    if getattr(args, "quick", False):
        apply_quick_mode(config)
    if args.model is not None:
        config["model"]["architecture"] = args.model
        logger.info("模型架构已覆盖: %s", args.model)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        logger.info("训练轮数已覆盖: %d", args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
        logger.info("批处理大小已覆盖: %d", args.batch_size)
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
        logger.info("学习率已覆盖: %f", args.lr)
    if args.sparsity is not None:
        config["pruning"]["sparsity"] = args.sparsity
        logger.info("剪枝稀疏度已覆盖: %.2f", args.sparsity)
    if args.bits is not None:
        config["quantization"]["bits"] = args.bits
        logger.info("量化位宽已覆盖: %d", args.bits)
    if args.scheme is not None:
        config["quantization"]["scheme"] = args.scheme
        logger.info("量化方案已覆盖: %s", args.scheme)


# ============================================================
# 主函数
# ============================================================


def main() -> None:
    """主函数入口 —— 解析参数 + 执行流水线。"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # 打印标题
    print("=" * 70)
    print("  边缘 AI 模型压缩与部署全链路流水线")
    print("  MIT 6.5940 高效深度学习课程 - 最终项目")
    print("=" * 70)
    print(f"  运行模式: {args.mode}")
    print(f"  配置文件: {args.config}")
    print("=" * 70)
    print()

    # 加载配置
    try:
        config = load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error("配置加载失败: %s", e)
        sys.exit(1)

    # 合并命令行覆盖
    merge_cli_overrides(config, args)

    # 设置随机种子
    torch.manual_seed(config["general"]["seed"])
    np.random.seed(config["general"]["seed"])

    # 创建输出目录
    for dir_key in ["output_dir", "checkpoint_dir", "log_dir", "report_dir"]:
        dir_path = config["general"].get(dir_key, f"./{dir_key}")
        os.makedirs(dir_path, exist_ok=True)

    # 根据模式执行
    try:
        if args.mode == "full_pipeline":
            run_full_pipeline(config)
        else:
            run_single_mode(args.mode, config)
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        sys.exit(130)
    except Exception as e:
        logger.error("执行失败: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("程序正常退出")


if __name__ == "__main__":
    main()
