#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验运行脚本 —— 批量执行模型压缩实验。

本脚本支持：
1. 网格搜索（Grid Search）：在超参数空间中自动搜索最优配置
2. 批量实验：对多个模型架构运行完整压缩流水线
3. 结果收集：自动保存每组实验的精度、性能指标到 JSON

用法：
    # 网格搜索剪枝超参数
    python scripts/run_experiments.py --grid_search

    # 运行所有预设实验
    python scripts/run_experiments.py --all

    # 仅运行剪枝实验
    python scripts/run_experiments.py --exp prune

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_factory import create_model, get_model_info
from compression.pruner import create_pruner
from quantization.quantizer import create_quantizer
from benchmark.benchmarker import Benchmarker

logger = logging.getLogger(__name__)


# ============================================================
# CIFAR-10 数据加载器
# ============================================================


def get_cifar10_loaders(
    batch_size: int = 64,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """
    获取 CIFAR-10 训练和测试数据加载器。

    参数：
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        data_dir: 数据集缓存目录

    返回：
        (train_loader, test_loader)
    """
    # 训练集的增强变换
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    # 测试集的简单变换
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "CIFAR-10 数据加载完成: 训练=%d, 测试=%d", len(train_dataset), len(test_dataset)
    )

    return train_loader, test_loader


# ============================================================
# 训练和评估辅助函数
# ============================================================


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """评估模型精度，返回 Top-1 准确率（0.0 ~ 1.0）。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """
    快速训练模型并返回结果。

    参数：
        model: PyTorch 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 运行设备

    返回：
        dict: 包含 best_accuracy 和 final_accuracy 的训练结果
    """
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc

        logger.info(
            "Epoch %d/%d: loss=%.4f, acc=%.2f%%", epoch, epochs, train_loss, acc * 100
        )

    return {"best_accuracy": best_acc, "final_accuracy": acc}


# ============================================================
# 实验定义
# ============================================================


def run_pruning_experiments(
    output_dir: str = "./experiments",
) -> list[dict[str, Any]]:
    """
    剪枝超参数网格搜索实验。

    搜索空间：
    - 模型架构: mobilenetv2, resnet18
    - 剪枝方法: magnitude, channel
    - 稀疏度: [0.3, 0.5, 0.7, 0.9]

    返回：
        实验结果列表
    """
    logger.info("=" * 60)
    logger.info("剪枝实验：网格搜索")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

    # 搜索空间
    architectures = ["mobilenetv2", "resnet18"]
    methods = ["magnitude", "channel"]
    sparsities = [0.3, 0.5, 0.7, 0.9]

    results: list[dict[str, Any]] = []

    for arch, method, sp in product(architectures, methods, sparsities):
        logger.info("--- 实验: arch=%s, method=%s, sparsity=%.2f ---", arch, method, sp)

        try:
            # 创建并训练基线模型
            model = create_model(arch)
            train_result = train_model(
                model, train_loader, test_loader, epochs=5, lr=0.01, device=device
            )

            # 剪枝
            pruner = create_pruner(model, method=method)
            pruned = pruner.prune(sparsity=sp)

            # 评估剪枝后的精度
            pruned_acc = evaluate(pruned, test_loader, device)

            # 基准测试
            bench = Benchmarker(pruned, input_shape=(3, 32, 32))
            bench_result = bench.run_full_benchmark(
                warmup_runs=5,
                measure_runs=30,
                calculate_flops=True,
                measure_memory=False,
            )

            entry = {
                "architecture": arch,
                "method": method,
                "target_sparsity": sp,
                "applied_sparsity": pruner.sparsity_applied,
                "baseline_accuracy": train_result["best_accuracy"],
                "pruned_accuracy": pruned_acc,
                "accuracy_drop": train_result["best_accuracy"] - pruned_acc,
                "benchmark": bench_result.get("summary", {}),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(entry)

            logger.info(
                "结果: 基线精度=%.2f%%, 剪枝后精度=%.2f%%, 下降=%.2f%%",
                train_result["best_accuracy"] * 100,
                pruned_acc * 100,
                (train_result["best_accuracy"] - pruned_acc) * 100,
            )

        except Exception as e:
            logger.error("实验失败: %s", e)
            results.append(
                {
                    "architecture": arch,
                    "method": method,
                    "target_sparsity": sp,
                    "error": str(e),
                }
            )

    # 保存结果
    result_path = os.path.join(output_dir, "pruning_grid_search.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("剪枝实验结果已保存: %s", result_path)
    return results


def run_quantization_experiments(
    output_dir: str = "./experiments",
) -> list[dict[str, Any]]:
    """
    量化超参数网格搜索实验。

    搜索空间：
    - 模型架构: mobilenetv2, resnet18
    - 量化位宽: [8, 4, 2]
    - 量化方案: per_tensor, per_channel

    返回：
        实验结果列表
    """
    logger.info("=" * 60)
    logger.info("量化实验：网格搜索")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

    architectures = ["mobilenetv2", "resnet18"]
    bit_widths = [8, 4, 2]
    schemes = ["per_tensor", "per_channel"]

    results: list[dict[str, Any]] = []

    for arch, bits, scheme in product(architectures, bit_widths, schemes):
        logger.info("--- 实验: arch=%s, bits=%d, scheme=%s ---", arch, bits, scheme)

        try:
            # 创建并训练基线模型
            model = create_model(arch)
            train_result = train_model(
                model, train_loader, test_loader, epochs=5, lr=0.01, device=device
            )

            # 量化
            quantizer = create_quantizer(bits=bits, scheme=scheme, method="ptq")
            quantized = quantizer.quantize(model)

            # 评估
            quant_acc = evaluate(quantized, test_loader, device)

            # 基准测试
            bench = Benchmarker(quantized, input_shape=(3, 32, 32))
            bench_result = bench.run_full_benchmark(
                warmup_runs=5,
                measure_runs=30,
                calculate_flops=True,
                measure_memory=False,
            )

            entry = {
                "architecture": arch,
                "bits": bits,
                "scheme": scheme,
                "baseline_accuracy": train_result["best_accuracy"],
                "quantized_accuracy": quant_acc,
                "accuracy_drop": train_result["best_accuracy"] - quant_acc,
                "benchmark": bench_result.get("summary", {}),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(entry)

            logger.info(
                "结果: 基线精度=%.2f%%, 量化后精度=%.2f%%, 下降=%.2f%%",
                train_result["best_accuracy"] * 100,
                quant_acc * 100,
                (train_result["best_accuracy"] - quant_acc) * 100,
            )

        except Exception as e:
            logger.error("实验失败: %s", e)
            results.append(
                {
                    "architecture": arch,
                    "bits": bits,
                    "scheme": scheme,
                    "error": str(e),
                }
            )

    # 保存结果
    result_path = os.path.join(output_dir, "quantization_grid_search.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("量化实验结果已保存: %s", result_path)
    return results


def run_all_experiments(output_dir: str = "./experiments") -> dict[str, Any]:
    """
    运行所有预设实验。

    参数：
        output_dir: 结果输出目录

    返回：
        包含所有实验结果的字典
    """
    all_results: dict[str, Any] = {}

    logger.info("=" * 70)
    logger.info("批量实验开始")
    logger.info("=" * 70)

    # 剪枝实验
    all_results["pruning"] = run_pruning_experiments(output_dir)

    # 量化实验
    all_results["quantization"] = run_quantization_experiments(output_dir)

    # 汇总
    summary_path = os.path.join(output_dir, "all_experiments.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("所有实验完成，结果已保存: %s", summary_path)

    return all_results


# ============================================================
# 主入口
# ============================================================


def main() -> None:
    """实验脚本主入口。"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="模型压缩实验运行器 —— 网格搜索和批量实验"
    )
    parser.add_argument(
        "--grid_search", action="store_true", help="运行超参数网格搜索实验"
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["prune", "quantize", "all"],
        default="all",
        help="选择实验类型 (默认: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="实验结果输出目录 (默认: ./experiments)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.grid_search:
        # 运行网格搜索
        if args.exp in ("prune", "all"):
            run_pruning_experiments(args.output_dir)
        if args.exp in ("quantize", "all"):
            run_quantization_experiments(args.output_dir)
    elif args.exp == "all":
        run_all_experiments(args.output_dir)
    else:
        if args.exp == "prune":
            run_pruning_experiments(args.output_dir)
        elif args.exp == "quantize":
            run_quantization_experiments(args.output_dir)


if __name__ == "__main__":
    main()
