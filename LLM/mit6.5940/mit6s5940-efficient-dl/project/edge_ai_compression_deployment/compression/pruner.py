#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型剪枝模块 —— 支持多种剪枝策略。

本模块实现了三种剪枝方法：
1. MagnitudePruner（幅度剪枝）：移除绝对值最小的权重（非结构化稀疏）
2. ChannelPruner（通道剪枝）：移除 L1/L2 范数最小的通道（结构化稀疏）
3. GradualPruner（渐进剪枝）：迭代小步剪枝 + 每轮微调恢复精度

参考论文：
- Deep Compression (Han et al., ICLR 2016) - 幅度剪枝
- Learning both Weights and Connections (Han et al., NeurIPS 2015)

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# 剪枝基础类（抽象基类）
# ============================================================


class BasePruner(ABC):
    """所有剪枝器的抽象基类。"""

    def __init__(self, model: nn.Module) -> None:
        """
        参数：
            model: 待剪枝的 PyTorch 模型
        """
        self.original_model = model
        self.pruned_model: nn.Module | None = None
        self._sparsity_applied: float = 0.0
        self._masks: dict[str, torch.Tensor] = {}

    @abstractmethod
    def prune(self, sparsity: float) -> nn.Module:
        """
        执行剪枝。

        参数：
            sparsity: 目标稀疏度（0.0 ~ 1.0，表示要置零的比例）

        返回：
            剪枝后的模型
        """
        ...

    @property
    def sparsity_applied(self) -> float:
        """获取实际应用的稀疏度。"""
        return self._sparsity_applied

    @staticmethod
    def _get_prunable_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
        """
        获取模型中所有可剪枝的模块（Conv2d 和 Linear 层）。

        参数：
            model: 目标模型

        返回：
            元组列表：[(模块名称, 模块实例), ...]
        """
        prunable: list[tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable.append((name, module))
        return prunable

    @staticmethod
    def calculate_sparsity(model: nn.Module) -> dict[str, float]:
        """
        计算模型的当前稀疏度。

        参数：
            model: 目标模型

        返回：
            dict: 包含总稀疏度及各层稀疏度的字典
        """
        total_zeros = 0
        total_params = 0
        per_layer: dict[str, float] = {}

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                zeros = (param == 0).sum().item()
                total = param.numel()
                total_zeros += zeros
                total_params += total
                per_layer[name] = zeros / total

        return {
            "total_sparsity": total_zeros / total_params if total_params > 0 else 0,
            "per_layer": per_layer,
        }


# ============================================================
# 幅度剪枝器（Deep Compression 论文的核心方法）
# ============================================================


class MagnitudePruner(BasePruner):
    """
    幅度剪枝器 —— 按权重绝对值大小进行非结构化剪枝。

    算法原理：
    1. 对每个可剪枝层，收集所有权重的绝对值
    2. 计算该层的剪枝阈值（第 sparsity 百分位数）
    3. 将低于阈值的权重重置为零
    4. 生成二值 mask 用于后续微调

    参考：Deep Compression (Han et al., ICLR 2016)
    """

    def prune(self, sparsity: float) -> nn.Module:
        """
        执行幅度剪枝。

        参数：
            sparsity: 目标稀疏度（例如 0.5 表示移除 50% 的权重）

        返回：
            剪枝后的模型（深拷贝，不修改原始模型）
        """
        logger.info("开始幅度剪枝，目标稀疏度: %.2f%%", sparsity * 100)

        self.pruned_model = copy.deepcopy(self.original_model)
        prunable_modules = self._get_prunable_modules(self.pruned_model)

        total_pruned = 0
        total_params = 0

        for name, module in prunable_modules:
            if not hasattr(module, "weight") or module.weight is None:
                continue

            # 获取权重（2D/4D）
            weight = module.weight.data
            num_params = weight.numel()
            total_params += num_params

            # 计算该层的剪枝阈值
            # 全局阈值法：按绝对值排序，取第 sparsity 分位数
            flat_weight = weight.abs().flatten()
            k = int(sparsity * num_params)
            if k == 0:
                continue

            threshold = torch.kthvalue(flat_weight, k).values.item()

            # 生成剪枝 mask：低于阈值的权重标记为 0
            mask = (weight.abs() > threshold).float()
            self._masks[name] = mask.clone()

            # 应用 mask（就地修改权重）
            module.weight.data.mul_(mask)

            pruned_count = num_params - mask.sum().item()
            total_pruned += pruned_count

            logger.debug(
                "层 %s: 剪枝 %d/%d 参数 (阈值: %.6f)",
                name,
                pruned_count,
                num_params,
                threshold,
            )

        self._sparsity_applied = total_pruned / total_params if total_params > 0 else 0
        logger.info(
            "幅度剪枝完成: 总参数 %d, 剪枝 %d, 实际稀疏度 %.2f%%",
            total_params,
            total_pruned,
            self._sparsity_applied * 100,
        )

        return self.pruned_model


# ============================================================
# 通道剪枝器（结构化剪枝，利于硬件加速）
# ============================================================


class ChannelPruner(BasePruner):
    """
    通道剪枝器 —— 移除整个输出通道（结构化剪枝）。

    算法原理：
    1. 对每个卷积层，计算每个输出通道的重要性分数
    2. 按重要性排序，移除最不重要的通道
    3. 同时需要修改后续层的输入通道数（结构重排）

    重要性度量：
    - l1_norm: 按 L1 范数对卷积核求重要性
    - frobenius: 按 Frobenius 范数（权重矩阵范数的推广）
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: str = "frobenius",
    ) -> None:
        """
        参数：
            model: 待剪枝的模型
            criterion: 重要性度量标准 ("l1_norm" 或 "frobenius")
        """
        super().__init__(model)
        self.criterion = criterion
        if criterion not in ("l1_norm", "frobenius"):
            raise ValueError(
                f"无效的剪枝标准: '{criterion}'。仅支持: l1_norm, frobenius"
            )

    def prune(self, sparsity: float) -> nn.Module:
        """
        执行通道剪枝。

        参数：
            sparsity: 目标稀疏度（要保留多少比例的输出通道）

        返回：
            剪枝后的模型
        """
        logger.info(
            "开始通道剪枝，目标稀疏度: %.2f%% (标准: %s)",
            sparsity * 100,
            self.criterion,
        )

        # 通道剪枝需要修改网络结构，无法在原模型上做 mask
        # 实际实现中需要使用 torch.nn.utils.prune 或重建模型
        # 这里使用 mask 方案模拟通道剪枝的效果（将整个通道的权重置零）

        self.pruned_model = copy.deepcopy(self.original_model)
        prunable_modules = self._get_prunable_modules(self.pruned_model)

        total_pruned = 0
        total_channels = 0

        for name, module in prunable_modules:
            if not hasattr(module, "weight") or module.weight is None:
                continue

            weight = module.weight.data
            if weight.dim() != 4:
                # 跳过 Linear 层（通道剪枝主要用于 Conv）
                continue

            out_channels = weight.shape[0]
            total_channels += out_channels

            # 计算每个输出通道的重要性分数
            if self.criterion == "frobenius":
                # Frobenius 范数：对每个输出通道计算整个 kernel 的 L2 范数
                scores = weight.view(out_channels, -1).norm(p=2, dim=1)
            else:
                # L1 范数
                scores = weight.view(out_channels, -1).abs().sum(dim=1)

            # 选择保留的通道
            num_keep = max(1, int(out_channels * (1 - sparsity)))
            _, keep_indices = torch.topk(scores, num_keep)

            # 创建通道 mask
            channel_mask = torch.zeros(out_channels, device=weight.device)
            channel_mask[keep_indices] = 1

            # 将 mask 广播到整个权重 tensor
            mask_shape = [out_channels] + [1] * (weight.dim() - 1)
            channel_mask_reshaped = channel_mask.view(mask_shape)
            self._masks[name] = channel_mask_reshaped.clone()

            module.weight.data.mul_(channel_mask_reshaped)

            pruned_count = out_channels - num_keep
            total_pruned += pruned_count

            logger.debug(
                "层 %s: 保留 %d/%d 通道 (pruned=%d)",
                name,
                num_keep,
                out_channels,
                pruned_count,
            )

        self._sparsity_applied = (
            total_pruned / total_channels if total_channels > 0 else 0
        )
        logger.info(
            "通道剪枝完成: 总通道 %d, 剪枝 %d, 实际稀疏度 %.2f%%",
            total_channels,
            total_pruned,
            self._sparsity_applied * 100,
        )

        return self.pruned_model


# ============================================================
# 渐进剪枝器（迭代小步剪枝 + 每轮微调）
# ============================================================


class GradualPruner(BasePruner):
    """
    渐进剪枝器 —— 分多轮迭代剪枝，每轮只剪一小部分。

    算法原理：
    1. 初始稀疏度从 0 开始
    2. 每轮将稀疏度增加 step_sparsity
    3. 每轮剪枝后执行 fine-tune 恢复精度
    4. 重复直到达到目标稀疏度

    优势：避免一次性大比例剪枝导致的精度崩溃。
    参考：Learning both Weights and Connections (Han et al., NeurIPS 2015)
    """

    def __init__(
        self,
        model: nn.Module,
        finetune_fn: Callable[[nn.Module, int], nn.Module] | None = None,
    ) -> None:
        """
        参数：
            model: 待剪枝的模型
            finetune_fn: 微调函数，签名为 (model, epochs) -> model
        """
        super().__init__(model)
        self.finetune_fn = finetune_fn

    def prune(
        self,
        sparsity: float,
        iterations: int = 5,
        start_sparsity: float = 0.0,
    ) -> nn.Module:
        """
        执行渐进剪枝。

        参数：
            sparsity: 最终目标稀疏度
            iterations: 迭代轮数
            start_sparsity: 初始稀疏度

        返回：
            剪枝后的模型
        """
        logger.info(
            "开始渐进剪枝: 目标稀疏度 %.2f%%, %d 轮迭代", sparsity * 100, iterations
        )

        current_model = copy.deepcopy(self.original_model)

        # 每轮增加的稀疏度
        step = (sparsity - start_sparsity) / iterations

        for iteration in range(iterations):
            target = start_sparsity + step * (iteration + 1)
            logger.info(
                "--- 第 %d/%d 轮: 稀疏度 %.2f%% ---",
                iteration + 1,
                iterations,
                target * 100,
            )

            # 执行幅度剪枝
            sub_pruner = MagnitudePruner(current_model)
            current_model = sub_pruner.prune(target)

            # 如果提供了微调函数，执行微调
            if self.finetune_fn is not None:
                logger.info("执行微调恢复精度...")
                current_model = self.finetune_fn(current_model, 2)  # 每轮微调 2 epoch

            # 更新参考模型为当前状态
            self.original_model = copy.deepcopy(current_model)

        self.pruned_model = current_model
        sparsity_result = self.calculate_sparsity(self.pruned_model)
        self._sparsity_applied = sparsity_result["total_sparsity"]

        logger.info("渐进剪枝完成: 最终稀疏度 %.2f%%", self._sparsity_applied * 100)

        return self.pruned_model


# ============================================================
# 敏感度分析器 —— 帮助确定各层的最佳剪枝率
# ============================================================


class SensitivityAnalyzer:
    """
    敏感度分析器 —— 分析模型各层对剪枝的敏感度。

    方法：
    对每一层分别剪枝不同的比例，测量精度下降程度。
    敏感度高的层（精度下降大的层）应保留更多参数。
    """

    def __init__(
        self,
        model: nn.Module,
        eval_fn: Callable[[nn.Module], float],
    ) -> None:
        """
        参数：
            model: 待分析的模型
            eval_fn: 评估函数，输入模型返回精度值
        """
        self.model = model
        self.eval_fn = eval_fn

    def analyze(
        self,
        sparsity_levels: list[float] | None = None,
    ) -> dict[str, dict[float, float]]:
        """
        分析每一层的敏感度。

        参数：
            sparsity_levels: 测试的稀疏度级别列表（默认 [0.3, 0.5, 0.7, 0.9]）

        返回：
            dict: 格式为 {layer_name: {sparsity: accuracy_drop, ...}, ...}
        """
        if sparsity_levels is None:
            sparsity_levels = [0.3, 0.5, 0.7, 0.9]

        baseline_accuracy = self.eval_fn(self.model)
        logger.info("基线精度: %.2f%%", baseline_accuracy * 100)

        prunable = self._get_prunable_modules_static(self.model)
        sensitivity: dict[str, dict[float, float]] = {}

        for name, _ in prunable:
            layer_sensitivity: dict[float, float] = {}

            for sp in sparsity_levels:
                # 创建仅对该层剪枝的临时模型
                temp_model = copy.deepcopy(self.model)

                # 找到该层并单独剪枝
                target_module = None
                for mod_name, mod in temp_model.named_modules():
                    if mod_name == name:
                        target_module = mod
                        break

                if target_module is None or not hasattr(target_module, "weight"):
                    continue

                weight = target_module.weight.data
                flat = weight.abs().flatten()
                k = int(sp * flat.numel())
                if k == 0:
                    continue

                threshold = torch.kthvalue(flat, k).values.item()
                mask = (weight.abs() > threshold).float()
                target_module.weight.data.mul_(mask)

                # 评估精度
                accuracy = self.eval_fn(temp_model)
                drop = baseline_accuracy - accuracy
                layer_sensitivity[sp] = drop

                logger.debug(
                    "层 %s, 稀疏度 %.0f%%: 精度下降 %.4f", name, sp * 100, drop
                )

            sensitivity[name] = layer_sensitivity

        return sensitivity

    @staticmethod
    def _get_prunable_modules_static(
        model: nn.Module,
    ) -> list[tuple[str, nn.Module]]:
        """获取可剪枝模块（静态方法版本）。"""
        return BasePruner._get_prunable_modules(model)


# ============================================================
# 创建剪枝器的工厂函数
# ============================================================


def create_pruner(
    model: nn.Module,
    method: str = "magnitude",
    **kwargs: Any,
) -> BasePruner:
    """
    根据方法名称创建剪枝器实例。

    参数：
        model: 待剪枝的模型
        method: 剪枝方法（"magnitude", "channel", "gradual"）
        **kwargs: 传递给剪枝器构造函数的额外参数

    返回：
        剪枝器实例

    异常：
        ValueError: 当 method 不在支持列表中时抛出
    """
    method_lower = method.lower()

    if method_lower == "magnitude":
        return MagnitudePruner(model)
    elif method_lower == "channel":
        criterion = kwargs.get("criterion", "frobenius")
        return ChannelPruner(model, criterion=criterion)
    elif method_lower == "gradual":
        finetune_fn = kwargs.get("finetune_fn", None)
        return GradualPruner(model, finetune_fn=finetune_fn)
    else:
        raise ValueError(
            f"不支持的剪枝方法: '{method}'。支持的方法: magnitude, channel, gradual"
        )
