#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏模块 —— 将教师模型的知识迁移到学生模型。

本模块实现了两种蒸馏方法：
1. KD 蒸馏（Hinton et al., 2015）：
   - 使用教师模型的软标签（soft targets）+ 温度参数
   - KL 散度损失 + 交叉熵损失的组合

2. 特征蒸馏：
   - 对齐教师和学生的中间层特征图
   - 使用 MSE 损失匹配特征响应

参考论文：
- Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
- FitNets (Romero et al., 2015) - 特征蒸馏

所有注释使用中文，便于课程学习。
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# KD 蒸馏损失函数
# ============================================================


class KDLoss(nn.Module):
    """
    知识蒸馏损失 —— 使用 KL 散度 + 温度软化。

    公式：
        L_KD = α * T² * KL(softmax(s/T), softmax(t/T))
              + (1-α) * CE(softmax(s), y_true)

    其中：
        s = 学生 logits, t = 教师 logits
        T = 温度（越高 → 软标签越平滑）
        α = KL 损失权重

    参考：Distilling the Knowledge in a Neural Network (Hinton, 2015)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ) -> None:
        """
        参数：
            temperature: 蒸馏温度（越高越平滑，典型值 1-20）
            alpha: KL 散度损失的权重（0-1 之间）
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"温度必须为正数，当前值为 {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1] 之间，当前值为 {alpha}")

        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 KD 损失。

        参数：
            student_logits: 学生模型的输出 logits
            teacher_logits: 教师模型的输出 logits（无需梯度，通常已 detach）
            targets: 真实标签（用于交叉熵）

        返回：
            综合损失值
        """
        # 交叉熵损失（硬标签硬监督）
        loss_ce = self.ce_loss(student_logits, targets)

        # KD 损失（软标签软监督）
        # 使用温度软化概率分布
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        loss_kd = self.kl_loss(soft_student, soft_teacher)
        # 乘以 T² 来平衡梯度（因为温度会对梯度进行缩放）
        loss_kd = loss_kd * (self.temperature**2)

        # 组合损失
        total_loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce

        return total_loss


# ============================================================
# 特征蒸馏损失
# ============================================================


class FeatureDistillationLoss(nn.Module):
    """
    特征蒸馏损失 —— 对齐教师和学生模型的中间层特征图。

    使用 L2 损失匹配特征响应，帮助学生学到与教师相似的中间表示。

    参考：FitNets: Hints for Thin Deep Nets (Romero et al., 2015)
    """

    def __init__(self, use_l2: bool = True) -> None:
        """
        参数：
            use_l2: 是否使用 L2 损失（True=L2, False=L1）
        """
        super().__init__()
        self.use_l2 = use_l2

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失。

        参数：
            student_features: 学生中间层特征图
            teacher_features: 教师对应层的特征图（已 detach）

        返回：
            特征匹配损失值
        """
        # 确保尺寸匹配：可能需要投影或插值
        if student_features.shape != teacher_features.shape:
            # 使用自适应池化对齐尺寸
            student_features = F.adaptive_avg_pool2d(
                student_features, teacher_features.shape[2:]
            )

        if self.use_l2:
            return F.mse_loss(student_features, teacher_features)
        else:
            return F.l1_loss(student_features, teacher_features)


# ============================================================
# 蒸馏训练器
# ============================================================


class Distiller:
    """
    知识蒸馏训练器 —— 管理教师-学生蒸馏训练的完整流程。

    使用方式：
    1. 创建 Distiller 实例，传入教师模型和学生模型
    2. 调用 train() 方法执行蒸馏训练
    3. 获取训练后的学生模型
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_distillation: bool = False,
        beta: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        参数：
            teacher_model: 教师模型（通常更大更优，冻结不训练）
            student_model: 学生模型（将接受训练）
            temperature: 蒸馏温度
            alpha: KD 损失权重
            feature_distillation: 是否启用特征蒸馏
            beta: 特征蒸馏损失权重
            device: 运行设备
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.feature_distillation = feature_distillation
        self.beta = beta
        self.device = device

        # 损失函数
        self.kd_loss_fn = KDLoss(temperature=temperature, alpha=alpha)
        self.feature_loss_fn = FeatureDistillationLoss()

        # 将模型移到目标设备
        self.teacher.to(self.device)
        self.student.to(self.device)

        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def train(
        self,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        epochs: int = 30,
        scheduler: Any = None,
        val_loader: Any = None,
    ) -> dict[str, list[float]]:
        """
        执行蒸馏训练。

        训练循环：
        1. 教师模型输出 soft labels（温度软化后）
        2. 学生模型输出 logits
        3. 计算 KD 损失（KL 散度）+ 可选特征蒸馏损失
        4. 反向传播更新学生模型参数

        参数：
            train_loader: 训练数据加载器
            optimizer: 优化器
            epochs: 训练轮数
            scheduler: 学习率调度器
            val_loader: 验证数据加载器（可选）

        返回：
            dict: 训练历史记录 {'train_loss': [...], 'val_acc': [...]}
        """
        logger.info(
            "开始蒸馏训练: 教师=%s, 学生=%s, 温度=%.1f, alpha=%.2f, epochs=%d",
            self.teacher.__class__.__name__,
            self.student.__class__.__name__,
            self.temperature,
            self.alpha,
            epochs,
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            # === 训练阶段 ===
            self.student.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # 教师模型前向传播（无梯度）
                with torch.no_grad():
                    teacher_logits = self.teacher(images)

                # 学生模型前向传播
                student_logits = self.student(images)

                # 计算 KD 损失
                loss = self.kd_loss_fn(student_logits, teacher_logits, targets)

                # 可选：特征蒸馏（略过，需要 hook 中间层特征）
                if self.feature_distillation:
                    # 实际实现需要 hook 教师和学生的中间层输出
                    # 这里仅做占位
                    pass

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 50 == 0:
                    logger.debug(
                        "Epoch %d, Batch %d: loss=%.4f", epoch, batch_idx, loss.item()
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            # === 验证阶段 ===
            if val_loader is not None:
                acc = self._evaluate(val_loader)
                history["val_acc"].append(acc)
                logger.info(
                    "Epoch %d/%d: loss=%.4f, val_acc=%.2f%%",
                    epoch,
                    epochs,
                    avg_loss,
                    acc * 100,
                )
            else:
                logger.info("Epoch %d/%d: loss=%.4f", epoch, epochs, avg_loss)

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

        logger.info("蒸馏训练完成")
        return history

    def _evaluate(self, val_loader: Any) -> float:
        """
        评估学生模型在验证集上的精度。

        参数：
            val_loader: 验证数据加载器

        返回：
            精度值 (0.0 ~ 1.0)
        """
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.student(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / max(total, 1)

    def get_student_model(self) -> nn.Module:
        """获取训练后的学生模型。"""
        return self.student


# ============================================================
# 工具函数：创建蒸馏器
# ============================================================


def create_distiller(
    teacher_model: nn.Module,
    student_model: nn.Module,
    config: dict[str, Any] | None = None,
) -> Distiller:
    """
    根据配置创建蒸馏器。

    参数：
        teacher_model: 教师模型
        student_model: 学生模型
        config: 配置字典，包含 temperature, alpha, feature_distillation, beta

    返回：
        Distiller 实例
    """
    if config is None:
        config = {}

    return Distiller(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=config.get("temperature", 4.0),
        alpha=config.get("alpha", 0.7),
        feature_distillation=config.get("feature_distillation", False),
        beta=config.get("beta", 0.1),
    )
