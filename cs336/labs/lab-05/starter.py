"""
Lab 05: Data Pipeline & Alignment — 起始代码

完成以下内容:
  - TextCleaning: 清洗原始文本
  - QualityFilter: 基于规则的质量过滤
  - DataMixer: 混合多个数据源
  - dpo_loss: Direct Preference Optimization 损失
  - grpo_loss (加分项): Group Relative Policy Optimization
"""

from __future__ import annotations

import re
import hashlib
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 任务 1: 数据管线
# ══════════════════════════════════════════════════════════════════════


class TextCleaner:
    """在训练前清洗原始网页文本。"""

    # 用于清洗的正则表达式模式
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<[^>]+>")
    EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
    # 保留 Unicode 字母、数字、基本标点符号和空白字符
    CLEAN_PATTERN = re.compile(r"[^\w\s.,!?;:'\"()\[\]{}@#$%^&*+=/\\|~`<>\-]")

    @staticmethod
    def clean(text: str) -> str:
        """清洗单个文本样本。

        步骤:
        1. 移除 HTML 标签
        2. 将 URL 替换为 <URL> 标记
        3. 将邮箱替换为 <EMAIL> 标记
        4. 规范化空白字符（多个空格/换行 -> 单个空格）
        5. 去除首尾空白字符
        """
        # TODO: 实现文本清洗
        raise NotImplementedError("TextCleaner.clean() not implemented")

    @staticmethod
    def clean_batch(texts: List[str]) -> List[str]:
        """批量清洗文本。"""
        return [TextCleaner.clean(t) for t in texts]


class QualityFilter:
    """基于规则的训练数据质量过滤。"""

    def __init__(
        self,
        min_chars: int = 100,
        max_chars: int = 100_000,
        max_word_repetition_ratio: float = 0.3,
        min_avg_word_length: float = 2.0,
        max_avg_word_length: float = 10.0,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_word_repetition_ratio = max_word_repetition_ratio
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length

    def should_keep(self, text: str) -> bool:
        """如果文本通过所有质量过滤，返回 True。

        检查项:
        1. 长度在 [min_chars, max_chars] 范围内
        2. 单词重复率 < max_word_repetition_ratio
           (最常见单词的频率 / 总单词数)
        3. 平均单词长度在 [min_avg_word_length, max_avg_word_length] 范围内
        """
        # TODO: 实现质量过滤
        raise NotImplementedError("QualityFilter.should_keep() not implemented")

    def filter(self, texts: List[str]) -> List[str]:
        """过滤文本列表，仅保留通过过滤的文本。"""
        return [t for t in texts if self.should_keep(t)]


class DataMixer:
    """按指定比例混合多个数据源。"""

    def __init__(self, ratios: Dict[str, float]):
        """使用 source_name -> ratio 映射初始化混合器。

        所有 ratio 之和应约等于 1.0。
        """
        self.ratios = ratios

    def sample_batch(self, sources: Dict[str, List[str]], batch_size: int) -> List[str]:
        """按混合比例采样一个 batch。

        Args:
            sources: 将 source_name 映射到文本列表的字典。
            batch_size: 要返回的样本数量。

        Returns:
            包含 batch_size 个文本样本的列表。
        """
        # TODO: 实现数据混合
        raise NotImplementedError("DataMixer.sample_batch() not implemented")

    @staticmethod
    def compute_minhash_signature(
        text: str, n_gram: int = 3, num_hashes: int = 128
    ) -> List[int]:
        """为文本计算 MinHash 签名。

        步骤:
        1. 从文本生成 n-grams
        2. 用 num_hashes 个不同的哈希函数对每个 n-gram 进行哈希
        3. 对每个哈希函数，取最小的哈希值
        4. 返回 min-hash 值列表（签名）
        """
        # TODO: 实现 MinHash
        raise NotImplementedError(
            "DataMixer.compute_minhash_signature() not implemented"
        )


# ══════════════════════════════════════════════════════════════════════
# 任务 2: DPO Loss
# ══════════════════════════════════════════════════════════════════════


def dpo_loss(
    policy_chosen_logps: torch.Tensor,  # log π_θ(y_w | x)
    policy_rejected_logps: torch.Tensor,  # log π_θ(y_l | x)
    reference_chosen_logps: torch.Tensor,  # log π_ref(y_w | x)
    reference_rejected_logps: torch.Tensor,  # log π_ref(y_l | x)
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 Rafailov et al. (2023) 提出的 DPO 损失。

    L_DPO = -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))

    Args:
        policy_chosen_logps: policy 模型下 chosen 响应的对数概率。
        policy_rejected_logps: policy 模型下 rejected 响应的对数概率。
        reference_chosen_logps: reference 模型下 chosen 响应的对数概率。
        reference_rejected_logps: reference 模型下 rejected 响应的对数概率。
        beta: 温度参数。
        label_smoothing: 如果 > 0，对二元目标进行标签平滑。

    Returns:
        (loss, chosen_rewards, rejected_rewards)
        - loss: 标量 DPO 损失
        - chosen_rewards: chosen 的 reward margin（用于日志记录）
        - rejected_rewards: rejected 的 reward margin（用于日志记录）
    """
    # TODO: 实现 DPO loss
    raise NotImplementedError("dpo_loss() not implemented")


# ══════════════════════════════════════════════════════════════════════
# 加分任务: GRPO Loss
# ══════════════════════════════════════════════════════════════════════


def grpo_loss(
    log_probs: torch.Tensor,  # (batch, num_samples_per_prompt)
    advantages: torch.Tensor,  # (batch, num_samples_per_prompt)
    old_log_probs: torch.Tensor,  # (batch, num_samples_per_prompt) — 用于 ratio 计算
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    """简化版 GRPO 损失 (Group Relative Policy Optimization)。

    L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) + β * KL(π || π_old)]

    Args:
        log_probs: 当前 policy 的对数概率。
        advantages: 在每个 group 内计算的 advantage（已归一化）。
        old_log_probs: 旧 policy 的对数概率（用于 ratio clip）。
        epsilon: clip 范围。
        beta: KL 惩罚系数。

    Returns:
        标量 GRPO 损失。
    """
    # TODO: 实现 GRPO loss（加分项）
    raise NotImplementedError("grpo_loss() not implemented")


# ══════════════════════════════════════════════════════════════════════
# 任务 3: RLHF vs DPO — 知识问答
# ══════════════════════════════════════════════════════════════════════


def answer_alignment_questions() -> str:
    return """
Q1: 解释 RLHF 的三步流程

YOUR ANSWER HERE

Q2: DPO 如何通过 reparameterization 简化 RLHF？

YOUR ANSWER HERE

Q3: 分析 DPO 的优缺点（何时用 RLHF，何时用 DPO）

YOUR ANSWER HERE

Q4: GRPO 与 DPO 的区别

YOUR ANSWER HERE
"""


if __name__ == "__main__":
    print("Lab 05 starter — 实现数据管线和 DPO loss。")
