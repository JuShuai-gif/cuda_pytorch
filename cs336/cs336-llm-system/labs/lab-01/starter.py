"""
Lab 01: Tokenization & Training Basics — 起始代码

完成以下占位实现：
  - BPETokenizer: train, encode, decode
  - cross_entropy_loss_manual
  - compute_perplexity

不要导入任何分词相关的库（tiktoken、tokenizers 等）。
仅使用 Python 内置的 collections 和 typing 模块。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# 任务 1: Byte-Pair Encoding (BPE)
# ──────────────────────────────────────────────────────────────────────


class BPETokenizer:
    """一个简单的 Byte-Pair Encoding 分词器。"""

    def __init__(self) -> None:
        # vocab: token字符串 -> token_id
        self.vocab: Dict[str, int] = {}
        # reverse: token_id -> token字符串
        self.id_to_token: Dict[int, str] = {}
        # 按顺序排列的 merge 规则: (a, b) 对列表
        self.merges: List[Tuple[str, str]] = []
        # 特殊的词尾标记
        self.eow: str = "</w>"

    # ------------------------------------------------------------------
    # 你的代码: train(corpus, vocab_size)
    #
    # 1. 将每个词拆分为字符，并在末尾添加 self.eow
    #    示例: "hello" -> ["h", "e", "l", "l", "o", "</w>"]
    #
    # 2. 用所有唯一字符（+ eow 标记）初始化 vocab
    #
    # 3. 当 len(vocab) < vocab_size 时循环:
    #    a) 统计所有相邻 token 对的频率
    #    b) 找到频率最高的 pair
    #    c) 在所有词中合并该 pair
    #    d) 将新 token 加入 vocab
    #    e) 记录该 merge 规则
    #
    # 4. 存储 self.vocab、self.id_to_token、self.merges
    # ------------------------------------------------------------------

    def train(self, corpus: List[str], vocab_size: int) -> None:
        """在单词字符串列表上训练 BPE 分词器。

        Args:
            corpus: 单词（字符串）列表。
            vocab_size: 目标词汇量大小（包含基础字符）。
        """
        # TODO: 实现 BPE 训练
        raise NotImplementedError("train() not implemented")

    # ------------------------------------------------------------------
    # 你的代码: encode(text)
    #
    # 1. 将文本按空格分割为单词（为简化处理）
    # 2. 对每个单词，先按字符级别拆分
    # 3. 按顺序对每个单词应用 merge 规则
    # 4. 将得到的 tokens 映射为对应的 ID
    # 5. 返回 token ID 列表
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """将文本字符串编码为 token ID 列表。

        Args:
            text: 输入文本字符串。

        Returns:
            整数 token ID 列表。
        """
        # TODO: 实现编码
        raise NotImplementedError("encode() not implemented")

    # ------------------------------------------------------------------
    # 你的代码: decode(ids)
    #
    # 1. 将每个 ID 映射回对应的 token 字符串
    # 2. 拼接所有 tokens
    # 3. 移除 self.eow 标记并还原空格
    # 4. 返回重建后的文本字符串
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """将 token ID 列表解码回文本字符串。

        Args:
            ids: 整数 token ID 列表。

        Returns:
            解码后的文本字符串。
        """
        # TODO: 实现解码
        raise NotImplementedError("decode() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 2: Cross-Entropy Loss（手动实现）
# ──────────────────────────────────────────────────────────────────────


def cross_entropy_loss_manual(
    logits: torch.Tensor,  # shape: (batch, vocab_size) 或 (seq_len, vocab_size)
    targets: torch.Tensor,  # shape: (batch,) 或 (seq_len,)
    ignore_index: int = -100,
) -> torch.Tensor:
    """手动计算交叉熵损失。

    实现步骤:
    1. 从 logits 计算 softmax 概率
    2. 对每个 target，计算 -log(prob[target])
    3. 对未被忽略的位置求平均

    使用 log-sum-exp 技巧保证数值稳定性。

    Args:
        logits: 模型输出的原始 logits。
        targets: 真实 token 索引。
        ignore_index: 在损失计算中忽略的索引。

    Returns:
        标量交叉熵损失值。
    """
    # TODO: 实现手动交叉熵损失
    raise NotImplementedError("cross_entropy_loss_manual() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 3: Perplexity（困惑度）
# ──────────────────────────────────────────────────────────────────────


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """从 logits 和 targets 计算困惑度（perplexity）。

    Perplexity = exp(cross-entropy loss)

    Args:
        logits: 模型输出的原始 logits。
        targets: 真实 token 索引。
        ignore_index: 要忽略的索引。

    Returns:
        困惑度（浮点数）。
    """
    # TODO: 实现困惑度计算
    raise NotImplementedError("compute_perplexity() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 工具函数: 与 PyTorch 对比验证正确性
# ──────────────────────────────────────────────────────────────────────


def verify_loss_implementation() -> bool:
    """快速检查手动实现的 loss 是否与 F.cross_entropy 一致。"""
    torch.manual_seed(42)
    logits = torch.randn(4, 100)
    targets = torch.randint(0, 100, (4,))

    manual = cross_entropy_loss_manual(logits, targets, ignore_index=-100)
    reference = F.cross_entropy(logits, targets, ignore_index=-100)

    return torch.allclose(manual, reference, atol=1e-6)


if __name__ == "__main__":
    print("Lab 01 starter — 运行 'python test.py' 验证你的实现。")
