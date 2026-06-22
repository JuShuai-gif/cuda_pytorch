"""
Lab 01 解答: Tokenization & Training Basics

BPE 分词器、交叉熵损失与困惑度的完整实现。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 任务 1: Byte-Pair Encoding (BPE)
# ══════════════════════════════════════════════════════════════════════


class BPETokenizer:
    """一个简单的 Byte-Pair Encoding 分词器。"""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.eow: str = "</w>"

    # ------------------------------------------------------------------
    def train(self, corpus: List[str], vocab_size: int) -> None:
        """训练 BPE 分词器。每个单词被拆分为字符并追加 EOW 标记。"""
        # 步骤 1: 将每个单词表示为字符 token 列表
        # 每个条目: 一个单词的 token 字符串列表
        word_splits: List[List[str]] = [list(word) + [self.eow] for word in corpus]

        # 步骤 2: 构建初始 vocab（所有唯一字符）
        chars: set[str] = set()
        for word in word_splits:
            chars.update(word)
        # 排序以保证多个 pair 频率相同时的确定性
        sorted_chars = sorted(chars)
        vocab_list = sorted_chars[:]  # 可变副本

        self.merges = []

        # 步骤 3: 迭代直到达到目标 vocab 大小
        while len(vocab_list) < vocab_size:
            pair_counts: Counter = Counter()
            for word in word_splits:
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += 1

            if not pair_counts:
                break  # 无法继续合并

            # 最佳 pair: 频率最高; 平局时按字典序
            best_pair: Tuple[str, str] = max(
                pair_counts, key=lambda p: (pair_counts[p], p)
            )
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            # 在所有单词上进行合并
            for idx, word in enumerate(word_splits):
                new_word: List[str] = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word_splits[idx] = new_word

            vocab_list.append(new_token)

        # 步骤 4: 构建 vocab 字典
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """按顺序应用 merge 规则，将文本转换为 ID 序列。"""
        words = text.split()
        ids: List[int] = []

        for word in words:
            tokens = list(word) + [self.eow]

            # 按顺序应用 merge 规则
            for a, b in self.merges:
                new_tokens: List[str] = []
                i = 0
                while i < len(tokens):
                    if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for tok in tokens:
                # 对未见过的 token 回退到类似 UNK 的编码:
                # 按字符逐一编码
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
                else:
                    for ch in tok:
                        ids.append(self.vocab.get(ch, 0))

        return ids

    # ------------------------------------------------------------------
    def decode(self, ids: List[int]) -> str:
        """将 ID 序列转换回文本。"""
        tokens = [self.id_to_token.get(i, "?") for i in ids]
        text = "".join(tokens)
        # 从 EOW 标记还原空格
        text = text.replace(self.eow, " ")
        return text.strip()


# ══════════════════════════════════════════════════════════════════════
# 任务 2: Cross-Entropy Loss（手动实现）
# ══════════════════════════════════════════════════════════════════════


def cross_entropy_loss_manual(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """通过 log-softmax + NLL 手动计算交叉熵损失。

    使用 log-sum-exp 技巧保证数值稳定性。
    """
    # log_softmax(x_i) = x_i - log(sum(exp(x_j)))
    # 使用 LSE 技巧: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    x_max = logits.max(dim=-1, keepdim=True).values  # (..., 1)
    lse = x_max + torch.log(
        torch.sum(torch.exp(logits - x_max), dim=-1, keepdim=True)
    )  # (..., 1)
    log_probs = logits - lse  # (..., V)

    # 收集目标 token 的 log-prob
    # targets shape: (...)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 屏蔽被忽略的位置
    mask = (targets != ignore_index).float()
    loss = (nll * mask).sum() / mask.sum().clamp(min=1)

    return loss


# ══════════════════════════════════════════════════════════════════════
# 任务 3: Perplexity（困惑度）
# ══════════════════════════════════════════════════════════════════════


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Perplexity = exp(cross-entropy loss)。"""
    loss = cross_entropy_loss_manual(logits, targets, ignore_index=ignore_index)
    return math.exp(loss.item())


# ══════════════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════════════


def verify_loss_implementation() -> bool:
    """快速检查手动实现的 loss 是否与 F.cross_entropy 一致。"""
    torch.manual_seed(42)
    logits = torch.randn(4, 100)
    targets = torch.randint(0, 100, (4,))

    manual = cross_entropy_loss_manual(logits, targets, ignore_index=-100)
    reference = F.cross_entropy(logits, targets, ignore_index=-100)

    return torch.allclose(manual, reference, atol=1e-6)


if __name__ == "__main__":
    print("=== Lab 01 解答验证 ===\n")

    # 验证 BPE
    corpus = ["hello", "world", "hello", "help", "hell"]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=30)
    print(f"Vocab 大小: {len(tokenizer.vocab)}")
    print(f"Merges: {tokenizer.merges[:5]}...")

    test_text = "hello world"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Encode('{test_text}') = {encoded}")
    print(f"Decode -> '{decoded}'")

    # 验证 loss
    ok = verify_loss_implementation()
    print(f"\nCross-entropy 验证: {'PASS' if ok else 'FAIL'}")

    # 验证 perplexity
    logits = torch.randn(2, 10)
    targets = torch.tensor([3, 7])
    ppl = compute_perplexity(logits, targets)
    print(f"Perplexity: {ppl:.4f}")
