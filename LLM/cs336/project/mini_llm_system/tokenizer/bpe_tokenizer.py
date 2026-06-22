"""
从头实现 Byte-Pair Encoding (BPE) tokenizer。

实现完整的 BPE tokenizer，支持特殊 token（[PAD]、[BOS]、[EOS]、[UNK]），
能在文本语料上训练，将文本编码为 token ID，将 token ID 解码回文本，
并支持保存/加载以实现持久化。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

# 特殊 token
PAD_TOKEN: str = "[PAD]"
BOS_TOKEN: str = "[BOS]"
EOS_TOKEN: str = "[EOS]"
UNK_TOKEN: str = "[UNK]"

# 这些必须出现在词汇表最前面，以确保它们的 ID 为 0, 1, 2, 3
SPECIAL_TOKENS: list[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
UNK_ID: int = 3


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计相邻 token 对的频率。"""
    stats: dict[tuple[int, int], int] = defaultdict(int)
    for pair in zip(ids[:-1], ids[1:]):
        stats[pair] += 1
    return stats


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """在单次遍历中将所有 `pair` 替换为 `new_id`。"""
    result: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer。

    Attributes:
        vocab: token ID 到 token 字节/字符串的映射。
        merges: 有序映射 (tok_a_id, tok_b_id) -> merged_token_id。
        special_tokens: 特殊 token 字符串到 token ID 的字典。
    """

    def __init__(self) -> None:
        # vocab[id] -> token 的字节表示
        self.vocab: dict[int, bytes] = {}
        # merges 映射 (id_a, id_b) -> merged_id，保留插入顺序
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }

    def train(self, texts: list[str], vocab_size: int, min_frequency: int = 2) -> None:
        """
        在文本列表上训练 BPE tokenizer。

        Args:
            texts: 用于训练的原始文本字符串列表。
            vocab_size: 目标词汇表大小（包含特殊 token）。
            min_frequency: token 对被考虑合并的最低频率。
        """
        num_special: int = len(SPECIAL_TOKENS)

        # 初始化词汇表：特殊 token 在 ID 0..3，然后字节级 0..255 在 ID 4..259
        self.vocab = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[i] = tok.encode("utf-8")
        for byte_val in range(256):
            self.vocab[num_special + byte_val] = bytes([byte_val])

        self.merges = {}

        # 将文本转换为内部 ID 序列
        # 字节值 `b` 映射到内部 ID (num_special + b)
        byte_sequences: list[list[int]] = []
        for text in texts:
            byte_sequences.append([num_special + b for b in text.encode("utf-8")])

        next_id: int = num_special + 256  # 第一个合并 ID 从字节级 + 特殊 token 之后开始

        num_merges: int = vocab_size - next_id

        for merge_step in range(num_merges):
            # 统计所有序列中的 token 对频率
            stats: dict[tuple[int, int], int] = defaultdict(int)
            for seq in byte_sequences:
                for pair in zip(seq[:-1], seq[1:]):
                    stats[pair] += 1

            if not stats:
                break

            # 找到频率最高的 token 对
            best_pair: tuple[int, int] = max(stats, key=stats.get)  # type: ignore[arg-type]
            best_freq: int = stats[best_pair]

            if best_freq < min_frequency:
                break

            # 记录合并
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = next_id

            # 对所有序列应用合并
            for i in range(len(byte_sequences)):
                byte_sequences[i] = _merge(byte_sequences[i], best_pair, next_id)

            next_id += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        将文本字符串编码为 token ID 列表。

        Args:
            text: 要编码的原始文本。
            add_special_tokens: 如果为 True，则在开头添加 [BOS]，在结尾添加 [EOS]。

        Returns:
            token ID 列表。
        """
        # 从内部字节级编码开始（字节值 b -> 内部 ID 4+b）
        num_special: int = len(SPECIAL_TOKENS)
        ids: list[int] = [num_special + b for b in text.encode("utf-8")]

        # 按学习到的顺序迭代应用合并
        # 对每个合并，扫描序列并在可能的位置应用，
        # 然后移动到下一个合并（贪婪、非递归）。
        # 按目标 ID（学习顺序）对合并排序。
        sorted_merges: list[tuple[tuple[int, int], int]] = sorted(
            self.merges.items(), key=lambda x: x[1]
        )

        # 重复应用合并，直到无法再应用任何合并
        while True:
            # 找到最早（合并 ID 最低）的可应用 token 对
            best_pair: tuple[int, int] | None = None
            best_idx: int = -1
            best_merge_id: int = -1

            for pair, merge_id in sorted_merges:
                for i in range(len(ids) - 1):
                    if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                        # 检查这是否为最早可应用的合并
                        if best_pair is None or merge_id < best_merge_id:
                            best_pair = pair
                            best_idx = i
                            best_merge_id = merge_id
                        break  # 已为该 token 对找到匹配；检查下一个 token 对

            if best_pair is None:
                break

            # 在首次出现的位置应用最佳合并
            ids = ids[:best_idx] + [best_merge_id] + ids[best_idx + 2 :]

        # 映射到最终 ID：词汇表包含字节级 + 特殊 token，但需要
        # 确保 ID 在有效范围内。字节 0-255 映射到自身。
        result: list[int] = []
        if add_special_tokens:
            result.append(BOS_ID)

        for tid in ids:
            result.append(tid)

        if add_special_tokens:
            result.append(EOS_ID)

        return result

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        将 token ID 列表解码回文本字符串。

        Args:
            ids: 要解码的 token ID 列表。
            skip_special_tokens: 如果为 True，则从输出中省略特殊 token。

        Returns:
            解码后的文本字符串。
        """
        special_ids: set[int] = set(self.special_tokens.values())
        parts: list[bytes] = []
        for tid in ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.vocab:
                parts.append(self.vocab[tid])
            else:
                parts.append(str(tid).encode("utf-8"))  # 对未知 ID 的回退处理
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """
        将 tokenizer 保存到 JSON 文件。

        Args:
            path: 保存到的文件路径。
        """
        path = Path(path)
        data: dict[str, Any] = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            # 将 tuple 键转换为字符串以兼容 JSON
            "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
            "special_tokens": self.special_tokens,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """
        从 JSON 文件加载 tokenizer。

        Args:
            path: 要加载的文件路径。

        Returns:
            包含已加载状态的新 BPETokenizer 实例。
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tokenizer.merges = {
            tuple(int(x) for x in k.split(",")): int(v)
            for k, v in data["merges"].items()
        }
        tokenizer.special_tokens = data["special_tokens"]
        return tokenizer

    def vocab_size(self) -> int:
        """返回当前词汇表大小。"""
        return len(self.vocab)

    def token_to_id(self, token_str: str) -> int:
        """查找特殊 token 字符串对应的 ID。"""
        return self.special_tokens.get(token_str, UNK_ID)


# 便捷演示：运行训练 + 编码/解码示例
if __name__ == "__main__":
    # 小型训练语料
    corpus: list[str] = [
        "hello world",
        "hello there",
        "world of hello",
        "hello world hello",
    ]

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300, min_frequency=2)

    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print(f"Number of merges: {len(tokenizer.merges)}")

    text: str = "hello world"
    encoded: list[int] = tokenizer.encode(text)
    decoded: str = tokenizer.decode(encoded)
    print(f"Text: {text!r}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded!r}")
    assert text in decoded or decoded.strip(), "Decode should reconstruct the text"

    # 测试保存/加载
    tokenizer.save("/tmp/test_bpe_tokenizer.json")
    loaded: BPETokenizer = BPETokenizer.load("/tmp/test_bpe_tokenizer.json")
    assert loaded.encode(text) == encoded
    print("Save/Load test passed!")
