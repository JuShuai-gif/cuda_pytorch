"""
从零实现 Byte-Pair Encoding (BPE) tokenizer。

实现了以下论文中描述的 BPE 算法：
  Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"

该 tokenizer 在字节层面工作：
  - 基础词汇表：256 个单字节 token（0-255）
  - 训练：迭代地合并最频繁的相邻 token 对
  - 编码：按顺序应用学到的合并规则，生成 token ID 序列
  - 解码：将 token ID 映射回字节，然后解码为 UTF-8 字符串

包含自包含的测试用例，可在直接执行该文件时运行。

用法：
    from bpe import BPETokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(corpus_text, vocab_size=300)
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """基于字节级基础词汇表的 Byte-Pair Encoding tokenizer。

    Attributes:
        vocab: 从 token ID 到其字节表示的映射。
        merges: 从 token 对到合并后 token ID 的有序映射。
                顺序很重要：编码时先应用较早的合并规则。
    """

    def __init__(self) -> None:
        # 基础词汇表：每个字节值（0-255）对应一个条目
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # 学到的合并规则：(id_a, id_b) -> new_merged_id
        self.merges: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        """在 `text` 上训练 BPE tokenizer，使其达到 `vocab_size` 个 token。

        从 256 个单字节 token 开始，执行 `vocab_size - 256` 次合并操作。
        每次迭代：
          1. 统计当前序列中所有相邻 token 对的频率。
          2. 选择频率最高的 token 对。
          3. 将该对合并为一个新的 token ID，并更新 vocab/merges。

        Args:
            text: 训练语料（UTF-8 字符串）。
            vocab_size: 目标词汇表大小，必须 >= 256。
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")

        # 使用整数 ID 来表示（初始时为字节值）
        ids: list[int] = list(text.encode("utf-8"))
        num_merges: int = vocab_size - 256

        for i in range(num_merges):
            # 统计相邻 token 对的频率
            pair_counts = self._count_adjacent_pairs(ids)
            if not pair_counts:
                break  # 没有更多可合并的 token 对

            # 选出频率最高的 token 对
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])

            # 为合并后的 token 对创建新的 token ID
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 将合并应用到当前 ID 序列
            ids = self._merge(ids, best_pair, new_id)

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """将 UTF-8 字符串编码为 token ID 列表。

        从单个字节开始，然后迭代地应用学到的合并规则。
        每一步选择具有**最小**新 token ID 的合并（即训练期间最早学到的合并），
        这在 BPE 中是标准做法。

        Args:
            text: 要分词的输入字符串。

        Returns:
            token ID 列表。
        """
        ids: list[int] = list(text.encode("utf-8"))

        if not self.merges:
            return ids

        while len(ids) >= 2:
            # 找到所有相邻 token 对，并选择其合并 ID 最小的那个
            best_pair: tuple[int, int] | None = None
            best_merge_id: int = float("inf")  # type: ignore[assignment]

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                merge_id = self.merges.get(pair)
                if merge_id is not None and merge_id < best_merge_id:
                    best_pair = pair
                    best_merge_id = merge_id

            if best_pair is None:
                break  # 没有更多可应用的合并规则

            ids = self._merge(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        """将 token ID 列表解码回 UTF-8 字符串。

        Args:
            ids: 要解码的 token ID 列表。

        Returns:
            解码后的 UTF-8 字符串。对于无效的字节序列，
            使用 'replace' 错误处理方式。
        """
        tokens: bytes = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 静态辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _count_adjacent_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
        """统计 token ID 序列中每个相邻 token 对的出现次数。"""
        counts: dict[tuple[int, int], int] = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """将 ID 序列中的每个 `pair` 出现替换为 `new_id`。"""
        new_ids: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """当前词汇表大小。"""
        return len(self.vocab)

    def compression_ratio(self, text: str) -> float:
        """返回给定文本的 UTF-8 字节数与 token 数之比。

        值越高表示压缩效果越好（每个字节对应更少的 token）。
        """
        ids = self.encode(text)
        num_bytes = len(text.encode("utf-8"))
        num_tokens = len(ids)
        if num_tokens == 0:
            return 0.0
        return num_bytes / num_tokens


# ---------------------------------------------------------------------------
# 测试用例（运行方式：python bpe.py）
# ---------------------------------------------------------------------------


def run_tests() -> None:
    """运行 BPETokenizer 类的所有测试用例。"""

    # ------------------------------------------------------------------
    # 测试 1：小词汇表的基本往返测试
    # ------------------------------------------------------------------
    print("Test 1: Basic encode/decode round-trip ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = "the cat in the hat the cat on the mat"
    tokenizer.train(corpus, vocab_size=270)  # 256 + 14 次合并

    test_strings = [
        "the",
        "the cat",
        "the cat in the hat",
        "hello world",
    ]
    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        assert decoded == s, f"Round-trip failed: {s!r} -> {ids} -> {decoded!r}"
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 2：空字符串
    # ------------------------------------------------------------------
    print("Test 2: Empty string encoding ...", end=" ")
    tokenizer = BPETokenizer()
    tokenizer.train("hello world", vocab_size=260)
    ids = tokenizer.encode("")
    assert ids == [], f"Expected [], got {ids}"
    decoded = tokenizer.decode(ids)
    assert decoded == "", f"Expected '', got {decoded!r}"
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 3：Unicode 处理（Emoji、中文、带重音字符）
    # ------------------------------------------------------------------
    print("Test 3: Unicode handling ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = "Hello, 🌍! 你好! こんにちは! Café résumé naïve"
    tokenizer.train(corpus, vocab_size=512)

    unicode_strings = [
        "Hello, 🌍!",
        "你好世界",
        "こんにちは",
        "Café résumé",
        "🌍🌍🌍",
        "αβγδε",
    ]
    for s in unicode_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
    # 所有字符串都应能成功解码，不抛出异常
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 4：Unicode 字符串往返测试
    # ------------------------------------------------------------------
    print("Test 4: Unicode round-trip ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = (
        "Hello, 🌍! 你好世界! こんにちは! Café résumé naïve "
        "the quick brown fox jumps over the lazy dog "
        "the quick brown fox the quick brown fox "
    )
    tokenizer.train(corpus, vocab_size=512)

    original = "Hello, 🌍! 你好世界! Café résumé"
    ids = tokenizer.encode(original)
    decoded = tokenizer.decode(ids)
    assert decoded == original, (
        f"Unicode round-trip failed: {original!r} != {decoded!r}"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 5：更大词汇表能提高压缩比
    # ------------------------------------------------------------------
    print("Test 5: Compression ratio improves ...", end=" ")
    corpus = (
        "the cat in the hat the cat on the mat "
        "the quick brown fox jumps over the lazy dog "
        "the cat the hat the mat the fox the dog "
    ) * 10

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=256)  # 不进行合并
    ratio_baseline = tokenizer.compression_ratio(corpus)
    assert ratio_baseline == 1.0, f"Baseline ratio should be 1.0, got {ratio_baseline}"

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300)
    ratio_trained = tokenizer.compression_ratio(corpus)
    assert ratio_trained > 1.0, (
        f"Trained ratio ({ratio_trained}) must exceed baseline ({ratio_baseline})"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 6：词汇表大小正确
    # ------------------------------------------------------------------
    print("Test 6: Vocabulary size ...", end=" ")
    tokenizer = BPETokenizer()
    assert tokenizer.vocab_size == 256, (
        f"Initial vocab should be 256, got {tokenizer.vocab_size}"
    )

    tokenizer.train("hello world", vocab_size=300)
    # 应在 256 到 300 之间（如果文本太短，可能小于 300）
    assert 256 <= tokenizer.vocab_size <= 300, (
        f"Trained vocab should be in [256, 300], got {tokenizer.vocab_size}"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 7：重复文本能产生有意义的合并
    # ------------------------------------------------------------------
    print("Test 7: Repetitive text merges ...", end=" ")
    tokenizer = BPETokenizer()
    tokenizer.train("abababab cdcdcdcd abababab cdcdcdcd", vocab_size=260)
    # 应该已经合并了一些模式
    assert tokenizer.vocab_size > 256, "Expected merges to increase vocabulary"
    ids = tokenizer.encode("abab")
    # 如果 'ab' 已被合并，产生的 token 数应少于字节数
    assert len(ids) < 4, f"Expected <4 tokens for 'abab', got {len(ids)}"
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 8：特殊字符（换行符、制表符等）
    # ------------------------------------------------------------------
    print("Test 8: Special characters ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = "line1\nline2\nline3\tindented\ttext"
    tokenizer.train(corpus, vocab_size=270)

    test = "line1\nline2"
    ids = tokenizer.encode(test)
    decoded = tokenizer.decode(ids)
    assert decoded == test, f"Special char round-trip failed: {test!r} != {decoded!r}"
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 9：在真实文本上使用较大词汇表
    # ------------------------------------------------------------------
    print("Test 9: Large vocab ...", end=" ")
    tokenizer = BPETokenizer()
    corpus = (
        "Byte-Pair Encoding (BPE) is a subword tokenization algorithm "
        "originally developed for data compression and later adapted "
        "for neural machine translation. The key idea is to start with "
        "individual characters (or bytes) and iteratively merge the most "
        "frequent adjacent pair of tokens. This produces a vocabulary "
        "that balances character-level and word-level representations. "
        "BPE was popularized in NLP by Sennrich et al. (2016) and is "
        "used by GPT-2, RoBERTa, and many other language models."
    )
    tokenizer.train(corpus, vocab_size=400)
    assert tokenizer.vocab_size > 256, "Expected merges to increase vocabulary"
    orig_ids = tokenizer.encode(corpus)
    decoded = tokenizer.decode(orig_ids)
    assert decoded == corpus, "Full corpus round-trip failed"
    print("PASSED")

    # ------------------------------------------------------------------
    # 测试 10：确定性行为
    # ------------------------------------------------------------------
    print("Test 10: Deterministic encoding ...", end=" ")
    tokenizer = BPETokenizer()
    tokenizer.train("aaaaabbbbb aaaaabbbbb", vocab_size=260)
    ids1 = tokenizer.encode("aaaaabbbbb")
    ids2 = tokenizer.encode("aaaaabbbbb")
    assert ids1 == ids2, f"Encoding should be deterministic: {ids1} != {ids2}"
    print("PASSED")

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
