"""
BPE tokenizer 的单元测试。
"""

import os
import sys
import tempfile

# 确保包可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe_tokenizer import (
    BPETokenizer,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
)


def test_special_tokens() -> None:
    """验证特殊 token ID 是否正确。"""
    tokenizer = BPETokenizer()
    assert tokenizer.special_tokens["[PAD]"] == PAD_ID
    assert tokenizer.special_tokens["[BOS]"] == BOS_ID
    assert tokenizer.special_tokens["[EOS]"] == EOS_ID
    assert tokenizer.special_tokens["[UNK]"] == UNK_ID
    print("test_special_tokens: PASSED")


def test_train_and_encode() -> None:
    """在小语料库上训练 tokenizer 并验证编解码往返一致性。"""
    corpus: list[str] = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox",
        "hello world hello world hello world",
        "abcdefghijklmnopqrstuvwxyz",
    ]

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300, min_frequency=2)

    # 验证词汇表已经增长
    base_vocab_size: int = 256 + 4  # 字节 + 特殊 token
    assert tokenizer.vocab_size() >= base_vocab_size, (
        f"Vocab should be >= {base_vocab_size}, got {tokenizer.vocab_size()}"
    )

    test_text: str = "the quick brown fox"
    encoded: list[int] = tokenizer.encode(test_text, add_special_tokens=True)
    decoded: str = tokenizer.decode(encoded, skip_special_tokens=True)

    # 经过 BPE 合并后，解码文本应与输入一致
    assert decoded == test_text, f"Round-trip failed: {test_text!r} != {decoded!r}"
    print("test_train_and_encode: PASSED")


def test_encode_without_special_tokens() -> None:
    """测试不带特殊 token 的编码。"""
    corpus: list[str] = ["hello world"]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300, min_frequency=1)

    encoded: list[int] = tokenizer.encode("hello", add_special_tokens=False)
    assert BOS_ID not in encoded, "BOS should not be present"
    assert EOS_ID not in encoded, "EOS should not be present"
    print("test_encode_without_special_tokens: PASSED")


def test_save_and_load() -> None:
    """测试保存/加载的持久化功能。"""
    corpus: list[str] = ["the quick brown fox", "the lazy dog"]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=280, min_frequency=1)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path: str = f.name
    try:
        tokenizer.save(tmp_path)
        loaded: BPETokenizer = BPETokenizer.load(tmp_path)

        assert loaded.vocab_size() == tokenizer.vocab_size(), "Vocab size mismatch"
        assert len(loaded.merges) == len(tokenizer.merges), "Merges count mismatch"

        test_text: str = "the quick brown fox"
        assert loaded.encode(test_text) == tokenizer.encode(test_text), (
            "Encode mismatch after load"
        )
        assert loaded.decode(loaded.encode(test_text)) == tokenizer.decode(
            tokenizer.encode(test_text)
        ), "Decode mismatch after load"
        print("test_save_and_load: PASSED")
    finally:
        os.unlink(tmp_path)


def test_unknown_token() -> None:
    """测试对训练中未出现字符的处理。"""
    corpus: list[str] = ["abc"]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=270, min_frequency=1)

    # 对包含训练数据中未出现字符的文本进行编码
    encoded: list[int] = tokenizer.encode("xyz", add_special_tokens=False)
    decoded: str = tokenizer.decode(encoded, skip_special_tokens=True)
    # 不应崩溃；回退机制会处理未知字节
    assert len(decoded) > 0, "Decode should produce output even for unknown tokens"
    print("test_unknown_token: PASSED")


if __name__ == "__main__":
    test_special_tokens()
    test_train_and_encode()
    test_encode_without_special_tokens()
    test_save_and_load()
    test_unknown_token()
    print("\nAll tokenizer tests passed!")
