"""
Lab 01 单元测试: Tokenization & Training Basics

运行方式:  python test.py
"""

import math
import unittest

import torch
import torch.nn.functional as F

from solution import (
    BPETokenizer,
    cross_entropy_loss_manual,
    compute_perplexity,
)


class TestBPETokenizer(unittest.TestCase):
    """测试 BPE 分词器的 train / encode / decode。"""

    def setUp(self) -> None:
        self.corpus = ["hello", "world", "hello", "help", "hell"]
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(self.corpus, vocab_size=30)

    def test_vocab_size(self) -> None:
        """Vocab 大小不应超过目标值。"""
        self.assertLessEqual(len(self.tokenizer.vocab), 30)

    def test_roundtrip(self) -> None:
        """对已知单词，encode -> decode 应无损往返。"""
        text = "hello"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_empty_text(self) -> None:
        """空字符串应产生空的 token 列表。"""
        encoded = self.tokenizer.encode("")
        self.assertEqual(encoded, [])

    def test_unknown_word_handling(self) -> None:
        """未知单词不应崩溃，且应产生 tokens。"""
        encoded = self.tokenizer.encode("zzzunknown")
        self.assertGreater(len(encoded), 0)
        decoded = self.tokenizer.decode(encoded)
        # 使用回退编码时，decode 应产生非空字符串
        self.assertIsInstance(decoded, str)

    def test_vocab_consistency(self) -> None:
        """Vocab 中的每个 token 都应一致映射。"""
        for token, idx in self.tokenizer.vocab.items():
            self.assertEqual(self.tokenizer.id_to_token[idx], token)

    def test_merges_sorted_deterministic(self) -> None:
        """相同的语料库应产生相同的 merges（确定性）。"""
        t1 = BPETokenizer()
        t1.train(self.corpus, vocab_size=30)
        t2 = BPETokenizer()
        t2.train(self.corpus, vocab_size=30)
        self.assertEqual(t1.merges, t2.merges)

    def test_eow_present_in_encoded(self) -> None:
        """词尾标记应出现在分词结果中。"""
        text = "hello"
        encoded = self.tokenizer.encode(text)
        decoded_tokens = [self.tokenizer.id_to_token[i] for i in encoded]
        self.assertIn(self.tokenizer.eow, decoded_tokens)


class TestCrossEntropyLoss(unittest.TestCase):
    """测试手动交叉熵实现与 PyTorch 参考实现的一致性。"""

    def test_basic(self) -> None:
        """简单情况: 手动 loss 应等于 F.cross_entropy。"""
        torch.manual_seed(123)
        logits = torch.randn(8, 50)
        targets = torch.randint(0, 50, (8,))
        manual = cross_entropy_loss_manual(logits, targets)
        ref = F.cross_entropy(logits, targets)
        self.assertTrue(
            torch.allclose(manual, ref, atol=1e-6),
            f"manual={manual:.6f}, ref={ref:.6f}",
        )

    def test_ignore_index(self) -> None:
        """带有 ignore_index 的位置不应影响 loss。"""
        torch.manual_seed(42)
        logits = torch.randn(6, 20)
        targets = torch.tensor([1, 2, 3, 4, 5, 6])
        manual = cross_entropy_loss_manual(logits, targets, ignore_index=-100)
        ref = F.cross_entropy(logits, targets, ignore_index=-100)
        self.assertTrue(torch.allclose(manual, ref, atol=1e-6))

        # 所有 target 被忽略时应返回 0
        targets_ignored = torch.full((6,), -100)
        manual_zero = cross_entropy_loss_manual(logits, targets_ignored)
        self.assertTrue(torch.allclose(manual_zero, torch.tensor(0.0)))

    def test_perfect_prediction(self) -> None:
        """如果模型 100% 自信且正确，loss ~ 0。"""
        logits = torch.zeros(1, 10)
        logits[0, 5] = 1e9  # 给索引 5 极高的 logit 值
        targets = torch.tensor([5])
        manual = cross_entropy_loss_manual(logits, targets)
        self.assertLess(manual.item(), 0.01)

    def test_batch_shape(self) -> None:
        """应能处理 2D 和 3D 的 logits。"""
        # 2D: (batch, vocab)
        logits_2d = torch.randn(4, 10)
        targets_2d = torch.randint(0, 10, (4,))
        loss_2d = cross_entropy_loss_manual(logits_2d, targets_2d)
        self.assertEqual(loss_2d.dim(), 0)  # 标量

        # 3D: (batch, seq_len, vocab)
        logits_3d = torch.randn(2, 5, 10)
        targets_3d = torch.randint(0, 10, (2, 5))
        loss_3d = cross_entropy_loss_manual(logits_3d, targets_3d)
        self.assertEqual(loss_3d.dim(), 0)


class TestPerplexity(unittest.TestCase):
    """测试困惑度计算。"""

    def test_definition(self) -> None:
        """PPL 应等于 exp(cross-entropy)。"""
        torch.manual_seed(99)
        logits = torch.randn(5, 30)
        targets = torch.randint(0, 30, (5,))
        ppl = compute_perplexity(logits, targets)
        ce = F.cross_entropy(logits, targets).item()
        self.assertAlmostEqual(ppl, math.exp(ce), places=5)

    def test_perfect_perplexity(self) -> None:
        """完美模型 -> PPL ~ 1。"""
        logits = torch.zeros(1, 10)
        logits[0, 3] = 1e9
        targets = torch.tensor([3])
        ppl = compute_perplexity(logits, targets)
        self.assertAlmostEqual(ppl, 1.0, places=4)

    def test_uniform_perplexity(self) -> None:
        """均匀 logits -> PPL ~ vocab_size。"""
        V = 50
        logits = torch.zeros(1, V)  # 全零 -> softmax 后均匀分布
        targets = torch.randint(0, V, (1,))
        ppl = compute_perplexity(logits, targets)
        self.assertAlmostEqual(ppl, float(V), delta=0.5)


if __name__ == "__main__":
    unittest.main()
