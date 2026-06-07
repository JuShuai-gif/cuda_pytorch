"""
第03讲 — Transformer：基础冒烟测试。

验证形状和梯度流；不进行基准测试。
"""

from __future__ import annotations

import torch

try:
    from .transformer import (
        MultiHeadAttention,
        TransformerBlock,
        TransformerLM,
        causal_mask,
    )
except ImportError:
    from transformer import (  # type: ignore[no-redef]
        MultiHeadAttention,
        TransformerBlock,
        TransformerLM,
        causal_mask,
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _check_shape(tensor: torch.Tensor, expected: tuple, name: str) -> bool:
    ok = tensor.shape == expected
    print(
        f"  {name}: {tensor.shape} {'✓' if ok else '✗ (expected ' + str(expected) + ')'}"
    )
    return ok


def test_causal_mask() -> None:
    """验证 causal mask 结构。"""
    cm = causal_mask(4)
    assert cm.shape == (1, 1, 4, 4)
    # 上三角应为 -inf
    for i in range(4):
        for j in range(4):
            if j > i:
                assert cm[0, 0, i, j].item() == float("-inf"), f"({i},{j}) not masked"
            else:
                assert cm[0, 0, i, j].item() == 0.0, f"({i},{j}) incorrectly masked"


def test_mha() -> None:
    """测试 MultiHeadAttention 形状。"""
    B, S, dim = 2, 8, 64
    mha = MultiHeadAttention(dim=dim, num_heads=4, num_kv_heads=2)
    x = torch.randn(B, S, dim)
    out = mha(x, mask=causal_mask(S))
    assert out.shape == (B, S, dim)


def test_mha_gradient() -> None:
    """测试 MultiHeadAttention 中的梯度流。"""
    B, S, dim = 2, 8, 64
    mha = MultiHeadAttention(dim=dim, num_heads=4, num_kv_heads=2)
    x = torch.randn(B, S, dim, requires_grad=True)
    out = mha(x, mask=causal_mask(S))
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_transformer_block() -> None:
    """测试 TransformerBlock（pre-norm / post-norm）。"""
    B, S, dim = 2, 8, 64
    cm = causal_mask(S)
    x = torch.randn(B, S, dim)
    for pre_norm in [True, False]:
        block = TransformerBlock(dim=dim, num_heads=4, pre_norm=pre_norm)
        out = block(x, mask=cm)
        assert out.shape == (B, S, dim), f"pre_norm={pre_norm} shape mismatch"


def test_transformer_lm() -> None:
    """测试 TransformerLM 前向 + 反向传播。"""
    vocab = 256
    B, S = 2, 16
    model = TransformerLM(
        vocab_size=vocab, dim=64, num_layers=2, num_heads=4, num_kv_heads=2
    )

    ids = torch.randint(0, vocab, (B, S))
    logits = model(ids)
    assert logits.shape == (B, S, vocab)

    # 梯度流
    loss = logits.sum()
    loss.backward()


def test_weight_tying() -> None:
    """测试 embedding 和 LM head 之间的 weight tying。"""
    model = TransformerLM(
        vocab_size=100, dim=32, num_layers=1, num_heads=2, weight_tying=True
    )
    assert model.lm_head.weight is model.token_embedding.weight  # 同一个张量


def test_gated_mlp() -> None:
    """测试带门控 MLP 的 TransformerBlock。"""
    B, S, dim = 2, 8, 64
    block = TransformerBlock(dim=dim, num_heads=4, gated_mlp=True)
    x = torch.randn(B, S, dim)
    out = block(x, mask=causal_mask(S))
    assert out.shape == (B, S, dim)


def test_rms_norm() -> None:
    """测试基于 RMSNorm 的 TransformerBlock。"""
    B, S, dim = 2, 8, 64
    block = TransformerBlock(dim=dim, num_heads=4, rms_norm=True)
    x = torch.randn(B, S, dim)
    out = block(x, mask=causal_mask(S))
    assert out.shape == (B, S, dim)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running transformer tests …\n")

    tests = [
        ("causal_mask", test_causal_mask),
        ("MultiHeadAttention shapes", test_mha),
        ("MultiHeadAttention gradients", test_mha_gradient),
        ("TransformerBlock (pre/post norm)", test_transformer_block),
        ("TransformerLM forward/backward", test_transformer_lm),
        ("Weight tying", test_weight_tying),
        ("Gated MLP", test_gated_mlp),
        ("RMSNorm", test_rms_norm),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed == len(tests):
        print("All tests passed!")
