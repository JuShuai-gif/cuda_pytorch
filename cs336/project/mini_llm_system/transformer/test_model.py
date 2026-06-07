"""
Transformer 模型的测试。

验证张量形状、前向传播以及各组件级别的正确性。
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from transformer.config import MiniLLMConfig
from transformer.layers import MiniLLM, SwiGLU_FFN, TransformerBlock
from transformer.attention import (
    ScaledDotProductAttention,
    CausalAttention,
    GroupedQueryAttention,
    FlashAttentionSimple,
)
from transformer.normalization import RMSNorm
from transformer.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


def test_config() -> None:
    """测试 config 的初始化和参数验证。"""
    config = MiniLLMConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=3072,
    )
    assert config.head_dim == 64, f"head_dim should be 64, got {config.head_dim}"
    assert config.hidden_size // config.num_heads == config.head_dim
    print("test_config: PASSED")


def test_rms_norm() -> None:
    """测试 RMSNorm 的张量形状和归一化功能。"""
    norm = RMSNorm(hidden_size=768, eps=1e-5)
    x = torch.randn(2, 16, 768)
    out = norm(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    # 输出的 RMS 应接近 1
    rms = torch.sqrt(out.float().pow(2).mean(dim=-1) + 1e-5).mean()
    assert 0.9 < rms < 1.1, f"RMS should be ~1.0, got {rms:.4f}"
    print("test_rms_norm: PASSED")


def test_rope() -> None:
    """测试 RoPE 的预计算和应用。"""
    dim, max_len = 64, 128
    rope = RotaryEmbedding(dim=dim, max_seq_len=max_len)

    cos, sin = rope.forward(16)
    assert cos.shape == (1, 1, 16, dim), f"cos shape: {cos.shape}"
    assert sin.shape == (1, 1, 16, dim), f"sin shape: {sin.shape}"

    q = torch.randn(2, 8, 16, dim)
    k = torch.randn(2, 8, 16, dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("test_rope: PASSED")


def test_scaled_dot_product_attention() -> None:
    """测试基本的 attention 张量形状。"""
    batch, n_heads, seq, head_dim = 2, 8, 32, 64
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    v = torch.randn(batch, n_heads, seq, head_dim)

    attn = ScaledDotProductAttention()
    out = attn(q, k, v)
    assert out.shape == q.shape, f"Shape: {out.shape}"
    print("test_scaled_dot_product_attention: PASSED")


def test_causal_attention() -> None:
    """测试因果性：位置 i 不能关注大于 i 的位置。"""
    batch, n_heads, seq, head_dim = 1, 1, 4, 8
    q = torch.zeros(batch, n_heads, seq, head_dim)
    k = torch.zeros(batch, n_heads, seq, head_dim)
    v = (
        torch.arange(seq, dtype=torch.float32)
        .view(1, 1, seq, 1)
        .expand(batch, n_heads, seq, head_dim)
    )

    attn = CausalAttention()
    out = attn(q, k, v)

    # 由于 q 和 k 为零，attention 在可用位置上均匀分布
    # 位置 0：仅关注位置 0 => 输出 = v[0]
    # 位置 1：关注位置 0,1 => 输出 = mean(v[0], v[1])
    for pos in range(seq):
        expected: torch.Tensor = v[:, :, : pos + 1, :].mean(dim=2)
        assert torch.allclose(out[:, :, pos, :], expected[:, :, :], atol=1e-5), (
            f"Causality broken at position {pos}"
        )
    print("test_causal_attention: PASSED")


def test_grouped_query_attention() -> None:
    """测试 GQA 前向传播和 KV cache。"""
    batch, seq, hidden = 2, 16, 256
    config = MiniLLMConfig(
        hidden_size=hidden,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=512,
    )

    gqa = GroupedQueryAttention(
        hidden_size=hidden,
        num_heads=8,
        num_kv_heads=2,
        head_dim=config.head_dim,
    )

    x = torch.randn(batch, seq, hidden)
    out, kv_cache = gqa(x)
    assert out.shape == (batch, seq, hidden), f"GQA out shape: {out.shape}"
    assert kv_cache[0].shape[1] == 2, f"KV cache should have 2 KV heads"

    # 测试增量解码
    x1 = x[:, :1, :]
    _, cache = gqa(x1, kv_cache=None)
    x2 = x[:, 1:2, :]
    out2, _ = gqa(x2, kv_cache=cache)
    assert out2.shape == (batch, 1, hidden)
    print("test_grouped_query_attention: PASSED")


def test_flash_attention_simple() -> None:
    """测试分块 flash attention 与 causal attention 结果一致。"""
    batch, n_heads, seq, head_dim = 2, 4, 32, 32
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    v = torch.randn(batch, n_heads, seq, head_dim)

    causal = CausalAttention()
    out_causal = causal(q, k, v)

    flash = FlashAttentionSimple(block_size=8)
    out_flash = flash(q, k, v, causal=True)

    # 由于 online softmax 的数值特性，允许一定的数值误差
    max_diff: float = (out_flash - out_causal).abs().max().item()
    assert max_diff < 1e-4, f"Flash vs causal max diff too large: {max_diff:.6f}"
    print(f"test_flash_attention_simple: PASSED (max_diff={max_diff:.2e})")


def test_swiglu_ffn() -> None:
    """测试 SwiGLU FFN 前向传播。"""
    ffn = SwiGLU_FFN(hidden_size=256, intermediate_size=1024)
    x = torch.randn(2, 16, 256)
    out = ffn(x)
    assert out.shape == x.shape
    print("test_swiglu_ffn: PASSED")


def test_transformer_block() -> None:
    """测试单个 transformer block。"""
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=1,
        num_heads=8,
        num_kv_heads=2,
        intermediate_size=1024,
    )
    block = TransformerBlock(config)
    x = torch.randn(2, 16, 256)
    out, _ = block(x)
    assert out.shape == x.shape, f"Block out shape: {out.shape}"
    print("test_transformer_block: PASSED")


def test_mini_llm_forward() -> None:
    """测试完整模型的前向传播。"""
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        intermediate_size=1024,
        max_seq_len=512,
    )
    model = MiniLLM(config)

    batch, seq = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    # 前向传播
    logits, kv_caches = model(input_ids)
    assert logits.shape == (batch, seq, config.vocab_size), (
        f"Logits shape: {logits.shape}"
    )
    assert len(kv_caches) == config.num_layers

    # 验证 KV cache 的形状
    for cache in kv_caches:
        assert cache is not None
        assert cache[0].shape == (batch, config.num_kv_heads, seq, config.head_dim)

    print(f"test_mini_llm_forward: PASSED (output shape: {logits.shape})")


def test_mini_llm_generate() -> None:
    """测试带 cache 和不带 cache 的生成。"""
    config = MiniLLMConfig(
        vocab_size=100,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=128,
    )
    model = MiniLLM(config)

    # 测试不同的生成策略
    prompt = torch.randint(0, config.vocab_size, (1, 4))

    # 测试贪心搜索
    gen = model.generate(prompt, max_new_tokens=5, temperature=0.0, use_cache=True)
    assert gen.shape[1] == 9, f"Greedy gen shape: {gen.shape}"

    # 测试带 cache
    gen_cached = model.generate(
        prompt, max_new_tokens=3, temperature=1.0, use_cache=True
    )
    assert gen_cached.shape[1] == 7

    # 测试不带 cache
    gen_nocache = model.generate(
        prompt, max_new_tokens=3, temperature=1.0, use_cache=False
    )
    assert gen_nocache.shape[1] == 7

    print("test_mini_llm_generate: PASSED")


if __name__ == "__main__":
    # 运行所有测试
    test_config()
    test_rms_norm()
    test_rope()
    test_scaled_dot_product_attention()
    test_causal_attention()
    test_grouped_query_attention()
    test_flash_attention_simple()
    test_swiglu_ffn()
    test_transformer_block()
    test_mini_llm_forward()
    test_mini_llm_generate()
    print("\nAll transformer model tests passed!")
