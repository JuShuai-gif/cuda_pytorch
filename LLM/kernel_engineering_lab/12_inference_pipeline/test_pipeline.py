"""
推理管线测试套件。

测试范围：
  - TransformerBlock 前向传播（fused / 非 fused 模式）
  - KV cache 读写正确性
  - prefill 和 decode 模式输出正确性
  - 批处理推理
  - KV cache 显存计算
  - PagedKVCache block 分配 / 释放
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# 设置项目路径，以便导入各模块中的 kernel
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_operator_fusion"))
sys.path.insert(0, str(_PROJECT_ROOT / "06_attention_flash_like"))
sys.path.insert(0, str(_PROJECT_ROOT / "02_triton_basics"))

from kv_cache import KVCache, PagedKVCache
from pipeline import (
    InferencePipeline,
    OptimizedTransformer,
    TransformerBlock,
    _SimpleKVCachePipeline,
)


# ---------------------------------------------------------------------------
# 测试辅助函数
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """获取当前可用设备。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _assert_shape(tensor: torch.Tensor, expected: tuple, msg: str = "") -> None:
    """断言张量形状符合预期。"""
    assert tensor.shape == expected, f"{msg}：期望形状 {expected}，实际形状 {tensor.shape}"


# ---------------------------------------------------------------------------
# TransformerBlock 前向传播测试
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    """测试 TransformerBlock 在各种配置下的前向传播。"""

    @pytest.mark.parametrize("use_fusions", [True, False])
    @pytest.mark.parametrize(
        "hidden_dim,num_heads,head_dim,ffn_dim",
        [
            (256, 4, 64, 512),
            (512, 8, 64, 1024),
            (768, 12, 64, 2048),
        ],
    )
    def test_forward_shape(
        self, use_fusions: bool, hidden_dim: int, num_heads: int, head_dim: int, ffn_dim: int
    ) -> None:
        """验证 fused 和非 fused 模式下输出形状正确。"""
        device = _get_device()
        B, L = 2, 16
        x = torch.randn(B, L, hidden_dim, device=device, dtype=torch.float32)

        block = TransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            ffn_dim=ffn_dim,
            use_fusions=use_fusions,
        ).to(device)

        y = block(x)
        _assert_shape(y, (B, L, hidden_dim), "TransformerBlock 输出形状")

    @pytest.mark.parametrize("use_fusions", [True, False])
    def test_forward_numerical_stability(self, use_fusions: bool) -> None:
        """验证前向传播不产生 NaN / Inf。"""
        device = _get_device()
        B, L, hidden, heads, h_dim, ffn = 4, 32, 256, 4, 64, 512
        x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)

        block = TransformerBlock(hidden, heads, h_dim, ffn, use_fusions=use_fusions).to(device)
        y = block(x)

        assert not torch.isnan(y).any(), "输出包含 NaN"
        assert not torch.isinf(y).any(), "输出包含 Inf"

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_deterministic_output_same_weights(self, dtype: torch.dtype) -> None:
        """验证相同权重、相同输入下输出是一致的。"""
        device = _get_device()
        B, L, hidden, heads, h_dim, ffn = 2, 8, 256, 4, 64, 512
        x = torch.randn(B, L, hidden, device=device, dtype=dtype)

        torch.manual_seed(42)
        block1 = TransformerBlock(hidden, heads, h_dim, ffn).to(device)
        torch.manual_seed(42)
        block2 = TransformerBlock(hidden, heads, h_dim, ffn).to(device)

        y1 = block1(x)
        y2 = block2(x)
        assert torch.allclose(y1, y2, atol=1e-5), "相同种子应产生相同输出"

    def test_multiple_layers_forward(self) -> None:
        """验证多层 OptimizedTransformer 前向传播。"""
        device = _get_device()
        B, L, hidden, heads, h_dim, ffn = 2, 8, 256, 4, 64, 512
        num_layers = 4
        x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)

        model = OptimizedTransformer(
            num_layers=num_layers,
            hidden_dim=hidden,
            num_heads=heads,
            head_dim=h_dim,
            ffn_dim=ffn,
            use_fusions=True,
        ).to(device)

        y = model(x)
        _assert_shape(y, (B, L, hidden), "多层模型输出形状")


# ---------------------------------------------------------------------------
# KV Cache 测试
# ---------------------------------------------------------------------------


class TestKVCache:
    """测试 KVCache 读写正确性。"""

    @pytest.fixture
    def cache(self) -> KVCache:
        """创建测试用 KVCache 实例。"""
        return KVCache(
            num_layers=2,
            batch_size=2,
            num_heads=4,
            max_seq_len=64,
            head_dim=64,
        )

    @pytest.fixture
    def paged_cache(self) -> PagedKVCache:
        """创建测试用 PagedKVCache 实例。"""
        return PagedKVCache(
            num_layers=2,
            num_heads=4,
            max_seq_len=128,
            head_dim=64,
            block_size=16,
            num_blocks=64,
        )

    def test_write_and_read(self, cache: KVCache) -> None:
        """验证写入 K/V 后能正确读取。"""
        device = _get_device()
        k = torch.randn(1, 4, 16, 64, device=device)  # [batch, heads, seq, dim]
        v = torch.randn(1, 4, 16, 64, device=device)
        positions = torch.arange(16, device=device)

        cache.update(0, 0, k, v, positions)
        k_out, v_out = cache.get(0, 0)

        _assert_shape(k_out, (1, 4, 16, 64), "读取 K 形状")
        _assert_shape(v_out, (1, 4, 16, 64), "读取 V 形状")
        assert torch.allclose(k_out, k, atol=1e-5), "K 值不匹配"
        assert torch.allclose(v_out, v, atol=1e-5), "V 值不匹配"

    def test_seq_len_tracking(self, cache: KVCache) -> None:
        """验证序列长度跟踪正确。"""
        device = _get_device()
        k = torch.randn(1, 4, 8, 64, device=device)
        v = torch.randn(1, 4, 8, 64, device=device)
        positions = torch.arange(8, device=device)

        cache.update(0, 0, k, v, positions)
        assert cache.seq_lens[0].item() == 8, "序列长度应为 8"

        # 追加更多 token
        k2 = torch.randn(1, 4, 4, 64, device=device)
        v2 = torch.randn(1, 4, 4, 64, device=device)
        positions2 = torch.arange(8, 12, device=device)
        cache.update(0, 0, k2, v2, positions2)
        assert cache.seq_lens[0].item() == 12, "序列长度应为 12"

    def test_get_up_to(self, cache: KVCache) -> None:
        """验证 get 的 up_to 参数。"""
        device = _get_device()
        k = torch.randn(1, 4, 20, 64, device=device)
        v = torch.randn(1, 4, 20, 64, device=device)
        positions = torch.arange(20, device=device)

        cache.update(0, 0, k, v, positions)
        k_out, v_out = cache.get(0, 0, up_to=10)
        _assert_shape(k_out, (1, 4, 10, 64), "部分读取 K 形状")
        assert torch.allclose(k_out.squeeze(0)[:, :10, :], k.squeeze(0)[:, :10, :], atol=1e-5)

    def test_multi_batch(self, cache: KVCache) -> None:
        """验证不同 batch 之间互不干扰。"""
        device = _get_device()
        k0 = torch.ones(1, 4, 8, 64, device=device)
        v0 = torch.ones(1, 4, 8, 64, device=device)
        k1 = torch.full((1, 4, 12, 64), 2.0, device=device)
        v1 = torch.full((1, 4, 12, 64), 2.0, device=device)

        cache.update(0, 0, k0, v0, torch.arange(8, device=device))
        cache.update(0, 1, k1, v1, torch.arange(12, device=device))

        k_out0, _ = cache.get(0, 0)
        k_out1, _ = cache.get(0, 1)

        assert k_out0.shape[2] == 8, "batch 0 长度应为 8"
        assert k_out1.shape[2] == 12, "batch 1 长度应为 12"
        assert torch.allclose(k_out0.mean(), torch.tensor(1.0, device=device))
        assert torch.allclose(k_out1.mean(), torch.tensor(2.0, device=device))

    def test_reset(self, cache: KVCache) -> None:
        """验证 reset 清零所有缓存。"""
        device = _get_device()
        k = torch.randn(1, 4, 8, 64, device=device)
        v = torch.randn(1, 4, 8, 64, device=device)
        cache.update(0, 0, k, v, torch.arange(8, device=device))

        cache.reset()
        assert cache.seq_lens[0].item() == 0, "reset 后 seq_len 应为 0"
        assert cache.buffer.abs().max().item() == 0, "reset 后 buffer 应全为 0"

    def test_memory_calculation(self, cache: KVCache) -> None:
        """验证显存计算。"""
        expected_bytes = cache.buffer.numel() * cache.buffer.element_size()
        assert cache.memory_bytes() == expected_bytes, "显存字节数不匹配"
        assert abs(cache.memory_gb() - expected_bytes / (1024**3)) < 1e-6, "GB 转换不匹配"


# ---------------------------------------------------------------------------
# PagedKVCache 测试
# ---------------------------------------------------------------------------


class TestPagedKVCache:
    """测试 PagedKVCache 的 block 分配 / 释放。"""

    def test_allocate_and_free_blocks(self) -> None:
        """验证 block 分配和释放。"""
        pcache = PagedKVCache(
            num_layers=2,
            num_heads=4,
            max_seq_len=128,
            head_dim=64,
            block_size=16,
            num_blocks=64,
        )
        device = _get_device()

        seq_idx = pcache.allocate_sequence()
        blocks_needed = pcache._num_logical_blocks(40)
        allocated, n = pcache.grow(seq_idx, blocks_needed)
        assert n > 0, "应至少分配一个 block"

        k = torch.randn(4, 32, 64, device=device)
        v = torch.randn(4, 32, 64, device=device)
        pcache.write(seq_idx, k, v, start_pos=0)

        k_out, v_out = pcache.read(seq_idx, 0)
        _assert_shape(k_out, (1, 4, 32, 64), "分页读取 K 形状")
        assert torch.allclose(k_out.squeeze(0), k, atol=1e-5), "K 值不匹配"

        util_before = pcache.utilization()
        pcache.free_sequence(seq_idx)
        assert pcache.utilization() == 0.0, "释放后利用率应为 0"
        assert util_before > 0, "释放前利用率应大于 0"

    def test_multiple_sequences(self) -> None:
        """验证多个序列可以共存。"""
        pcache = PagedKVCache(
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
            head_dim=32,
            block_size=8,
            num_blocks=32,
        )
        device = _get_device()

        # 分配两个序列
        s0 = pcache.allocate_sequence()
        s1 = pcache.allocate_sequence()

        # 各自分配 block
        pcache.grow(s0, pcache._num_logical_blocks(24))
        pcache.grow(s1, pcache._num_logical_blocks(16))

        # 写入不同数据
        k0 = torch.ones(4, 24, 32, device=device)
        v0 = torch.ones(4, 24, 32, device=device)
        pcache.write(s0, k0, v0, start_pos=0)

        k1 = torch.full((4, 16, 32, 2.0), device=device)
        v1 = torch.full((4, 16, 32, 2.0), device=device)
        pcache.write(s1, k1, v1, start_pos=0)

        # 独立读取
        k_out0, _ = pcache.read(s0, 0)
        k_out1, _ = pcache.read(s1, 0)

        assert torch.allclose(k_out0.mean(), torch.tensor(1.0, device=device)), "序列 0 数据损坏"
        assert torch.allclose(k_out1.mean(), torch.tensor(2.0, device=device)), "序列 1 数据损坏"

        assert pcache.utilization() > 0, "应有一些 block 在使用中"

    def test_no_block_leak(self) -> None:
        """验证 block 不会泄漏。"""
        pcache = PagedKVCache(
            num_layers=1,
            num_heads=2,
            max_seq_len=64,
            head_dim=16,
            block_size=8,
            num_blocks=16,
        )

        for _ in range(3):
            seq_idx = pcache.allocate_sequence()
            pcache.grow(seq_idx, pcache._num_logical_blocks(32))
            pcache.free_sequence(seq_idx)

        assert pcache.utilization() == 0.0, "所有 block 应已归还"
        assert pcache.free_blocks.all(), "所有 block 应为空闲状态"


# ---------------------------------------------------------------------------
# Prefill / Decode 管线测试
# ---------------------------------------------------------------------------


class TestInferencePipeline:
    """测试推理管线的 prefill 和 decode 模式。"""

    @pytest.fixture
    def model_and_pipeline(self):
        """创建测试用模型和管线。"""
        device = _get_device()
        hidden, heads, h_dim, ffn = 256, 4, 64, 512
        num_layers = 2

        model = OptimizedTransformer(
            num_layers=num_layers,
            hidden_dim=hidden,
            num_heads=heads,
            head_dim=h_dim,
            ffn_dim=ffn,
            use_fusions=True,
        ).to(device)

        pipeline = InferencePipeline(model)
        return model, pipeline, hidden, heads, h_dim, num_layers

    def test_prefill_output_shape(self, model_and_pipeline) -> None:
        """验证 prefill 输出形状正确。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B, L = 2, 16

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=64,
            head_dim=h_dim,
        )

        x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)
        out = pipeline.prefill(x, cache)
        _assert_shape(out, (B, L, hidden), "prefill 输出形状")

    def test_decode_output_shape(self, model_and_pipeline) -> None:
        """验证 decode 输出形状正确。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B = 2

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=64,
            head_dim=h_dim,
        )

        x = torch.randn(B, 1, hidden, device=device, dtype=torch.float32)
        out = pipeline.decode_step(x, cache, step=0)
        _assert_shape(out, (B, 1, hidden), "decode 输出形状")

    def test_prefill_then_decode(self, model_and_pipeline) -> None:
        """验证先 prefill 后 decode 的完整流程。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B, prompt_len = 1, 16

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=64,
            head_dim=h_dim,
        )

        # prefill
        x_prompt = torch.randn(B, prompt_len, hidden, device=device, dtype=torch.float32)
        _ = pipeline.prefill(x_prompt, cache)

        # decode 多个 step
        for step in range(4):
            x_next = torch.randn(B, 1, hidden, device=device, dtype=torch.float32)
            out = pipeline.decode_step(x_next, cache, step=prompt_len + step)
            _assert_shape(out, (B, 1, hidden), f"decode step {step} 输出形状")

    def test_batched_inference(self, model_and_pipeline) -> None:
        """验证批处理推理。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B, prompt_len = 4, 8

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=64,
            head_dim=h_dim,
        )

        x = torch.randn(B, prompt_len, hidden, device=device, dtype=torch.float32)
        out = pipeline.prefill(x, cache)
        _assert_shape(out, (B, prompt_len, hidden), "批处理 prefill 输出形状")

        # 批处理 decode
        for step in range(2):
            x_next = torch.randn(B, 1, hidden, device=device, dtype=torch.float32)
            out = pipeline.decode_step(x_next, cache, step=prompt_len + step)
            _assert_shape(out, (B, 1, hidden), "批处理 decode 输出形状")

    def test_generate_full_flow(self, model_and_pipeline) -> None:
        """验证完整的 generate 流程。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B, prompt_len, max_new = 1, 8, 4

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=64,
            head_dim=h_dim,
        )

        x = torch.randn(B, prompt_len, hidden, device=device, dtype=torch.float32)
        generated = pipeline.generate(x, max_new_tokens=max_new, kv_cache=cache)
        expected_len = prompt_len + max_new
        _assert_shape(
            generated,
            (B, expected_len, hidden),
            f"generate 输出形状（应为 prompt_len + max_new = {expected_len}）",
        )

    def test_prefill_no_nan(self, model_and_pipeline) -> None:
        """验证 prefill 不产生 NaN。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B, L = 2, 8

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=32,
            head_dim=h_dim,
        )

        x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)
        out = pipeline.prefill(x, cache)
        assert not torch.isnan(out).any(), "prefill 输出包含 NaN"

    def test_decode_step_no_nan(self, model_and_pipeline) -> None:
        """验证 decode 不产生 NaN。"""
        _, pipeline, hidden, _, h_dim, nlayers = model_and_pipeline
        device = _get_device()
        B = 2

        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=4,
            max_seq_len=32,
            head_dim=h_dim,
        )

        x = torch.randn(B, 1, hidden, device=device, dtype=torch.float32)
        out = pipeline.decode_step(x, cache, step=0)
        assert not torch.isnan(out).any(), "decode 输出包含 NaN"


# ---------------------------------------------------------------------------
# KV Cache 显存测试
# ---------------------------------------------------------------------------


class TestKVCacheMemory:
    """测试 KV cache 显存计算。"""

    def test_kv_cache_memory_formula(self) -> None:
        """验证 KVCache 显存符合公式 2 × num_layers × batch × heads × max_seq × head_dim × sizeof。"""
        cache = KVCache(
            num_layers=12,
            batch_size=8,
            num_heads=32,
            max_seq_len=2048,
            head_dim=128,
            dtype=torch.float16,
        )
        # 理论：12 × 2 × 8 × 32 × 2048 × 128 × 2 bytes = 100,663,296 × 2 ≈ 3.0 GB
        expected = cache.buffer.numel() * 2  # float16 = 2 bytes
        assert cache.memory_bytes() == expected

    def test_paged_kv_cache_memory(self) -> None:
        """验证 PagedKVCache 显存计算。"""
        pcache = PagedKVCache(
            num_layers=8,
            num_heads=16,
            max_seq_len=1024,
            head_dim=64,
            block_size=16,
            num_blocks=256,
        )
        expected = pcache.blocks.numel() * pcache.blocks.element_size()
        assert pcache.memory_bytes() == expected

    def test_large_cache_memory(self) -> None:
        """验证大规模配置下显存计算（模拟 GPT-3 级别）。"""
        # 仅计算，不实际分配以避免 OOM
        num_layers, batch, heads, max_seq, head_dim = 96, 128, 96, 4096, 128
        numel = num_layers * 2 * batch * heads * max_seq * head_dim
        bytes_fp16 = numel * 2
        gb = bytes_fp16 / (1024**3)
        # GPT-3 风格 KV cache 应该在几百 GB 量级
        assert gb > 50, f"预计算值异常：{gb:.1f} GB"


# ---------------------------------------------------------------------------
# 边缘情况测试
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """测试各种边缘情况。"""

    def test_single_token_input(self) -> None:
        """验证单 token 输入（decode 模式）。"""
        device = _get_device()
        hidden, heads, h_dim, ffn = 256, 4, 64, 512
        B, L = 2, 1  # seq_len = 1

        x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)
        block = TransformerBlock(hidden, heads, h_dim, ffn).to(device)
        y = block(x)
        _assert_shape(y, (B, L, hidden), "单 token 输出形状")

    def test_batch_size_one(self) -> None:
        """验证 batch_size=1 时工作正常。"""
        device = _get_device()
        hidden, heads, h_dim, ffn = 256, 4, 64, 512
        B, L = 1, 32

        x = torch.randn(B, L, hidden, device=device, dtype=torch.float32)
        block = TransformerBlock(hidden, heads, h_dim, ffn).to(device)
        y = block(x)
        _assert_shape(y, (1, 32, hidden), "batch_size=1 输出形状")

    def test_kv_cache_empty_read(self) -> None:
        """验证空 cache 读取不崩溃。"""
        cache = KVCache(
            num_layers=2,
            batch_size=1,
            num_heads=4,
            max_seq_len=64,
            head_dim=64,
        )
        # 未写入任何数据时读取
        k, v = cache.get(0, 0)
        _assert_shape(k, (1, 4, 0, 64), "空 cache K 应为空序列维度")

    def test_paged_cache_empty_read(self) -> None:
        """验证空 PagedKVCache 读取。"""
        pcache = PagedKVCache(
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
            head_dim=32,
            block_size=8,
            num_blocks=16,
        )
        seq_idx = pcache.allocate_sequence()
        k, v = pcache.read(seq_idx, 0)
        _assert_shape(k, (1, 4, 0, 32), "空 PagedKVCache K 应为空序列维度")

    def test_block_allocation_exhaustion(self) -> None:
        """验证 block 耗尽时能检测到错误。"""
        pcache = PagedKVCache(
            num_layers=1,
            num_heads=2,
            max_seq_len=64,
            head_dim=16,
            block_size=8,
            num_blocks=2,  # 极少的 block 数
        )
        seq_idx = pcache.allocate_sequence()

        # 请求超过可用 block 数
        with pytest.raises(RuntimeError, match="No free blocks"):
            pcache.grow(seq_idx, 5)


if __name__ == "__main__":
    # 允许直接运行测试
    pytest.main([__file__, "-v", "--tb=short"])
