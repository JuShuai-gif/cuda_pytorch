"""
简化的推理引擎，编排 KV cache、分页 attention、
以及自回归生成循环。

此模块将 lecture-10 和 lecture-11 的组件整合在一起：
  - KV Cache 管理（连续和分页式）
  - Prefill / decode 阶段处理
  - 采样策略（greedy、temperature、top-k、top-p）
  - 可选的 speculative decoding

推理引擎为文本生成提供了一个统一的接口，
支持可配置的后端策略。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import torch
import torch.nn.functional as F

# 复用 lecture-10 的 KV cache（概念上的；在这里重新实现
# 以便自包含运行）


# =========================================================================
# KV Cache 后端
# =========================================================================


class KVCacheBackend(Enum):
    """可用的 KV cache 管理策略。"""

    CONTIGUOUS = auto()  # 标准连续 KV cache（lecture-10）
    PAGED = auto()  # PagedAttention 风格（lecture-11）


class ContiguousKVCache:
    """简单的连续 KV cache，与 lecture-10 相同。"""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = torch.device(device)

        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_len = 0

    def prefill(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """存储所有 prompt token 的 K 和 V。"""
        add_len = k.size(2)
        self.k_cache[layer_idx, :, :, self.seq_len : self.seq_len + add_len] = k
        self.v_cache[layer_idx, :, :, self.seq_len : self.seq_len + add_len] = v

    def decode_append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """追加一个 token 的 K 和 V（不推进 seq_len）。"""
        self.k_cache[layer_idx, :, :, self.seq_len] = k.squeeze(2)
        self.v_cache[layer_idx, :, :, self.seq_len] = v.squeeze(2)

    def get(
        self, layer_idx: int, length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取某一层的当前 K 和 V。

        Args:
            layer_idx: 层索引
            length: 要返回的位置数（默认：self.seq_len）
        """
        end = length if length is not None else self.seq_len
        return (
            self.k_cache[layer_idx, :, :, :end],
            self.v_cache[layer_idx, :, :, :end],
        )

    def advance(self, n: int = 1) -> None:
        """推进序列指针。"""
        self.seq_len += n

    def reset(self) -> None:
        """重置 cache。"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


class PagedKVCacheBackend:
    """PagedAttention 风格的 KV cache 后端（简化版）。"""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 256,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = torch.device(device)

        # 将所有 block 预分配为一个扁平池
        # 形状：(num_layers, max_blocks, num_kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(
            num_layers,
            max_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.v_pool = torch.zeros_like(self.k_pool)

        # 空闲 block 管理
        self.free_blocks: list[int] = list(range(max_blocks))
        self.active_seq_len: int = 0
        self.block_table: list[int] = []

    def allocate_block(self) -> int:
        """从空闲池中分配一个 block。"""
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")
        return self.free_blocks.pop(0)

    def free_block(self, block_idx: int) -> None:
        """将一个 block 归还空闲池。"""
        self.free_blocks.append(block_idx)

    def ensure_capacity(self, total_tokens: int) -> None:
        """确保为 total_tokens 分配了足够的 block。"""
        needed = (total_tokens + self.block_size - 1) // self.block_size
        while len(self.block_table) < needed:
            blk = self.allocate_block()
            self.block_table.append(blk)

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """在逻辑位置 pos 写入 K 和 V。"""
        seq_len = k.size(2)  # 在 prefill 期间可能 >1
        for offset in range(seq_len):
            logical_pos = pos + offset
            blk = logical_pos // self.block_size
            off = logical_pos % self.block_size
            if blk < len(self.block_table):
                phys = self.block_table[blk]
                self.k_pool[layer_idx, phys, :, off] = k[:, :, offset]
                self.v_pool[layer_idx, phys, :, off] = v[:, :, offset]

    def read(
        self, layer_idx: int, total_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """读取 total_tokens 个逻辑位置的完整 K 和 V。"""
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        remaining = total_tokens

        for blk in range(len(self.block_table)):
            if remaining <= 0:
                break
            phys = self.block_table[blk]
            take = min(self.block_size, remaining)
            k_parts.append(self.k_pool[layer_idx, phys, :, :take])
            v_parts.append(self.v_pool[layer_idx, phys, :, :take])
            remaining -= take

        if not k_parts:
            return (
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim, device=self.device),
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim, device=self.device),
            )
        k_cat = torch.cat(k_parts, dim=1)  # (num_kv_heads, total, head_dim)
        v_cat = torch.cat(v_parts, dim=1)
        # 添加 batch 维度以与 ContiguousKVCache 接口兼容
        return k_cat.unsqueeze(0), v_cat.unsqueeze(0)

    # ---- 用于与 ContiguousKVCache 接口配合的 API 兼容包装器 ----

    @property
    def seq_len(self) -> int:
        """当前序列长度（active_seq_len 的别名）。"""
        return self.active_seq_len

    def prefill(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """存储所有 prompt token 的 K 和 V（兼容 shim）。"""
        self.ensure_capacity(self.active_seq_len + k.size(2))
        self.write(layer_idx, k, v, pos=self.active_seq_len)

    def decode_append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """追加一个 token 的 K 和 V（兼容 shim）。"""
        self.ensure_capacity(self.active_seq_len + 1)
        self.write(layer_idx, k, v, pos=self.active_seq_len)

    def get(
        self, layer_idx: int, length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取某一层的当前 K 和 V（兼容 shim）。"""
        end = length if length is not None else self.active_seq_len
        return self.read(layer_idx, total_tokens=end)

    def advance(self, n: int = 1) -> None:
        """推进序列指针（兼容 shim）。"""
        self.active_seq_len += n

    def reset(self) -> None:
        """重置 cache（兼容 shim）。"""
        self.k_pool.zero_()
        self.v_pool.zero_()
        self.active_seq_len = 0
        self.block_table = []
        self.free_blocks = list(range(self.max_blocks))


# =========================================================================
# 采样策略（lecture-10 的复述）
# =========================================================================


class SamplingStrategy(Enum):
    GREEDY = auto()
    TEMPERATURE = auto()
    TOP_K = auto()
    TOP_P = auto()


@dataclass
class SamplingConfig:
    """token 采样的配置。"""

    strategy: SamplingStrategy = SamplingStrategy.GREEDY
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """从 logits 中采样下一个 token。

        Args:
            logits: (batch, vocab_size) 原始 logits

        Returns:
            (batch,) 形状的 token id tensor
        """
        if self.strategy == SamplingStrategy.GREEDY:
            return logits.argmax(dim=-1)

        logits = logits / max(self.temperature, 1e-9)

        if self.strategy == SamplingStrategy.TOP_K and self.top_k > 0:
            topk_vals, _ = torch.topk(logits, self.top_k, dim=-1)
            min_val = topk_vals[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val, torch.full_like(logits, float("-inf")), logits
            )

        if self.strategy == SamplingStrategy.TOP_P and self.top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > self.top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            mask = mask.scatter(dim=-1, index=sorted_idx, src=mask)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# =========================================================================
# 推理引擎
# =========================================================================


class ModelInterface:
    """推理引擎使用的模型的抽象接口。"""

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: ContiguousKVCache | PagedKVCacheBackend | None = None,
        layer_idx: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """返回 logits 的前向传播。

        Args:
            input_ids: (batch, seq_len) token id
            kv_cache: KV cache 后端
            layer_idx: 当前层（用于每层的 KV cache）
            is_prefill: 这是 prefill（完整 prompt）还是 decode 步骤

        Returns:
            形状为 (batch, seq_len, vocab_size) 的 logits
        """
        raise NotImplementedError


class DummyModel(ModelInterface):
    """一个用于演示的最小模型，仍然使用 KV cache。"""

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_kv_heads: int = 4,
        head_dim: int = 32,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 模拟训练参数的固定权重
        self._embed = torch.randn(vocab_size, hidden_size) * 0.1
        self._q_proj = torch.randn(num_layers, hidden_size, hidden_size) * 0.1
        self._k_proj = (
            torch.randn(num_layers, hidden_size, num_kv_heads * head_dim) * 0.1
        )
        self._v_proj = (
            torch.randn(num_layers, hidden_size, num_kv_heads * head_dim) * 0.1
        )
        self._o_proj = (
            torch.randn(num_layers, num_kv_heads * head_dim, hidden_size) * 0.1
        )
        self._lm_head = torch.randn(hidden_size, vocab_size) * 0.1

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: ContiguousKVCache | PagedKVCacheBackend | None = None,
        layer_idx: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """模拟的前向传播，涉及 KV cache 交互。"""
        batch_size, seq_len = input_ids.shape

        # 嵌入 token
        x = self._embed[input_ids]  # (batch, seq_len, hidden)

        # 逐层处理
        for layer in range(self.num_layers):
            q = x @ self._q_proj[layer]
            k = x @ self._k_proj[layer]
            v = x @ self._v_proj[layer]

            # 为 KV head 重塑 K, V
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )

            if kv_cache is not None:
                if is_prefill:
                    kv_cache.prefill(layer, k, v)
                    # 在 prefill 期间，直接使用完整的 prompt K/V
                    k_full, v_full = k, v
                else:
                    # Decode：追加单个 token，然后读取完整 cache
                    kv_cache.decode_append(layer, k, v)
                    k_full, v_full = kv_cache.get(layer, length=kv_cache.seq_len + 1)
            else:
                k_full, v_full = k, v

            # 简化的 attention：计算 attention 输出并投影回去
            d_k = self.head_dim
            scale = 1.0 / math.sqrt(d_k)

            # 将 KV head 展平回来以与 query 组合
            kv_seq_len = k_full.size(2)
            k_flat = k_full.transpose(1, 2).reshape(
                batch_size, kv_seq_len, self.num_kv_heads * self.head_dim
            )
            v_flat = v_full.transpose(1, 2).reshape(
                batch_size, kv_seq_len, self.num_kv_heads * self.head_dim
            )

            # 简化的全局 attention：pool K, V 然后 attend
            k_pooled = k_flat.mean(dim=1, keepdim=True)  # (batch, 1, kv_dim)
            v_pooled = v_flat.mean(dim=1, keepdim=True)  # (batch, 1, kv_dim)
            attn_out = v_pooled.expand(-1, seq_len, -1)  # (batch, seq_len, kv_dim)

            # Output 投影
            o = attn_out @ self._o_proj[layer]  # (batch, seq_len, hidden)
            x = x + o

        logits = x @ self._lm_head
        return logits


@dataclass
class GenerationOutput:
    """生成请求的结果。"""

    prompt_ids: list[int]
    generated_ids: list[int]
    num_tokens_generated: int
    finish_reason: str  # "max_length"、"eos" 等


class InferenceEngine:
    """
    统一的推理引擎，结合了 KV cache、采样和
    生成循环策略。

    支持：
      - 连续或分页式 KV cache 后端
      - Greedy、temperature、top-k、top-p 采样
      - Prefill / decode 阶段管理
    """

    def __init__(
        self,
        model: ModelInterface,
        max_seq_len: int = 512,
        batch_size: int = 1,
        kv_backend: KVCacheBackend = KVCacheBackend.CONTIGUOUS,
        block_size: int = 16,
        sampling: SamplingConfig | None = None,
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.kv_backend_type = kv_backend
        self.sampling = sampling or SamplingConfig()

        # 构建合适的 KV cache 后端
        if kv_backend == KVCacheBackend.PAGED:
            self.kv_cache: ContiguousKVCache | PagedKVCacheBackend = (
                PagedKVCacheBackend(
                    num_layers=model.num_layers,
                    num_kv_heads=model.num_kv_heads,
                    head_dim=model.head_dim,
                    block_size=block_size,
                    max_blocks=(max_seq_len // block_size) + 10,
                )
            )
        else:
            self.kv_cache = ContiguousKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_layers=model.num_layers,
                num_kv_heads=model.num_kv_heads,
                head_dim=model.head_dim,
            )

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
        verbose: bool = False,
    ) -> GenerationOutput:
        """给定 prompt 生成 token。

        Args:
            prompt_ids: 输入 token id
            max_new_tokens: 最大生成 token 数
            eos_token_id: 当生成此 token 时停止生成
            verbose: 打印进度

        Returns:
            包含 prompt 和生成的 token 的 GenerationOutput
        """
        current_ids = torch.tensor([prompt_ids], dtype=torch.long)

        # ---- Prefill：处理完整 prompt ----
        logits = self.model.forward(
            current_ids,
            kv_cache=self.kv_cache,
            is_prefill=True,
        )
        if isinstance(self.kv_cache, ContiguousKVCache):
            self.kv_cache.advance(len(prompt_ids))
        elif isinstance(self.kv_cache, PagedKVCacheBackend):
            self.kv_cache.advance(len(prompt_ids))

        # 采样第一个 token
        first_token = self.sampling.sample(logits[:, -1, :])
        generated: list[int] = [first_token.item()]

        if verbose:
            print(f"  Prefill done. First token: {generated[0]}")

        # ---- Decode：每次生成一个 token ----
        for step in range(1, max_new_tokens):
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long)

            logits = self.model.forward(
                last_token,
                kv_cache=self.kv_cache,
                is_prefill=False,
            )
            if isinstance(self.kv_cache, ContiguousKVCache):
                self.kv_cache.advance(1)
            elif isinstance(self.kv_cache, PagedKVCacheBackend):
                self.kv_cache.advance(1)
            next_token = self.sampling.sample(logits[:, -1, :])
            token_id = next_token.item()

            generated.append(token_id)

            if verbose and (step < 5 or step % 10 == 0):
                print(f"  Step {step}: token={token_id}")

            # 检查停止条件
            if eos_token_id is not None and token_id == eos_token_id:
                break

        finish_reason = "max_length"
        if eos_token_id is not None and generated[-1] == eos_token_id:
            finish_reason = "eos"

        return GenerationOutput(
            prompt_ids=prompt_ids,
            generated_ids=generated,
            num_tokens_generated=len(generated),
            finish_reason=finish_reason,
        )

    def reset(self) -> None:
        """重置引擎状态（KV cache 等）。"""
        self.kv_cache.reset()

    def info(self) -> dict:
        """返回引擎配置信息。"""
        return {
            "kv_backend": self.kv_backend_type.name,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "sampling_strategy": self.sampling.strategy.name,
        }


# =========================================================================
# 演示
# =========================================================================


def demo_contiguous_vs_paged() -> None:
    """比较连续和分页式 KV cache 后端。"""
    print("=" * 70)
    print("Inference Engine Demo: Contiguous vs Paged KV Cache")
    print("=" * 70)

    model = DummyModel(
        vocab_size=50, hidden_size=64, num_layers=2, num_kv_heads=4, head_dim=16
    )

    prompt = [1, 5, 10, 3, 7]

    # ---- 连续后端 ----
    print("\n--- Contiguous KV Cache ---")
    engine_contig = InferenceEngine(
        model=model,
        max_seq_len=128,
        kv_backend=KVCacheBackend.CONTIGUOUS,
        sampling=SamplingConfig(strategy=SamplingStrategy.GREEDY),
    )

    output1 = engine_contig.generate(prompt, max_new_tokens=8, verbose=True)
    print(f"\n  Result: prompt={output1.prompt_ids}")
    print(f"  Generated: {output1.generated_ids}")
    print(f"  Finish: {output1.finish_reason}")

    # ---- 分页后端 ----
    print("\n--- Paged KV Cache ---")
    engine_paged = InferenceEngine(
        model=model,
        max_seq_len=128,
        kv_backend=KVCacheBackend.PAGED,
        block_size=8,
        sampling=SamplingConfig(strategy=SamplingStrategy.GREEDY),
    )

    output2 = engine_paged.generate(prompt, max_new_tokens=8, verbose=True)
    print(f"\n  Result: prompt={output2.prompt_ids}")
    print(f"  Generated: {output2.generated_ids}")
    print(f"  Finish: {output2.finish_reason}")

    print(f"\n  Contiguous engine info: {engine_contig.info()}")
    print(f"  Paged engine info:      {engine_paged.info()}")


def demo_sampling_strategies() -> None:
    """通过引擎演示不同的采样策略。"""
    print("\n" + "=" * 70)
    print("Inference Engine: Sampling Strategies")
    print("=" * 70)

    model = DummyModel(
        vocab_size=30, hidden_size=32, num_layers=1, num_kv_heads=2, head_dim=16
    )

    prompt = [1, 2, 3]
    strategies = [
        ("Greedy", SamplingConfig(strategy=SamplingStrategy.GREEDY)),
        (
            "Temperature=2.0",
            SamplingConfig(strategy=SamplingStrategy.TEMPERATURE, temperature=2.0),
        ),
        (
            "Top-K=5",
            SamplingConfig(strategy=SamplingStrategy.TOP_K, temperature=1.0, top_k=5),
        ),
        (
            "Top-P=0.9",
            SamplingConfig(strategy=SamplingStrategy.TOP_P, temperature=1.0, top_p=0.9),
        ),
    ]

    torch.manual_seed(42)

    for name, cfg in strategies:
        engine = InferenceEngine(
            model=model,
            max_seq_len=64,
            kv_backend=KVCacheBackend.CONTIGUOUS,
            sampling=cfg,
        )
        output = engine.generate(prompt, max_new_tokens=6)
        print(f"\n  {name}:")
        print(f"    Prompt:     {output.prompt_ids}")
        print(f"    Generated:  {output.generated_ids}")
        print(f"    Tokens:     {output.num_tokens_generated}")
        engine.reset()


def demo_with_eos() -> None:
    """演示在 EOS token 上提前停止的生成。"""
    print("\n" + "=" * 70)
    print("Inference Engine: Early Stopping (EOS)")
    print("=" * 70)

    model = DummyModel(
        vocab_size=30, hidden_size=32, num_layers=1, num_kv_heads=2, head_dim=16
    )

    engine = InferenceEngine(
        model=model,
        max_seq_len=64,
        sampling=SamplingConfig(strategy=SamplingStrategy.GREEDY),
    )

    prompt = [5, 10]

    # 不使用 EOS（生成到 max_new_tokens）
    output_long = engine.generate(prompt, max_new_tokens=15, eos_token_id=None)
    print(f"\n  Without EOS: {output_long.generated_ids}")
    print(
        f"    Tokens: {output_long.num_tokens_generated}, "
        f"reason: {output_long.finish_reason}"
    )

    # 使用 EOS token id 0（如果采样到 0 会提前停止）
    engine.reset()
    output_eos = engine.generate(prompt, max_new_tokens=15, eos_token_id=0)
    print(f"\n  With EOS=0:  {output_eos.generated_ids}")
    print(
        f"    Tokens: {output_eos.num_tokens_generated}, "
        f"reason: {output_eos.finish_reason}"
    )

    print("\n  Key insight: The inference engine encapsulates the complexity of")
    print("  KV cache management, prefill/decode phases, and sampling into a")
    print("  clean API. Users can swap backends and strategies without changing")
    print("  the generation code.")


def main() -> None:
    demo_contiguous_vs_paged()
    demo_sampling_strategies()
    demo_with_eos()


if __name__ == "__main__":
    main()
