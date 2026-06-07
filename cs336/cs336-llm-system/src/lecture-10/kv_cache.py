"""
从头实现 KV Cache。

在自回归解码过程中，我们需要为所有之前的 token 重新计算 K 和 V。
KV Cache 通过缓存这些值来避免重复计算。

核心概念：
  - Prefill（预填充）：处理完整的 prompt，保存所有 token 的 K、V
  - Decode（解码）：每次只处理一个新 token，仅计算新 token 的 K、V，
    然后与缓存的 K、V 拼接

内存分析：
  KV cache 大小 = 2 × num_layers × batch_size × seq_len × num_kv_heads × head_dim × dtype_bytes
  以 LLaMA-2 7B 为例（32 层、32 头、128 维、fp16、batch=1、seq=4096）：
    = 2 × 32 × 1 × 4096 × 32 × 128 × 2 ≈ 2.1 GB
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# KV Cache
# =========================================================================


class KVCache:
    """
    简单的 KV cache，为每一层存储 key 和 value 张量。

    缓存沿着序列维度增长，随着新 token 的生成而扩展。
    """

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
        self.dtype = dtype
        self.device = device

        # 预分配完整的 KV cache：(2 × num_layers, batch, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # 批次中每个序列的当前长度
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为指定层更新 KV cache。

        参数：
            layer_idx: 目标 transformer 层索引（从 0 开始）
            k: 新的 key 张量，形状 (batch, num_kv_heads, new_seq_len, head_dim)
            v: 新的 value 张量，形状与 k 相同
            input_pos: 新 token 的位置索引（用于分页式更新）

        返回：
            (full_k, full_v): 拼接后的缓存 K、V 与新 K、V
        """
        new_len = k.size(2)

        if input_pos is not None:
            # 分页式更新：写入指定位置
            self.k_cache[layer_idx, :, :, input_pos] = k
            self.v_cache[layer_idx, :, :, input_pos] = v
            # 返回最大输入位置之前的完整缓存
            max_pos = int(input_pos.max().item()) + 1
            return (
                self.k_cache[layer_idx, :, :, :max_pos],
                self.v_cache[layer_idx, :, :, :max_pos],
            )
        else:
            # 顺序更新：追加到末尾
            end = self.seq_len + new_len
            self.k_cache[layer_idx, :, :, self.seq_len : end] = k
            self.v_cache[layer_idx, :, :, self.seq_len : end] = v
            return (
                self.k_cache[layer_idx, :, :, :end],
                self.v_cache[layer_idx, :, :, :end],
            )

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取某一层的当前 KV cache。"""
        return (
            self.k_cache[layer_idx, :, :, : self.seq_len],
            self.v_cache[layer_idx, :, :, : self.seq_len],
        )

    def advance(self, num_tokens: int) -> None:
        """将缓存指针向前移动 num_tokens 个位置。"""
        self.seq_len += num_tokens

    def reset(self) -> None:
        """重置缓存。"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0

    def memory_bytes(self) -> int:
        """返回 KV cache 占用的总内存（字节）。"""
        num_elements = self.k_cache.numel() + self.v_cache.numel()
        bytes_per_element = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(self.dtype, 4)
        return num_elements * bytes_per_element

    def active_memory_bytes(self) -> int:
        """返回当前已存储的 KV 对所实际占用的内存。"""
        num_elements = (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.seq_len
            * self.head_dim
        )
        bytes_per_element = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(self.dtype, 4)
        return num_elements * bytes_per_element


# =========================================================================
# 带 KV Cache 的因果注意力
# =========================================================================


def causal_attention_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: KVCache | None = None,
    layer_idx: int = 0,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    可选使用 KV cache 的因果注意力。

    在 prefill 模式（完整 prompt）下，对所有 token 计算注意力。
    在 decode 模式（逐 token）下，使用之前缓存的 K 和 V，
    仅计算新 token 的 K 和 V。

    参数：
        q: Query 张量 (batch, num_heads, seq_len, head_dim)
        k: Key 张量   (batch, num_kv_heads, seq_len, head_dim)
        v: Value 张量 (batch, num_kv_heads, seq_len, head_dim)
        kv_cache: 可选的 KV cache
        layer_idx: 缓存访问的层索引
        mask: 可选的注意力掩码

    返回：
        (output, k_full, v_full): 注意力输出以及所使用的完整 K、V
    """
    if kv_cache is not None:
        # 与缓存的 K、V 拼接
        k_full, v_full = kv_cache.get(layer_idx)
        k_full = torch.cat([k_full, k], dim=2)
        v_full = torch.cat([v_full, v], dim=2)

        # 将新的 K、V 存入缓存
        _ = kv_cache.update(layer_idx, k, v)
    else:
        k_full, v_full = k, v

    d_k = q.size(-1)
    num_queries = q.size(2)
    num_keys = k_full.size(2)

    # 如果是 GQA，扩展 KV 头
    num_q_heads = q.size(1)
    num_kv_heads = k_full.size(1)
    if num_q_heads != num_kv_heads:
        ratio = num_q_heads // num_kv_heads
        k_full = k_full.repeat_interleave(ratio, dim=1)
        v_full = v_full.repeat_interleave(ratio, dim=1)

    # 计算注意力分数
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale

    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_full)
    return output, k_full, v_full


# =========================================================================
# 演示
# =========================================================================


def demo_kv_cache() -> None:
    """演示 KV cache 的 prefill 和 decode 步骤。"""
    print("=" * 70)
    print("KV Cache Demo: Prefill and Decode")
    print("=" * 70)

    batch_size = 1
    num_layers = 2
    num_heads = 4
    num_kv_heads = 4
    head_dim = 64
    hidden_size = num_heads * head_dim
    max_seq_len = 32

    # 模拟投影（实际场景中来自真实模型）
    q_proj = nn.Linear(hidden_size, num_heads * head_dim)
    k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
    v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)

    # 创建 KV cache
    kv_cache = KVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # ---- PREFILL ----
    print("\n--- Prefill Phase ---")
    prompt_len = 8
    prompt = torch.randn(batch_size, prompt_len, hidden_size)

    for layer_idx in range(num_layers):
        x = (
            prompt
            if layer_idx == 0
            else torch.randn(batch_size, prompt_len, hidden_size)
        )

        q = q_proj(x).view(batch_size, prompt_len, num_heads, head_dim).transpose(1, 2)
        k = (
            k_proj(x)
            .view(batch_size, prompt_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            v_proj(x)
            .view(batch_size, prompt_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )

        output, k_full, v_full = causal_attention_with_kv_cache(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        kv_cache.advance(prompt_len)

        print(
            f"  Layer {layer_idx}: Q {list(q.shape)} → output {list(output.shape)}, KV cache shape: {list(k_full.shape)}"
        )

    print(
        f"\n  After prefill - KV cache memory: {kv_cache.active_memory_bytes() / 1024:.1f} KB"
    )

    # ---- DECODE ----
    print("\n--- Decode Phase (generating one token at a time) ---")
    num_new_tokens = 5

    for step in range(num_new_tokens):
        new_token_emb = torch.randn(batch_size, 1, hidden_size)

        for layer_idx in range(num_layers):
            x = new_token_emb
            q = q_proj(x).view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
            k = k_proj(x).view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
            v = v_proj(x).view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

            output, k_full, v_full = causal_attention_with_kv_cache(
                q=q,
                k=k,
                v=v,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )

        kv_cache.advance(1)

        if step == 0:
            mem = kv_cache.active_memory_bytes()
            print(
                f"  Step {step}: New token KV shape: {list(k.shape)}, Full KV: {list(k_full.shape)}"
            )
            print(f"  KV cache memory: {mem / 1024:.1f} KB")

    print(f"\n  After {num_new_tokens} decode steps:")
    print(f"  Final sequence length: {kv_cache.seq_len}")
    print(f"  KV cache memory: {kv_cache.active_memory_bytes() / 1024:.1f} KB")


def calculate_kv_cache_memory() -> None:
    """计算常用模型配置的 KV cache 内存。"""
    print("\n" + "=" * 70)
    print("KV Cache Memory Analysis for Popular LLMs")
    print("=" * 70)

    configs = [
        {
            "name": "LLaMA-2 7B",
            "layers": 32,
            "kv_heads": 32,
            "head_dim": 128,
            "dtype": "fp16",
        },
        {
            "name": "LLaMA-2 13B",
            "layers": 40,
            "kv_heads": 40,
            "head_dim": 128,
            "dtype": "fp16",
        },
        {
            "name": "LLaMA-2 70B",
            "layers": 80,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "fp16",
        },
        {
            "name": "Mistral 7B",
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "fp16",
        },
        {
            "name": "LLaMA-3 8B (GQA)",
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "fp16",
        },
        {
            "name": "LLaMA-3 70B (GQA)",
            "layers": 80,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "fp16",
        },
    ]

    bytes_per = {"fp16": 2, "bf16": 2, "fp32": 4}
    seq_lengths = [1024, 2048, 4096, 8192, 32768]

    print(
        f"\n  KV Cache = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes"
    )
    print(f"  (for batch_size=1, per token)\n")

    for cfg in configs:
        b = bytes_per[cfg["dtype"]]
        print(
            f"  {cfg['name']} ({cfg['layers']}L, {cfg['kv_heads']}KV heads, {cfg['head_dim']}d, {cfg['dtype']}):"
        )
        for sl in seq_lengths:
            mem = 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * sl * b
            print(f"    seq={sl:<6}: {mem / 1e9:.2f} GB ({mem / 1e6:.1f} MB)")
        print()

    print("  关键洞察：KV cache 随序列长度线性增长。")
    print("  在长上下文（32K+）场景下，KV cache 可能占据绝大部分推理内存。")


def main() -> None:
    demo_kv_cache()
    calculate_kv_cache_memory()


if __name__ == "__main__":
    main()
