"""
KV Cache 实现，用于高效的 autoregressive 推理。

KV cache 存储之前计算好的 Key 和 Value 张量，避免在生成过程中为每个新 token
重新计算它们。这将 attention 计算从每步 O(n^2)（重新计算所有 keys/values）
转变为每步 O(n)（仅为新 token 计算，然后与缓存的 K,V 进行 attention）。

核心概念：
- Prefill：在单次前向传播中处理整个 prompt，缓存所有 K,V。
- Decode：每次处理一个新 token，追加到缓存中。
- 预分配：缓存张量预先分配到 max_seq_len，避免生成过程中动态内存重新分配。
- PagedAttention：将 KV cache 拆分为固定大小的 block，实现高效内存管理，
  减少碎片化（用于 vLLM）。
"""

from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    """
    单个 transformer 层的预分配 Key-Value cache。

    张量一次性分配到 max_seq_len，并通过索引赋值就地写入，
    避免了每步调用 torch.cat() 的开销。

    属性：
        k_cache: [batch, num_kv_heads, max_seq_len, head_dim]。
        v_cache: [batch, num_kv_heads, max_seq_len, head_dim]。
        current_len: 当前存储的 token 数量。
        max_seq_len: 缓存能容纳的最大 token 数量。
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.k_cache: torch.Tensor = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.v_cache: torch.Tensor = torch.zeros(
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.current_len: int = 0
        self.max_seq_len: int = max_seq_len

    def update(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> None:
        """
        将新的 Key 和 Value 张量写入预分配的缓存中。

        在 prefill 阶段，从 current_len 位置开始写入所有 prompt token。
        在 decode 阶段，在 current_len 位置写入一个 token。

        参数：
            new_keys: [batch, num_kv_heads, new_seq_len, head_dim]。
            new_values: [batch, num_kv_heads, new_seq_len, head_dim]。
            input_pos: 可选的位置索引，用于非连续写入
                       （用于 PagedAttention 风格的操作）。
        """
        new_len: int = new_keys.size(2)
        if self.current_len + new_len > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: current_len={self.current_len}, "
                f"new_len={new_len}, max_seq_len={self.max_seq_len}"
            )

        if input_pos is not None:
            # 索引写入（例如用于 PagedAttention 或前缀共享）
            self.k_cache[:, :, input_pos] = new_keys
            self.v_cache[:, :, input_pos] = new_values
        else:
            # 在当前位置连续写入
            self.k_cache[:, :, self.current_len : self.current_len + new_len] = new_keys
            self.v_cache[:, :, self.current_len : self.current_len + new_len] = (
                new_values
            )

        self.current_len += new_len

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取 KV cache 中当前已使用的部分。

        返回：
            (keys, values) 元组，表示当前存储的 token，每个形状为
            [batch, num_kv_heads, current_len, head_dim]。
        """
        return (
            self.k_cache[:, :, : self.current_len],
            self.v_cache[:, :, : self.current_len],
        )

    def reset(self) -> None:
        """清空缓存以开始新的序列（将已使用区域置零）。"""
        self.k_cache[:, :, : self.current_len].zero_()
        self.v_cache[:, :, : self.current_len].zero_()
        self.current_len = 0

    @property
    def seq_len(self) -> int:
        """当前缓存的 token 数量。"""
        return self.current_len

    @property
    def memory_bytes(self) -> int:
        """已分配的内存大小（包括未使用的预分配空间）。"""
        return self.k_cache.element_size() * (
            self.k_cache.numel() + self.v_cache.numel()
        )

    @property
    def active_memory_bytes(self) -> int:
        """实际使用的内存（仅已使用部分）。"""
        if self.current_len == 0:
            return 0
        return (
            self.k_cache.element_size()
            * self.current_len
            * (self.k_cache.size(1) * self.k_cache.size(3))
            * 2
        )


class PagedAttention:
    """
    概念性的 PagedAttention 实现。

    PagedAttention（用于 vLLM）将 KV cache 划分为固定大小的 block
    （page），这些 block 可以在 GPU 内存中非连续分配。这受操作系统
    虚拟内存分页机制的启发。

    优势：
    - 消除了变长序列带来的内存碎片。
    - 实现了序列之间的高效内存共享（例如 beam search）。
    - 接近最优的内存利用率（约 96%+）。

    此类通过注释和伪代码演示核心概念。
    """

    def __init__(self, block_size: int = 16) -> None:
        self.block_size: int = block_size

    def explain(self) -> None:
        """打印 PagedAttention 的说明。"""
        explanation: str = """
PagedAttention: KV Cache 的虚拟内存
=============================================

问题：
  传统的 KV cache 为每个序列连续存储 keys/values。
  序列长度差异很大，导致：
    - 内部碎片（已保留但未使用的内存）。
    - 外部碎片（分配间隙中大量小的空闲空间）。
    - 约 20-40% 的内存浪费。

解决方案：
  将 KV cache 拆分为固定大小的 block（例如，每个 block 16 个 token）。
  每个序列维护一个 block 指针列表（page table）。
  随着序列增长，从空闲 block 池中分配新的 block。

类比操作系统虚拟内存：
  - Page：KV cache block（固定大小）。
  - Page table：每个序列的逻辑->物理 block 映射列表。
  - 物理内存：GPU 上的 block 池。
  - 逻辑内存：序列对其 KV cache 的视图（看起来是连续的）。

优势：
  - 零内部碎片（block 要么完全利用，要么在池中）。
  - 几乎零外部碎片（所有 block 大小相同）。
  - 内存共享：通过复制 page table 而不是数据来 fork 序列。
    多个序列可以共享同一前缀的物理 block。

实现概要：
  ```
  class BlockTable:
      blocks: List[KVBlock]  # 已分配 block 的有序列表

  class KVBlock:
      k: Tensor  # [num_heads, block_size, head_dim]
      v: Tensor  # [num_heads, block_size, head_dim]
      ref_count: int  # 用于共享 block 的引用计数

  class BlockManager:
      free_blocks: List[KVBlock]
      def allocate() -> KVBlock
      def free(block: KVBlock)  # 递减 ref_count，为 0 时释放
  ```

  在 attention 计算期间：
    1. 从序列的所有已分配 block 中收集 K,V。
    2. 正常计算 attention（这些 block 在逻辑上是连续的）。
    3. 收集步骤是额外开销（用计算换取内存效率）。
"""
        print(explanation)


# 快速测试
if __name__ == "__main__":
    batch, n_heads, max_seq, head_dim = 2, 4, 32, 64

    cache = KVCache(batch, n_heads, max_seq, head_dim)

    # Prefill：添加 4 个 token
    keys = torch.randn(batch, n_heads, 4, head_dim)
    values = torch.randn(batch, n_heads, 4, head_dim)
    cache.update(keys, values)
    assert cache.seq_len == 4
    k, v = cache.get()
    assert k.shape == (batch, n_heads, 4, head_dim)
    print(f"Prefill: seq_len={cache.seq_len}, active_mem={cache.active_memory_bytes}B")

    # Decode：添加 1 个 token
    new_k = torch.randn(batch, n_heads, 1, head_dim)
    new_v = torch.randn(batch, n_heads, 1, head_dim)
    cache.update(new_k, new_v)
    assert cache.seq_len == 5
    k, v = cache.get()
    assert k.shape == (batch, n_heads, 5, head_dim)
    print(f"Decode: seq_len={cache.seq_len}, active_mem={cache.active_memory_bytes}B")

    # 重置
    cache.reset()
    assert cache.seq_len == 0
    print(f"Reset: seq_len={cache.seq_len}")

    # PagedAttention 说明
    print()
    PagedAttention().explain()
