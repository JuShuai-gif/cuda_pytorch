"""
简化的 PagedAttention 实现。

PagedAttention 是 vLLM 背后的核心算法。它将 KV cache 管理在固定大小的
"页"（block）中，而非连续内存中，这几乎消除了内存碎片问题，
并支持跨序列的内存共享。

核心概念：
  - Block table（块表）：将逻辑 token 位置映射到物理页索引
  - 每个页存储固定数量 token 的 K/V（block_size）
  - 同一批次中的不同序列可以具有不同长度
  - 在生成过程中按需分配新页

这是一个*简化的*概念实现，使用 Python list 和
torch tensor 来说明核心思想。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# =========================================================================
# 页 / 块 结构
# =========================================================================


@dataclass
class KVCachePage:
    """单层 KV cache 的一个页（block）。

    Attributes:
        data: 形状为 (num_kv_heads, block_size, head_dim) 的 Tensor
    """

    data: torch.Tensor | None = None


@dataclass
class PagedKVCache:
    """
    每层的分页 KV cache。

    物理页存储在一个扁平列表中。每个序列维护一个
    block_table，将逻辑 block 索引映射到物理页索引。

    当序列的 KV cache 需要增长超出已分配的页时，
    会从空闲列表中分配一个新页。

    Attributes:
        num_kv_heads: KV attention head 数量
        head_dim: 每个 attention head 的维度
        block_size: 每页的 token 数量
        pages: 所有已分配页的列表（包括空闲和已使用的）
        free_pages: 空闲（可用）页的索引列表
    """

    num_kv_heads: int
    head_dim: int
    block_size: int
    pages: list[KVCachePage] = field(default_factory=list)
    free_pages: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """预先分配页池。"""
        # 不预先分配；页是惰性分配的

    def allocate_page(self, dtype: torch.dtype, device: torch.device) -> int:
        """分配一个新页并返回其索引。

        优先尝试重用空闲页；否则创建一个新页。

        Args:
            dtype: Tensor 数据类型
            device: Tensor 设备

        Returns:
            物理页索引
        """
        if self.free_pages:
            idx = self.free_pages.pop()
            # 重置已有页的数据
            self.pages[idx].data = torch.zeros(
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            return idx

        new_page = KVCachePage(
            data=torch.zeros(
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.pages.append(new_page)
        return len(self.pages) - 1

    def free_page(self, page_idx: int) -> None:
        """将一个页归还空闲池。

        Args:
            page_idx: 要释放的页的物理索引
        """
        self.free_pages.append(page_idx)

    def write(
        self,
        page_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0,
    ) -> None:
        """将 K 和 V tensor 写入指定偏移处的页中。

        Args:
            page_idx: 物理页索引
            k: 形状为 (num_kv_heads, seq_len, head_dim) 的 Key tensor
            v: 形状为 (num_kv_heads, seq_len, head_dim) 的 Value tensor
            offset: 页内的起始位置
        """
        seq_len = k.size(1)
        page = self.pages[page_idx]
        assert page.data is not None
        page.data[: k.size(0), offset : offset + seq_len, :] = k

    def read_k(
        self, page_idx: int, start: int = 0, end: int | None = None
    ) -> torch.Tensor:
        """从页中读取 K。

        Args:
            page_idx: 物理页索引
            start: 页内的起始位置
            end: 页内的结束位置（默认：block_size）

        Returns:
            形状为 (num_kv_heads, end - start, head_dim) 的 Key tensor
        """
        if end is None:
            end = self.block_size
        page = self.pages[page_idx]
        assert page.data is not None
        return page.data[: self.num_kv_heads, start:end, :]

    def read_v(
        self, page_idx: int, start: int = 0, end: int | None = None
    ) -> torch.Tensor:
        """从页中读取 V。

        Args:
            page_idx: 物理页索引
            start: 页内的起始位置
            end: 页内的结束位置（默认：block_size）

        Returns:
            形状为 (num_kv_heads, end - start, head_dim) 的 Value tensor
        """
        if end is None:
            end = self.block_size
        page = self.pages[page_idx]
        assert page.data is not None
        return page.data[: self.num_kv_heads, start:end, :]


@dataclass
class SequenceMetadata:
    """
    batch 中单个序列的元数据。

    Attributes:
        seq_id: 唯一序列标识符
        prompt_len: 原始 prompt 的长度
        current_len: 目前已处理的总 token 数（prompt + 已生成的）
        block_table: 逻辑 block → 物理页索引的映射
        status: "prefill" 或 "decode"
    """

    seq_id: int
    prompt_len: int
    current_len: int = 0
    block_table: list[int] = field(default_factory=list)
    status: str = "prefill"

    def num_blocks(self) -> int:
        """当前序列所需的 block 数量。"""
        return (self.current_len + self.block_size - 1) // self.block_size

    @property
    def block_size(self) -> int:
        return 16  # 默认值；应与 cache 的 block_size 匹配


# =========================================================================
# PagedAttention 管理器
# =========================================================================


class PagedAttentionManager:
    """
    跨多层和多个序列编排分页 KV cache。

    维护每层的分页 cache 和每个序列的 block table。
    支持同一 batch 内不同长度的序列。
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        # 每层的分页 KV cache
        self.layer_caches: list[PagedKVCache] = [
            PagedKVCache(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
            )
            for _ in range(num_layers)
        ]

        # 按 seq_id 索引的每个序列的元数据
        self.sequences: dict[int, SequenceMetadata] = {}

    def add_sequence(self, seq_id: int, prompt_len: int) -> None:
        """注册一个新序列。

        为 prompt token 分配初始页。

        Args:
            seq_id: 唯一序列标识符
            prompt_len: prompt 的 token 长度
        """
        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        block_table: list[int] = []

        # 跨所有层分配页
        for _ in range(num_blocks_needed):
            for layer_cache in self.layer_caches:
                page_idx = layer_cache.allocate_page(self.dtype, self.device)
            block_table.append(page_idx)

        self.sequences[seq_id] = SequenceMetadata(
            seq_id=seq_id,
            prompt_len=prompt_len,
            current_len=prompt_len,
            block_table=block_table,
            status="decode",  # prefill 之后，我们处于 decode 阶段
        )

    def remove_sequence(self, seq_id: int) -> None:
        """释放属于某个序列的所有页。

        Args:
            seq_id: 要移除的序列标识符
        """
        if seq_id not in self.sequences:
            return
        seq = self.sequences[seq_id]
        for page_idx in seq.block_table:
            for layer_cache in self.layer_caches:
                layer_cache.free_page(page_idx)
        del self.sequences[seq_id]

    def grow_sequence(self, seq_id: int) -> None:
        """为序列分配一个额外的 block（当它需要更多空间时）。

        在 decode 阶段当序列需要新页时调用。

        Args:
            seq_id: 序列标识符
        """
        seq = self.sequences[seq_id]
        page_idx: int = -1
        for layer_cache in self.layer_caches:
            page_idx = layer_cache.allocate_page(self.dtype, self.device)
        seq.block_table.append(page_idx)
        seq.current_len += 1

    def get_kv_for_sequence(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取序列当前长度的完整 K 和 V。

        跨 block table 中的所有页读取并拼接。

        Args:
            layer_idx: Transformer 层索引
            seq_id: 序列标识符

        Returns:
            (k_full, v_full): 每个形状为 (num_kv_heads, current_len, head_dim)
        """
        seq = self.sequences[seq_id]
        layer_cache = self.layer_caches[layer_idx]
        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []

        remaining = seq.current_len
        for block_idx, page_idx in enumerate(seq.block_table):
            take = min(self.block_size, remaining)
            k_chunks.append(layer_cache.read_k(page_idx, start=0, end=take))
            v_chunks.append(layer_cache.read_v(page_idx, start=0, end=take))
            remaining -= take
            if remaining <= 0:
                break

        return torch.cat(k_chunks, dim=1), torch.cat(v_chunks, dim=1)

    def logical_to_physical(self, seq_id: int, pos: int) -> tuple[int, int]:
        """将逻辑 token 位置转换为 (physical_page, offset_in_page)。

        Args:
            seq_id: 序列标识符
            pos: 逻辑 token 位置（从 0 开始）

        Returns:
            (physical_page_idx, offset_within_page)
        """
        seq = self.sequences[seq_id]
        block_idx = pos // self.block_size
        page_idx = seq.block_table[block_idx]
        offset = pos % self.block_size
        return page_idx, offset


# =========================================================================
# Paged Attention 操作（简化版）
# =========================================================================


def paged_attention(
    q: torch.Tensor,
    k_cache: PagedKVCache,
    v_cache: PagedKVCache,
    block_table: list[int],
    seq_len: int,
    block_size: int,
    num_kv_heads: int,
    scale: float | None = None,
) -> torch.Tensor:
    """
    简化的分页 attention：从各页收集 K/V 并计算 attention。

    在实际实现中，这将是一个融合的 GPU kernel。
    这里我们进行收集和拼接，作为概念演示。

    Args:
        q: Query tensor (num_heads, 1, head_dim) — 单 token decode
        k_cache: 某一层的分页 K cache
        v_cache: 某一层的分页 V cache
        block_table: 序列的逻辑 → 物理 block 映射
        seq_len: 当前序列长度
        block_size: 每页的 token 数
        num_kv_heads: KV head 数量
        scale: Attention 缩放因子（默认：1/sqrt(head_dim)）

    Returns:
        形状为 (num_heads, 1, head_dim) 的 Attention 输出
    """
    head_dim = q.size(-1)
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    # 从所有页收集 K 和 V
    k_chunks: list[torch.Tensor] = []
    v_chunks: list[torch.Tensor] = []
    remaining = seq_len

    for page_idx in block_table:
        take = min(block_size, remaining)
        if take <= 0:
            break
        k_chunks.append(k_cache.read_k(page_idx, start=0, end=take))
        v_chunks.append(v_cache.read_v(page_idx, start=0, end=take))
        remaining -= take

    k = torch.cat(k_chunks, dim=1)  # (num_kv_heads, seq_len, head_dim)
    v = torch.cat(v_chunks, dim=1)

    # 如果存在 GQA，需要扩展 KV head
    num_q_heads = q.size(0)
    if num_q_heads != num_kv_heads:
        ratio = num_q_heads // num_kv_heads
        k = k.repeat_interleave(ratio, dim=0)
        v = v.repeat_interleave(ratio, dim=0)

    # 计算 scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (num_heads, 1, seq_len)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # (num_heads, 1, head_dim)
    return output


# =========================================================================
# 演示
# =========================================================================


def demo_paged_attention() -> None:
    """演示分页 attention 的内存管理。"""
    print("=" * 70)
    print("PagedAttention Demo: Block Table and KV Cache Paging")
    print("=" * 70)

    torch.manual_seed(42)

    num_layers = 2
    num_kv_heads = 4
    head_dim = 64
    block_size = 4  # 为演示使用较小的 block size

    manager = PagedAttentionManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )

    # --- 添加不同长度的序列 ---
    print("\n--- Adding Sequences ---")
    sequences = [
        (1, 10),  # seq 1: 10 个 token → 3 个 block
        (2, 5),  # seq 2: 5 个 token  → 2 个 block
        (3, 12),  # seq 3: 12 个 token → 3 个 block
    ]

    for seq_id, prompt_len in sequences:
        manager.add_sequence(seq_id, prompt_len)
        seq = manager.sequences[seq_id]
        print(
            f"  Seq {seq_id}: prompt={prompt_len}, "
            f"blocks={seq.num_blocks()}, "
            f"block_table={seq.block_table}"
        )

    # --- 模拟 decode：增长一个序列 ---
    print("\n--- Growing Sequence 2 (adding 1 token) ---")
    manager.grow_sequence(2)
    seq2 = manager.sequences[2]
    print(
        f"  Seq 2: new current_len={seq2.current_len}, block_table={seq2.block_table}"
    )

    # --- 模拟分页 attention ---
    print("\n--- Paged Attention Computation ---")
    num_heads = 4
    q = torch.randn(num_heads, 1, head_dim)  # 单 token query

    for seq_id in [1, 2, 3]:
        seq = manager.sequences[seq_id]
        output = paged_attention(
            q=q,
            k_cache=manager.layer_caches[0],
            v_cache=manager.layer_caches[0],
            block_table=seq.block_table,
            seq_len=seq.current_len,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
        )
        print(
            f"  Seq {seq_id}: query shape={list(q.shape)}, "
            f"output shape={list(output.shape)}, "
            f"seq_len={seq.current_len}"
        )

    # --- 释放一个序列（内存回收）---
    print("\n--- Removing Sequence 3 ---")
    manager.remove_sequence(3)
    print(f"  Active sequences: {list(manager.sequences.keys())}")
    free_count = len(manager.layer_caches[0].free_pages)
    total_pages = len(manager.layer_caches[0].pages)
    print(f"  Free pages: {free_count}/{total_pages}")

    # --- 展示逻辑到物理的映射 ---
    print("\n--- Logical → Physical Mapping (Seq 2) ---")
    for pos in range(manager.sequences[2].current_len):
        phys_page, offset = manager.logical_to_physical(2, pos)
        print(f"  Logical pos {pos} → Physical page {phys_page}, offset {offset}")

    print("\n  Key insight: PagedAttention enables non-contiguous KV cache storage.")
    print("  This allows flexible memory allocation and sharing across sequences,")
    print("  which is the foundation of vLLM's high-throughput serving.")


def main() -> None:
    demo_paged_attention()


if __name__ == "__main__":
    main()
