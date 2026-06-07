"""
Paged KV cache memory management with block-based allocation.

Implements virtual-memory-style paging for KV cache tensors,
enabling near-zero internal fragmentation and efficient memory sharing
across sequences via copy-on-write.

Core components:
  - KVBlock: Fixed-size block (default 16 tokens) holding K/V tensors
  - BlockTable: Logical-to-physical block mapping (analogous to OS page tables)
  - BlockAllocator: Free-list based memory allocator with malloc/free semantics
  - KVCacheManager: Multi-layer orchestration with FP8/INT8 quantization

Reference:
  Kwon et al., "Efficient Memory Management for Large Language Model
  Serving with PagedAttention", SOSP 2023.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch


# ==============================================================================
#  KV cache quantization
# ==============================================================================


class KVCacheQuantization(Enum):
    """Quantization modes for KV cache tensors."""

    NONE = auto()  # Native dtype (FP16/BF16/FP32)
    FP8_E4M3 = auto()  # FP8 E4M3 format (halves decode memory bandwidth)
    INT8 = auto()  # INT8 quantization with per-block scaling


# ==============================================================================
#  KVBlock - fixed-size KV cache page
# ==============================================================================


@dataclass
class KVBlock:
    """A single page (block) of KV cache for one transformer layer.

    Each block stores B=block_size tokens worth of keys and values.
    Blocks are the unit of allocation and deallocation.

    Attributes:
        block_id: Unique physical block index.
        k: Key tensor of shape (n_kv_heads, block_size, head_dim).
        v: Value tensor of shape (n_kv_heads, block_size, head_dim).
        valid_tokens: Number of valid tokens in this block (0..block_size).
        ref_count: Reference count for copy-on-write sharing.
        layer_idx: Transformer layer this block belongs to.
    """

    block_id: int = -1
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    valid_tokens: int = 0
    ref_count: int = 1
    layer_idx: int = 0


# ==============================================================================
#  BlockTable - logical-to-physical address translation
# ==============================================================================


class BlockTable:
    """Maps logical block positions to physical block IDs.

    Analogous to an OS page table. Each sequence has its own block table.
    Supports copy-on-write: when forking a sequence, the block table
    shares physical blocks via reference counting.

    Attributes:
        seq_id: Sequence this block table belongs to.
        physical_blocks: List mapping logical_idx -> KVBlock objects.
        block_size: Number of tokens per block.
        total_tokens: Total number of tokens stored across all blocks.
    """

    def __init__(
        self,
        seq_id: int,
        block_size: int = 16,
    ) -> None:
        self.seq_id = seq_id
        self.block_size = block_size
        self._blocks: list[KVBlock] = []
        self._total_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        """Number of allocated blocks."""
        return len(self._blocks)

    @property
    def total_tokens(self) -> int:
        """Total tokens stored in this block table."""
        return self._total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        self._total_tokens = value

    def get_physical_id(self, logical_block: int) -> int:
        """Get physical block ID for a logical block index.

        Args:
            logical_block: Logical block index within the sequence.

        Returns:
            Physical block ID.

        Raises:
            IndexError: If logical_block is out of range.
        """
        if logical_block < 0 or logical_block >= len(self._blocks):
            raise IndexError(
                f"Logical block {logical_block} out of range [0, {len(self._blocks)})"
            )
        return self._blocks[logical_block].block_id

    def get_offset(self, token_pos: int) -> tuple[int, int]:
        """Translate logical token position to (physical_block_id, offset_within_block).

        Args:
            token_pos: Logical token position (0-indexed).

        Returns:
            Tuple of (physical_block_id, offset_in_block).
        """
        block_idx = token_pos // self.block_size
        offset = token_pos % self.block_size
        return self.get_physical_id(block_idx), offset

    def add_block(self, block: KVBlock) -> None:
        """Append a physical block to this table.

        Args:
            block: The KVBlock to add.
        """
        block.ref_count += 1
        self._blocks.append(block)

    def remove_last_block(self) -> Optional[KVBlock]:
        """Remove and return the last block (decrements ref_count).

        Returns:
            The removed block, or None if table is empty.
        """
        if not self._blocks:
            return None
        block = self._blocks.pop()
        block.ref_count = max(0, block.ref_count - 1)
        return block

    def get_block(self, logical_idx: int) -> KVBlock:
        """Get the KVBlock at a logical index.

        Args:
            logical_idx: Logical block index.

        Returns:
            The KVBlock object.
        """
        return self._blocks[logical_idx]

    def to_tensor(self, max_blocks: int) -> torch.Tensor:
        """Convert block table to a padded integer tensor.

        Args:
            max_blocks: Maximum number of blocks to pad to.

        Returns:
            Tensor of shape (max_blocks,) with physical block IDs and -1 padding.
        """
        table = torch.full((max_blocks,), -1, dtype=torch.int32)
        for i, blk in enumerate(self._blocks):
            if i < max_blocks:
                table[i] = blk.block_id
        return table

    def fork(self, new_seq_id: int) -> "BlockTable":
        """Create a copy-on-write fork of this block table.

        The new table shares all physical blocks. A write to a shared
        block triggers a physical copy (handled by KVCacheManager).

        Args:
            new_seq_id: Sequence ID for the new table.

        Returns:
            A new BlockTable sharing the same physical blocks.
        """
        new_table = BlockTable(seq_id=new_seq_id, block_size=self.block_size)
        new_table._blocks = list(self._blocks)
        new_table._total_tokens = self._total_tokens
        for blk in self._blocks:
            blk.ref_count += 1
        return new_table

    def __repr__(self) -> str:
        return (
            f"BlockTable(seq={self.seq_id}, blocks={self.num_blocks}, "
            f"tokens={self._total_tokens})"
        )


# ==============================================================================
#  BlockAllocator - free-list memory allocator
# ==============================================================================


class BlockAllocator:
    """Manages a pool of KV blocks with malloc/free semantics.

    Uses a free list for O(1) allocation and deallocation.
    Supports dynamic pool growth when the free list is exhausted.

    Attributes:
        block_size: Tokens per block (default 16, matching vLLM).
        n_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        dtype: Tensor data type.
        device: Device for tensor storage.
        initial_blocks: Number of blocks to pre-allocate.
    """

    def __init__(
        self,
        block_size: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
        initial_blocks: int = 256,
    ) -> None:
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        self._total_blocks = initial_blocks
        self._next_block_id = 0

        # Free list of block IDs available for allocation
        self._free_list: list[int] = list(range(initial_blocks))
        # All allocated blocks indexed by block_id
        self._all_blocks: dict[int, KVBlock] = {}
        # Per-layer pre-allocated tensor pool
        self._k_pool: torch.Tensor = torch.zeros(
            initial_blocks,
            n_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._v_pool: torch.Tensor = torch.zeros_like(self._k_pool)

        self._lock = threading.Lock()

    @property
    def free_count(self) -> int:
        """Number of currently free blocks."""
        return len(self._free_list)

    @property
    def allocated_count(self) -> int:
        """Number of currently allocated blocks."""
        return len(self._all_blocks)

    @property
    def total_capacity(self) -> int:
        """Total number of blocks in the pool."""
        return self._total_blocks

    def allocate(self, layer_idx: int = 0) -> KVBlock:
        """Allocate a new block from the free list.

        Args:
            layer_idx: The transformer layer this block is for.

        Returns:
            A new KVBlock with zeroed K and V tensors.

        Raises:
            RuntimeError: If no free blocks are available and growth is disabled.
        """
        with self._lock:
            if not self._free_list:
                self._grow(64)

            block_id = self._free_list.pop()

            blk = KVBlock(
                block_id=block_id,
                k=self._k_pool[block_id],
                v=self._v_pool[block_id],
                valid_tokens=0,
                ref_count=1,
                layer_idx=layer_idx,
            )
            self._all_blocks[block_id] = blk
            return blk

    def free(self, block: KVBlock) -> bool:
        """Return a block to the free list.

        The block is only actually freed when ref_count reaches 0.

        Args:
            block: The KVBlock to free.

        Returns:
            True if the block was returned to the free list.
        """
        with self._lock:
            block.ref_count = max(0, block.ref_count - 1)
            if block.ref_count > 0:
                return False

            self._free_list.append(block.block_id)
            if block.block_id in self._all_blocks:
                del self._all_blocks[block.block_id]

            # Zero the block for safety
            self._k_pool[block.block_id].zero_()
            self._v_pool[block.block_id].zero_()

            return True

    def write_kv(
        self,
        block_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0,
    ) -> None:
        """Write K and V tensors to a specific block at a given offset.

        Args:
            block_id: Physical block ID to write to.
            k: Key tensor of shape (n_kv_heads, n_tokens, head_dim).
            v: Value tensor of shape (n_kv_heads, n_tokens, head_dim).
            offset: Starting position within the block.
        """
        n_tokens = k.size(1)
        self._k_pool[block_id, :, offset : offset + n_tokens, :] = k
        self._v_pool[block_id, :, offset : offset + n_tokens, :] = v

    def read_k(
        self, block_id: int, start: int = 0, end: int | None = None
    ) -> torch.Tensor:
        """Read K tensor from a block.

        Args:
            block_id: Physical block ID.
            start: Start position within the block.
            end: End position (default: block_size).

        Returns:
            K tensor of shape (n_kv_heads, end-start, head_dim).
        """
        if end is None:
            end = self.block_size
        return self._k_pool[block_id, :, start:end, :]

    def read_v(
        self, block_id: int, start: int = 0, end: int | None = None
    ) -> torch.Tensor:
        """Read V tensor from a block.

        Args:
            block_id: Physical block ID.
            start: Start position within the block.
            end: End position (default: block_size).

        Returns:
            V tensor of shape (n_kv_heads, end-start, head_dim).
        """
        if end is None:
            end = self.block_size
        return self._v_pool[block_id, :, start:end, :]

    def get_k_pool(self) -> torch.Tensor:
        """Get the full K tensor pool."""
        return self._k_pool

    def get_v_pool(self) -> torch.Tensor:
        """Get the full V tensor pool."""
        return self._v_pool

    def memory_usage_bytes(self) -> int:
        """Total GPU memory used by the block pool in bytes."""
        k_bytes = self._k_pool.numel() * self._k_pool.element_size()
        v_bytes = self._v_pool.numel() * self._v_pool.element_size()
        return k_bytes + v_bytes

    def _grow(self, additional_blocks: int) -> None:
        """Expand the block pool by additional_blocks.

        Args:
            additional_blocks: Number of new blocks to add.
        """
        old_capacity = self._total_blocks
        self._total_blocks += additional_blocks

        new_k = torch.zeros(
            additional_blocks,
            self.n_kv_heads,
            self.block_size,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        new_v = torch.zeros_like(new_k)

        self._k_pool = torch.cat([self._k_pool, new_k], dim=0)
        self._v_pool = torch.cat([self._v_pool, new_v], dim=0)

        new_ids = list(range(old_capacity, self._total_blocks))
        self._free_list.extend(new_ids)


# ==============================================================================
#  KVCacheManager - multi-layer orchestration
# ==============================================================================


class KVCacheManager:
    """Manages paged KV cache across all transformer layers.

    This is the top-level interface for KV cache management. It maintains
    per-layer block allocators and per-sequence block tables, supporting:
      - Paged allocation with O(1) malloc/free
      - Copy-on-write for prefix sharing
      - FP8/INT8 quantization of cached KV tensors
      - Memory usage tracking and defragmentation statistics

    Args:
        num_layers: Number of transformer layers.
        n_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        block_size: Tokens per KV cache block (default 16).
        max_blocks: Maximum number of blocks in the pool.
        dtype: Tensor data type.
        device: Device for tensor storage.
        kv_quantization: KV cache quantization mode.
    """

    def __init__(
        self,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 2048,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
        kv_quantization: KVCacheQuantization = KVCacheQuantization.NONE,
    ) -> None:
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.quantization = kv_quantization

        # One allocator per layer (blocks are layer-specific)
        self._allocators: list[BlockAllocator] = [
            BlockAllocator(
                block_size=block_size,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                initial_blocks=max_blocks,
            )
            for _ in range(num_layers)
        ]

        # Per-sequence block tables
        self._block_tables: dict[int, BlockTable] = {}
        self._next_seq_id = 0

        self._lock = threading.Lock()

    # ---- Sequence management ----

    def register_sequence(self, prompt_len: int, seq_id: int | None = None) -> int:
        """Register a new sequence and allocate blocks for its prompt.

        Args:
            prompt_len: Number of tokens in the prompt.
            seq_id: Desired sequence ID (auto-assigned if None).

        Returns:
            The sequence ID.

        Raises:
            RuntimeError: If insufficient free blocks.
        """
        with self._lock:
            if seq_id is None:
                seq_id = self._next_seq_id
                self._next_seq_id += 1

            num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
            block_table = BlockTable(seq_id=seq_id, block_size=self.block_size)

            for layer_idx in range(self.num_layers):
                total_needed = num_blocks_needed
                if self._allocators[layer_idx].free_count < total_needed:
                    raise RuntimeError(
                        f"Layer {layer_idx}: need {total_needed} blocks, "
                        f"have {self._allocators[layer_idx].free_count}"
                    )
                for _ in range(num_blocks_needed):
                    blk = self._allocators[layer_idx].allocate(layer_idx=layer_idx)
                    if layer_idx == 0:
                        # Attach to block table only once using layer-0 blocks
                        block_table.add_block(blk)

            block_table.total_tokens = prompt_len
            self._block_tables[seq_id] = block_table
            return seq_id

    def remove_sequence(self, seq_id: int) -> None:
        """Free all blocks belonging to a sequence.

        Args:
            seq_id: Sequence to remove.
        """
        with self._lock:
            if seq_id not in self._block_tables:
                return

            block_table = self._block_tables.pop(seq_id)
            for layer_idx in range(self.num_layers):
                for logical_idx in range(block_table.num_blocks):
                    blk = block_table.get_block(logical_idx)
                    self._allocators[layer_idx].free(blk)

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int | None = None) -> int:
        """Fork a sequence via copy-on-write for prefix sharing.

        Args:
            parent_seq_id: Source sequence to fork.
            child_seq_id: Desired child ID (auto-assigned if None).

        Returns:
            The child sequence ID.
        """
        with self._lock:
            parent_table = self._block_tables.get(parent_seq_id)
            if parent_table is None:
                raise ValueError(f"Parent sequence {parent_seq_id} not found")

            if child_seq_id is None:
                child_seq_id = self._next_seq_id
                self._next_seq_id += 1

            child_table = parent_table.fork(new_seq_id=child_seq_id)
            self._block_tables[child_seq_id] = child_table
            return child_seq_id

    # ---- KV cache write/read ----

    def write_prefill(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write prefill KV tensors for a sequence at a given layer.

        Distributes the tokens across blocks in the block table.

        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            k: Key tensor of shape (n_kv_heads, n_tokens, head_dim).
            v: Value tensor of shape (n_kv_heads, n_tokens, head_dim).
        """
        block_table = self._block_tables.get(seq_id)
        if block_table is None:
            raise ValueError(f"Sequence {seq_id} not registered")

        n_tokens = k.size(1)
        tokens_remaining = n_tokens
        allocator = self._allocators[layer_idx]
        block_idx = 0

        while tokens_remaining > 0:
            if block_idx >= block_table.num_blocks:
                # Allocate new block
                blk = allocator.allocate(layer_idx=layer_idx)
                block_table.add_block(blk)

            phys_id = block_table.get_physical_id(block_idx)
            offset = 0  # For prefill, always start at beginning of block
            take = min(self.block_size - offset, tokens_remaining)

            chunk_k = k[:, :take, :]
            chunk_v = v[:, :take, :]
            allocator.write_kv(phys_id, chunk_k, chunk_v, offset=offset)

            tokens_remaining -= take
            block_idx += 1

    def write_decode(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write a single decode token's KV for a sequence.

        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            k: Single-token key tensor of shape (n_kv_heads, 1, head_dim).
            v: Single-token value tensor of shape (n_kv_heads, 1, head_dim).
        """
        block_table = self._block_tables.get(seq_id)
        if block_table is None:
            raise ValueError(f"Sequence {seq_id} not registered")

        current_len = block_table.total_tokens
        logical_block = current_len // self.block_size
        offset = current_len % self.block_size

        allocator = self._allocators[layer_idx]

        if logical_block >= block_table.num_blocks:
            blk = allocator.allocate(layer_idx=layer_idx)
            block_table.add_block(blk)

        phys_id = block_table.get_physical_id(logical_block)
        allocator.write_kv(phys_id, k, v, offset=offset)

        block_table.total_tokens += 1

    def get_kv_for_sequence(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read full K and V for a sequence at a given layer.

        Gathers from all blocks in the sequence's block table and
        concatenates into contiguous tensors.

        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (k, v) tensors, each of shape
            (n_kv_heads, total_tokens, head_dim).
        """
        block_table = self._block_tables.get(seq_id)
        if block_table is None:
            raise ValueError(f"Sequence {seq_id} not registered")

        allocator = self._allocators[layer_idx]
        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []

        remaining = block_table.total_tokens
        for logical_idx in range(block_table.num_blocks):
            if remaining <= 0:
                break
            phys_id = block_table.get_physical_id(logical_idx)
            take = min(self.block_size, remaining)
            k_chunks.append(allocator.read_k(phys_id, 0, take))
            v_chunks.append(allocator.read_v(phys_id, 0, take))
            remaining -= take

        if not k_chunks:
            empty_k = torch.zeros(
                self.n_kv_heads,
                0,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            empty_v = torch.zeros_like(empty_k)
            return empty_k, empty_v

        return torch.cat(k_chunks, dim=1), torch.cat(v_chunks, dim=1)

    def get_block_table_tensor(self, seq_id: int) -> torch.Tensor:
        """Get the block table for a sequence as a padded tensor.

        Args:
            seq_id: Sequence identifier.

        Returns:
            Tensor of shape (max_blocks,) with physical block IDs.
        """
        block_table = self._block_tables.get(seq_id)
        if block_table is None:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        return block_table.to_tensor(block_table.num_blocks)

    def get_k_pool(self, layer_idx: int) -> torch.Tensor:
        """Get the full K tensor pool for a layer."""
        return self._allocators[layer_idx].get_k_pool()

    def get_v_pool(self, layer_idx: int) -> torch.Tensor:
        """Get the full V tensor pool for a layer."""
        return self._allocators[layer_idx].get_v_pool()

    def get_seq_len(self, seq_id: int) -> int:
        """Get current sequence length (tokens stored in cache)."""
        bt = self._block_tables.get(seq_id)
        return bt.total_tokens if bt else 0

    # ---- Memory tracking ----

    def total_memory_mb(self) -> float:
        """Total GPU memory used by all KV cache blocks in MB."""
        total_bytes = sum(alloc.memory_usage_bytes() for alloc in self._allocators)
        return total_bytes / (1024**2)

    def fragmentation_ratio(self) -> float:
        """Ratio of allocated-but-unused tokens to total allocated tokens.

        Internal fragmentation occurs when blocks are partially filled
        (the last block of a sequence is rarely 100% full).

        Returns:
            Fragmentation ratio between 0.0 and 1.0.
        """
        total_allocated = 0
        total_used = 0
        for seq_id, bt in self._block_tables.items():
            for blk_idx in range(bt.num_blocks):
                blk = bt.get_block(blk_idx)
                total_allocated += self.block_size
                total_used += blk.valid_tokens

        if total_allocated == 0:
            return 0.0
        return 1.0 - (total_used / total_allocated)

    def active_sequences(self) -> list[int]:
        """Return list of active sequence IDs."""
        return list(self._block_tables.keys())


# ==============================================================================
#  Cache defragmentation utilities
# ==============================================================================


def compute_block_utilization(
    manager: KVCacheManager,
) -> float:
    """Compute KV cache block utilization across all layers.

    Args:
        manager: The KVCacheManager instance.

    Returns:
        Utilization ratio (allocated / total capacity).
    """
    total_allocated = sum(alloc.allocated_count for alloc in manager._allocators)
    total_capacity = sum(alloc.total_capacity for alloc in manager._allocators)
    if total_capacity == 0:
        return 0.0
    return total_allocated / total_capacity
