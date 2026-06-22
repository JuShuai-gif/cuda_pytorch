"""
KV cache implementations for LLM inference.

Two variants:
  1. KVCache: Contiguous pre-allocated KV cache.
  2. PagedKVCache: Block-based KV cache (simplified vLLM PagedAttention concept).

In production LLM inference (vLLM, TensorRT-LLM, TGI), KV cache is the
largest memory consumer. Efficient management is critical for:
  - Serving multiple sequences concurrently (batching)
  - Long-context inference (avoiding OOM from over-allocation)
  - Avoiding memory fragmentation
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


class KVCache:
    """Pre-allocated contiguous KV cache for inference.

    Shape: [num_layers, 2, batch, num_heads, max_seq_len, head_dim]
    - axis 1: 0 for K, 1 for V
    - Pre-allocated to max_seq_len for efficiency (single allocation)

    Suitable for single-sequence or small-batch inference where memory
    waste from pre-allocation is acceptable.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # [num_layers, 2 (K/V), batch, num_heads, max_seq_len, head_dim]
        self.buffer = torch.zeros(
            num_layers,
            2,
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        # Track how many tokens are filled per sequence
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def update(
        self,
        layer_idx: int,
        batch_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Write K and V at specified positions in the cache.

        Args:
            layer_idx: Layer index (0 to num_layers-1).
            batch_idx: Batch index (0 to batch_size-1).
            k: Key tensor [num_heads, seq_len, head_dim] or [1, num_heads, seq_len, head_dim].
            v: Value tensor same shape as k.
            positions: 1D tensor of position indices to write to [seq_len].
        """
        # Handle batch dimension if present
        if k.dim() == 4:
            k = k.squeeze(0)
            v = v.squeeze(0)

        assert k.dim() == 3, f"Expected 3D K after squeeze, got {k.dim()}D: {k.shape}"

        self.buffer[layer_idx, 0, batch_idx, :, positions, :] = k
        self.buffer[layer_idx, 1, batch_idx, :, positions, :] = v

        new_len = int(positions.max().item()) + 1
        if new_len > self.seq_lens[batch_idx]:
            self.seq_lens[batch_idx] = new_len

    def get(
        self,
        layer_idx: int,
        batch_idx: int,
        up_to: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the valid (filled) portion of K/V cache for a layer.

        Args:
            layer_idx: Layer index.
            batch_idx: Batch index.
            up_to: Optional limit on sequence length to return.

        Returns:
            (k_cache, v_cache): [1, num_heads, valid_len, head_dim]
        """
        limit = up_to if up_to is not None else int(self.seq_lens[batch_idx].item())
        k = self.buffer[layer_idx, 0, batch_idx : batch_idx + 1, :, :limit, :].clone()
        v = self.buffer[layer_idx, 1, batch_idx : batch_idx + 1, :, :limit, :].clone()
        return k, v

    def get_full(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the full K/V cache for a layer (all batches, up to max filled).

        Returns:
            (k_cache, v_cache): [batch, num_heads, max_filled, head_dim]
        """
        max_len = int(self.seq_lens.max().item())
        k = self.buffer[layer_idx, 0, :, :, :max_len, :].clone()
        v = self.buffer[layer_idx, 1, :, :, :max_len, :].clone()
        return k, v

    def reset(self) -> None:
        """Reset all cache entries and sequence lengths."""
        self.buffer.zero_()
        self.seq_lens.zero_()

    def memory_bytes(self) -> int:
        """Calculate total memory usage in bytes."""
        return self.buffer.numel() * self.buffer.element_size()

    def memory_gb(self) -> float:
        """Calculate total memory usage in GB."""
        return self.memory_bytes() / (1024**3)

    def fill_status(self) -> torch.Tensor:
        """Return sequence lengths for all batches."""
        return self.seq_lens.clone()


class PagedKVCache:
    """Block-based KV cache (simplified vLLM PagedAttention).

    Instead of allocating one contiguous buffer per sequence:
      - Divide max_seq_len into fixed-size blocks (e.g., 16 tokens each)
      - Maintain a pool of free blocks
      - Allocate blocks from the pool as sequences grow
      - Map logical positions to physical block indices

    Benefits over contiguous allocation:
      - No pre-allocation waste: only allocate blocks as needed
      - Less fragmentation: all sequences share a common block pool
      - Better memory utilization for batched inference

    This is a simplified demonstration of the core PagedAttention concept
    used in vLLM. Production implementations have many more optimizations.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # Physical blocks: [num_layers, 2 (K/V), num_blocks, num_heads, block_size, head_dim]
        self.blocks = torch.zeros(
            num_layers,
            2,
            num_blocks,
            num_heads,
            block_size,
            head_dim,
            device=device,
            dtype=dtype,
        )

        # Free block mask (True = free)
        self.free_blocks = torch.ones(num_blocks, dtype=torch.bool, device=device)

        # Block table per sequence: maps logical block index -> physical block idx
        # [num_layers, max_sequences, max_logical_blocks]
        # We'll use a dynamic approach instead
        self._seq_block_tables: List[List[List[int]]] = []
        self._seq_lengths: List[int] = []
        self._max_sequences = 0

    def allocate_sequence(self) -> int:
        """Allocate a new sequence slot. Returns sequence index."""
        seq_idx = len(self._seq_block_tables)
        self._seq_block_tables.append([])
        self._seq_lengths.append(0)
        self._max_sequences += 1
        return seq_idx

    def _num_logical_blocks(self, seq_len: int) -> int:
        """Number of blocks needed for a given sequence length."""
        return math.ceil(seq_len / self.block_size)

    def _allocate_block(self) -> int:
        """Allocate a single physical block. Returns block index or -1 if none free."""
        free_indices = torch.nonzero(self.free_blocks, as_tuple=True)[0]
        if len(free_indices) == 0:
            return -1
        block_idx = int(free_indices[0].item())
        self.free_blocks[block_idx] = False
        return block_idx

    def _free_block(self, block_idx: int) -> None:
        """Return a block to the free pool."""
        self.free_blocks[block_idx] = True

    def grow(
        self,
        seq_idx: int,
        num_blocks: int,
    ) -> Tuple[List[int], int]:
        """Allocate additional blocks for a sequence.

        Args:
            seq_idx: Sequence index.
            num_blocks: Total number of blocks needed.

        Returns:
            (newly_allocated_blocks, num_allocated)
        """
        current = len(self._seq_block_tables[seq_idx])
        needed = num_blocks - current
        new_blocks = []
        for _ in range(needed):
            b = self._allocate_block()
            if b < 0:
                raise RuntimeError(f"No free blocks available (needed {needed})")
            new_blocks.append(b)
        self._seq_block_tables[seq_idx].extend(new_blocks)
        return new_blocks, len(new_blocks)

    def write(
        self,
        seq_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Write K/V tokens into page-aligned blocks.

        Args:
            seq_idx: Sequence index.
            k: Key tensor [num_heads, seq_len, head_dim].
            v: Value tensor [num_heads, seq_len, head_dim].
            start_pos: Starting logical position to write.
        """
        num_heads, num_tokens, head_dim = k.shape
        block_table = self._seq_block_tables[seq_idx]

        for token_offset in range(num_tokens):
            logical_pos = start_pos + token_offset
            block_idx = logical_pos // self.block_size
            offset_in_block = logical_pos % self.block_size

            assert block_idx < len(block_table), (
                f"Block {block_idx} not allocated for seq {seq_idx}"
            )
            physical_block = block_table[block_idx]

            for layer_idx in range(self.num_layers):
                self.blocks[layer_idx, 0, physical_block, :, offset_in_block, :] = k[
                    :, token_offset, :
                ]
                self.blocks[layer_idx, 1, physical_block, :, offset_in_block, :] = v[
                    :, token_offset, :
                ]

        new_len = max(self._seq_lengths[seq_idx], start_pos + num_tokens)
        self._seq_lengths[seq_idx] = new_len

    def read(
        self,
        seq_idx: int,
        layer_idx: int,
        up_to: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K/V cache for a sequence at a given layer.

        Args:
            seq_idx: Sequence index.
            layer_idx: Layer index.
            up_to: Optional limit on sequence length.

        Returns:
            (k_cache, v_cache): [1, num_heads, valid_len, head_dim]
        """
        limit = up_to if up_to is not None else self._seq_lengths[seq_idx]
        if limit == 0:
            k = torch.zeros(
                1, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype
            )
            v = torch.zeros(
                1, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype
            )
            return k, v

        block_table = self._seq_block_tables[seq_idx]

        # Allocate output tensors
        k_out = torch.zeros(
            1, self.num_heads, limit, self.head_dim, device=self.device, dtype=self.dtype
        )
        v_out = torch.zeros(
            1, self.num_heads, limit, self.head_dim, device=self.device, dtype=self.dtype
        )

        for pos in range(limit):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            physical_block = block_table[block_idx]
            k_out[0, :, pos, :] = self.blocks[layer_idx, 0, physical_block, :, offset, :]
            v_out[0, :, pos, :] = self.blocks[layer_idx, 1, physical_block, :, offset, :]

        return k_out, v_out

    def free_sequence(self, seq_idx: int) -> None:
        """Free all blocks allocated to a sequence."""
        for block_idx in self._seq_block_tables[seq_idx]:
            self._free_block(block_idx)
        self._seq_block_tables[seq_idx] = []
        self._seq_lengths[seq_idx] = 0

    def memory_bytes(self) -> int:
        """Calculate total memory usage of the block pool in bytes."""
        return self.blocks.numel() * self.blocks.element_size()

    def utilization(self) -> float:
        """Fraction of blocks currently in use."""
        if self.num_blocks == 0:
            return 0.0
        used = int((~self.free_blocks).sum().item())
        return used / self.num_blocks


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        # Test KVCache
        cache = KVCache(
            num_layers=2,
            batch_size=2,
            num_heads=4,
            max_seq_len=64,
            head_dim=64,
        )
        print(f"KVCache memory: {cache.memory_gb():.4f} GB")

        k_test = torch.randn(4, 16, 64, device="cuda")
        v_test = torch.randn(4, 16, 64, device="cuda")
        positions = torch.arange(16, device="cuda")

        cache.update(0, 0, k_test, v_test, positions)
        k_out, v_out = cache.get(0, 0)

        assert k_out.shape == (1, 4, 16, 64)
        assert torch.allclose(k_out.squeeze(0), k_test, atol=1e-5)
        assert cache.seq_lens[0].item() == 16
        print("KVCache write/read: OK")

        # Test reset
        cache.reset()
        assert cache.seq_lens[0].item() == 0
        print("KVCache reset: OK")

        # Test PagedKVCache
        pcache = PagedKVCache(
            num_layers=2,
            num_heads=4,
            max_seq_len=128,
            head_dim=64,
            block_size=16,
            num_blocks=64,
        )
        print(f"PagedKVCache memory: {pcache.memory_bytes() / (1024**3):.4f} GB")

        seq_idx = pcache.allocate_sequence()
        blocks_needed = pcache._num_logical_blocks(32)
        pcache.grow(seq_idx, blocks_needed)

        k_test = torch.randn(4, 32, 64, device="cuda")
        v_test = torch.randn(4, 32, 64, device="cuda")
        pcache.write(seq_idx, k_test, v_test, start_pos=0)

        k_out, v_out = pcache.read(seq_idx, 0)
        assert k_out.shape == (1, 4, 32, 64)
        assert torch.allclose(k_out.squeeze(0), k_test, atol=1e-5)
        assert pcache.utilization() > 0

        pcache.free_sequence(seq_idx)
        assert pcache.utilization() == 0.0
        print("PagedKVCache allocate/write/read/free: OK")

        print("All KV cache demos passed!")
