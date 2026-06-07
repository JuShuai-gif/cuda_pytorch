"""
PagedAttention kernel in Triton for efficient LLM inference.

Implements the vLLM-style PagedAttention algorithm where KV cache
is stored in non-contiguous fixed-size blocks (pages), and attention
is computed by iterating over the block table.

Key features:
    - Indirect KV cache access via block tables
    - Configurable block sizes (16/32 tokens per block)
    - Supports batch processing with variable sequence lengths
    - Shared memory prefetch for block table lookups

Reference: Kwon et al., "Efficient Memory Management for Large
Language Model Serving with PagedAttention", SOSP 2023.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
#  PagedAttention kernel
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _paged_attention_kernel(
        Q_ptr,
        K_cache_ptr,
        V_cache_ptr,
        O_ptr,
        block_table_ptr,
        seq_lens_ptr,
        stride_qb: int,
        stride_qh: int,
        stride_qd: int,
        block_table_stride: int,
        N_KV_HEADS: int,
        HEAD_DIM: int,
        BLOCK_SIZE: tl.constexpr,  # tokens per KV cache page
        BLOCK_N: tl.constexpr,  # number of tokens processed at once in attention
        SCALE: tl.constexpr,
    ):
        """PagedAttention kernel for single-query decoding.

        Each program processes one query head of one sequence.
        The block table maps logical token positions to physical
        KV cache block indices.

        Args:
            Q_ptr: Query tensor (batch * n_heads, head_dim).
            K_cache_ptr: Paged K cache (num_blocks, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM).
            V_cache_ptr: Paged V cache (num_blocks, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM).
            O_ptr: Output tensor (batch * n_heads, head_dim).
            block_table_ptr: Block table (batch, max_blocks_per_seq).
            seq_lens_ptr: Sequence lengths (batch,).
        """
        pid = tl.program_id(0)
        batch_idx = pid // N_KV_HEADS
        head_idx = pid % N_KV_HEADS

        seq_len = tl.load(seq_lens_ptr + batch_idx)
        if seq_len <= 0:
            return

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q_ptr + pid * stride_qb + offs_d * stride_qd)

        # Online softmax statistics
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        # Scale Q once (FlashAttention-2 optimization)
        q = q * SCALE

        num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        num_kv_heads = N_KV_HEADS

        for block_idx in range(num_blocks):
            # Block table lookup
            physical_block = tl.load(
                block_table_ptr + batch_idx * block_table_stride + block_idx
            )

            tokens_in_block = min(BLOCK_SIZE, seq_len - block_idx * BLOCK_SIZE)

            # Load K and V from this block
            kv_head = head_idx % N_KV_HEADS  # handle GQA
            kv_head = min(kv_head, N_KV_HEADS - 1)

            k_base = (
                K_cache_ptr
                + physical_block * (N_KV_HEADS * BLOCK_SIZE * HEAD_DIM)
                + kv_head * (BLOCK_SIZE * HEAD_DIM)
            )
            v_base = (
                V_cache_ptr
                + physical_block * (N_KV_HEADS * BLOCK_SIZE * HEAD_DIM)
                + kv_head * (BLOCK_SIZE * HEAD_DIM)
            )

            # Process tokens in this block
            for tok in range(0, tokens_in_block, BLOCK_N):
                bn = min(BLOCK_N, tokens_in_block - tok)
                offs_n = tl.arange(0, BLOCK_N)
                mask_n = offs_n < bn

                k = tl.load(
                    k_base + (tok + offs_n) * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None],
                    other=0.0,
                ).to(tl.float32)
                v = tl.load(
                    v_base + (tok + offs_n) * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None],
                    other=0.0,
                ).to(tl.float32)

                # S = Q @ K^T: (1, HEAD_DIM) @ (HEAD_DIM, BN) = (BN,)
                s = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for d in range(HEAD_DIM):
                    s += q[d] * k[:, d]

                # Online softmax update
                m_ij = tl.max(s)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new)
                p = tl.where(mask_n, p, 0.0)
                l_i = alpha * l_i + tl.sum(p, axis=0)
                acc = acc * alpha
                # Accumulate P @ V
                acc += tl.sum(p[:, None] * v, axis=0)
                m_i = m_new

        # Final normalization
        l_i = l_i + 1e-12
        acc = acc / l_i

        tl.store(O_ptr + pid * stride_qb + offs_d * stride_qd, acc)

else:
    pass


def paged_attention_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention using paged KV cache.

    This is the core kernel used in vLLM-style inference where
    the KV cache is organized into fixed-size pages. The block
    table maps logical sequence positions to physical page indices.

    Args:
        q: Query tensor of shape (batch, n_heads, head_dim) for decode
           (single query token per sequence).
        k_cache: Paged key cache of shape
                 (num_blocks, n_kv_heads, block_size, head_dim).
        v_cache: Paged value cache of shape
                 (num_blocks, n_kv_heads, block_size, head_dim).
        block_table: Block table of shape (batch, max_blocks_per_seq).
                     Maps logical block index -> physical block index.
                     Unused entries should be set to -1 or 0.
        seq_lens: Sequence lengths of shape (batch,). Each value is the
                  total number of tokens (context + generated) for that sequence.
        block_size: Number of tokens per KV cache page (default 16).
        sm_scale: Softmax scale factor. Defaults to 1/sqrt(head_dim).

    Returns:
        Output tensor of shape (batch, n_heads, head_dim).

    Raises:
        ValueError: If input shapes are incompatible.
    """
    if not HAS_TRITON:
        return _paged_attention_pytorch_fallback(
            q, k_cache, v_cache, block_table, seq_lens, block_size, sm_scale
        )

    batch, n_heads, head_dim = q.shape
    _, n_kv_heads, cache_block_size, cache_head_dim = k_cache.shape

    if cache_block_size != block_size:
        raise ValueError(
            f"block_size mismatch: {block_size} vs cache {cache_block_size}"
        )
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: {head_dim} vs cache {cache_head_dim}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous()
    seq_lens = seq_lens.contiguous()

    o = torch.empty_like(q)

    # Flatten batch+heads into first dimension
    q_view = q.view(batch * n_heads, head_dim)

    grid = (batch * n_heads,)

    _paged_attention_kernel[grid](
        q_view,
        k_cache,
        v_cache,
        o.view(batch * n_heads, head_dim),
        block_table,
        seq_lens,
        q_view.stride(0) if q_view.dim() > 1 else head_dim,
        1 if q_view.dim() > 1 else head_dim,
        1,
        block_table.stride(0),
        n_kv_heads,
        head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=min(32, block_size),
        SCALE=sm_scale,
    )

    return o


# ==============================================================================
#  Block table utility functions
# ==============================================================================


def create_block_table(
    seq_lens: torch.Tensor,
    num_blocks: int,
    block_size: int = 16,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Create a block table for paged attention.

    Maps logical blocks to physical blocks. For simplicity,
    allocates physical blocks sequentially. In a real serving
    system, this would be managed by a block allocator.

    Args:
        seq_lens: Tensor of sequence lengths (batch,).
        num_blocks: Total number of physical KV cache blocks.
        block_size: Number of tokens per block.
        device: Device to allocate tensors on.

    Returns:
        Block table of shape (batch, max_blocks) with values -1
        for unused slots.
    """
    batch = seq_lens.shape[0]
    max_blocks_per_seq = (seq_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = (
        ((seq_lens.float() + block_size - 1) / block_size).long().sum().item()
    )

    if total_blocks_needed > num_blocks:
        raise ValueError(
            f"Not enough blocks: need {total_blocks_needed}, have {num_blocks}"
        )

    block_table = torch.full(
        (batch, max_blocks_per_seq), -1, dtype=torch.int32, device=device
    )

    next_free_block = 0
    for b in range(batch):
        n_blocks = (seq_lens[b].item() + block_size - 1) // block_size
        for i in range(n_blocks):
            block_table[b, i] = next_free_block
            next_free_block += 1

    return block_table


def block_table_lookup(
    block_table: torch.Tensor,
    seq_idx: int,
    logical_block: int,
) -> int:
    """Look up the physical block index for a given logical position.

    Args:
        block_table: Block table of shape (batch, max_blocks).
        seq_idx: Sequence index in the batch.
        logical_block: Logical block index within the sequence.

    Returns:
        Physical block index, or -1 if not allocated.

    Raises:
        IndexError: If seq_idx or logical_block are out of range.
    """
    if seq_idx >= block_table.shape[0]:
        raise IndexError(f"seq_idx {seq_idx} >= batch size {block_table.shape[0]}")
    if logical_block >= block_table.shape[1]:
        raise IndexError(
            f"logical_block {logical_block} >= max blocks {block_table.shape[1]}"
        )

    return int(block_table[seq_idx, logical_block].item())


# ==============================================================================
#  PyTorch fallback
# ==============================================================================


def _paged_attention_pytorch_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """PyTorch reference implementation for paged attention.

    Gathers K and V from pages using the block table and computes
    standard scaled dot-product attention.

    Args:
        q: Query (batch, n_heads, head_dim) - single token per sequence.
        k_cache: Paged K cache (num_blocks, n_kv_heads, block_size, head_dim).
        v_cache: Paged V cache (num_blocks, n_kv_heads, block_size, head_dim).
        block_table: (batch, max_blocks).
        seq_lens: (batch,).
        block_size: Tokens per block.
        sm_scale: Softmax scale.

    Returns:
        Output (batch, n_heads, head_dim).
    """
    batch, n_heads, head_dim = q.shape
    n_kv_heads = k_cache.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    max_blocks_per_seq = block_table.shape[1]
    outputs: List[torch.Tensor] = []

    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            outputs.append(
                torch.zeros(n_heads, head_dim, device=q.device, dtype=q.dtype)
            )
            continue

        k_chunks: List[torch.Tensor] = []
        v_chunks: List[torch.Tensor] = []

        remaining = seq_len
        for block_idx in range(max_blocks_per_seq):
            if remaining <= 0:
                break
            physical_block = int(block_table[b, block_idx].item())
            if physical_block < 0:
                break

            take = min(block_size, remaining)
            k_chunks.append(k_cache[physical_block, :, :take, :])
            v_chunks.append(v_cache[physical_block, :, :take, :])
            remaining -= take

        if not k_chunks:
            outputs.append(
                torch.zeros(n_heads, head_dim, device=q.device, dtype=q.dtype)
            )
            continue

        k = torch.cat(k_chunks, dim=1)  # (n_kv_heads, seq_len, head_dim)
        v = torch.cat(v_chunks, dim=1)

        # Handle GQA: expand KV heads to match Q heads
        if n_heads != n_kv_heads:
            ratio = n_heads // n_kv_heads
            k = k.repeat_interleave(ratio, dim=0)
            v = v.repeat_interleave(ratio, dim=0)

        q_b = q[b : b + 1]  # (1, n_heads, head_dim)

        scores = torch.matmul(q_b, k.transpose(-2, -1)) * sm_scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (1, n_heads, head_dim)
        outputs.append(out.squeeze(0))

    return torch.stack(outputs, dim=0)


# ==============================================================================
#  PagedAttention manager for multi-sequence serving
# ==============================================================================


class PagedAttentionManager:
    """Manages paged KV cache for multi-sequence LLM serving.

    Provides allocation, writing, and reading of KV cache pages
    across multiple layers with block table management.

    Attributes:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        block_size: Tokens per KV cache page.
        layer_caches: Per-layer paged KV cache tensors.
        block_tables: Per-sequence block table mapping.
        seq_lens: Current sequence lengths.
        free_blocks: Stack of available physical block indices.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        initial_blocks: int = 256,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        # Pre-allocate KV cache blocks for all layers
        self._layer_caches_k: List[torch.Tensor] = []
        self._layer_caches_v: List[torch.Tensor] = []
        for _ in range(num_layers):
            k = torch.zeros(
                initial_blocks,
                num_kv_heads,
                block_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
            v = torch.zeros(
                initial_blocks,
                num_kv_heads,
                block_size,
                head_dim,
                dtype=dtype,
                device=device,
            )
            self._layer_caches_k.append(k)
            self._layer_caches_v.append(v)

        self._total_blocks = initial_blocks
        self._next_seq_id = 0

        # Per-sequence state
        self._block_tables: dict[int, list[int]] = {}
        self._seq_lens: dict[int, int] = {}

        # Free block management
        self._free_blocks: list[int] = list(range(initial_blocks))

    def add_sequence(self, prompt_len: int) -> int:
        """Register a new sequence and allocate initial blocks.

        Args:
            prompt_len: Number of tokens in the prompt.

        Returns:
            Sequence ID for the new sequence.

        Raises:
            RuntimeError: If there are insufficient free blocks.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        if len(self._free_blocks) < num_blocks_needed:
            self._grow_cache(num_blocks_needed - len(self._free_blocks))

        block_table: list[int] = []
        for _ in range(num_blocks_needed):
            block_table.append(self._free_blocks.pop())

        self._block_tables[seq_id] = block_table
        self._seq_lens[seq_id] = prompt_len

        return seq_id

    def remove_sequence(self, seq_id: int) -> None:
        """Free all blocks owned by a sequence.

        Args:
            seq_id: Sequence to remove.
        """
        if seq_id in self._block_tables:
            self._free_blocks.extend(self._block_tables[seq_id])
            del self._block_tables[seq_id]
        if seq_id in self._seq_lens:
            del self._seq_lens[seq_id]

    def append_token(
        self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write a single new token's KV to the cache.

        Automatically allocates a new block if the current block is full.

        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            k: Key tensor of shape (n_kv_heads, 1, head_dim).
            v: Value tensor of shape (n_kv_heads, 1, head_dim).
        """
        current_len = self._seq_lens[seq_id]
        block_idx = current_len // self.block_size
        offset = current_len % self.block_size

        block_table = self._block_tables[seq_id]

        if block_idx >= len(block_table):
            if not self._free_blocks:
                self._grow_cache(1)
            block_table.append(self._free_blocks.pop())

        physical_block = block_table[block_idx]

        self._layer_caches_k[layer_idx][physical_block, :, offset : offset + 1, :] = k
        self._layer_caches_v[layer_idx][physical_block, :, offset : offset + 1, :] = v

        self._seq_lens[seq_id] = current_len + 1

    def get_k_cache(self, layer_idx: int) -> torch.Tensor:
        """Get the full K cache tensor for a layer."""
        return self._layer_caches_k[layer_idx]

    def get_v_cache(self, layer_idx: int) -> torch.Tensor:
        """Get the full V cache tensor for a layer."""
        return self._layer_caches_v[layer_idx]

    def get_block_table_tensor(
        self, seq_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Get block table as a padded tensor for kernel dispatch.

        Args:
            seq_ids: List of sequence IDs to include. If None, all sequences.

        Returns:
            Block table tensor of shape (batch, max_blocks) with -1 padding.
        """
        if seq_ids is None:
            seq_ids = list(self._block_tables.keys())

        if not seq_ids:
            return torch.empty(0, 0, dtype=torch.int32, device=self.device)

        max_blocks = max(len(self._block_tables.get(sid, [])) for sid in seq_ids)

        table = torch.full(
            (len(seq_ids), max_blocks),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        for i, sid in enumerate(seq_ids):
            bt = self._block_tables.get(sid, [])
            table[i, : len(bt)] = torch.tensor(
                bt, dtype=torch.int32, device=self.device
            )

        return table

    def get_seq_lens_tensor(self, seq_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Get sequence lengths as a tensor.

        Args:
            seq_ids: List of sequence IDs to include. If None, all sequences.

        Returns:
            Sequence lengths tensor of shape (batch,).
        """
        if seq_ids is None:
            seq_ids = list(self._seq_lens.keys())

        if not seq_ids:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        return torch.tensor(
            [self._seq_lens.get(sid, 0) for sid in seq_ids],
            dtype=torch.int32,
            device=self.device,
        )

    def _grow_cache(self, additional_blocks: int) -> None:
        """Expand the KV cache by allocating more blocks.

        Args:
            additional_blocks: Number of new blocks to add.
        """
        for l in range(self.num_layers):
            new_k = torch.zeros(
                additional_blocks,
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            new_v = torch.zeros(
                additional_blocks,
                self.num_kv_heads,
                self.block_size,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self._layer_caches_k[l] = torch.cat([self._layer_caches_k[l], new_k], dim=0)
            self._layer_caches_v[l] = torch.cat([self._layer_caches_v[l], new_v], dim=0)

        start_idx = self._total_blocks
        self._total_blocks += additional_blocks
        self._free_blocks.extend(range(start_idx, self._total_blocks))


# ==============================================================================
#  Correctness tests
# ==============================================================================


def test_paged_attention(tol: float = 1e-2) -> Tuple[bool, float]:
    """Verify paged_attention_kernel against PyTorch fallback."""
    if not torch.cuda.is_available():
        return True, 0.0

    torch.manual_seed(42)
    device = "cuda"
    batch, n_heads, head_dim = 2, 4, 64
    n_kv_heads = n_heads
    block_size = 16
    seq_len = 20

    q = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float16)

    num_blocks = 8
    k_cache = torch.randn(
        num_blocks,
        n_kv_heads,
        block_size,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    v_cache = torch.randn(
        num_blocks,
        n_kv_heads,
        block_size,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    block_table = torch.zeros(batch, 3, dtype=torch.int32, device=device)
    block_table[0, 0] = 0
    block_table[0, 1] = 1
    block_table[1, 0] = 2
    block_table[1, 1] = 3

    seq_lens = torch.tensor([seq_len, 17], dtype=torch.int32, device=device)

    y_ref = _paged_attention_pytorch_fallback(
        q, k_cache, v_cache, block_table, seq_lens, block_size
    )

    if HAS_TRITON:
        y_kernel = paged_attention_kernel(
            q, k_cache, v_cache, block_table, seq_lens, block_size
        )
        max_diff = (y_ref - y_kernel).abs().max().item()
        return max_diff < tol, max_diff

    return True, 0.0


def test_manager(tol: float = 1e-2) -> Tuple[bool, float]:
    """Test PagedAttentionManager operations."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manager = PagedAttentionManager(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        initial_blocks=16,
        dtype=torch.float32,
        device=device,
    )

    seq1 = manager.add_sequence(20)  # needs 2 blocks
    seq2 = manager.add_sequence(5)  # needs 1 block

    assert seq1 == 0 and seq2 == 1, f"Unexpected seq ids: {seq1}, {seq2}"

    # Write a token
    k = torch.randn(4, 1, 64, device=device)
    v = torch.randn(4, 1, 64, device=device)
    manager.append_token(seq1, 0, k, v)
    assert manager._seq_lens[seq1] == 21

    # Verify block tensor
    bt = manager.get_block_table_tensor([seq1, seq2])
    assert bt.shape[0] == 2, f"Unexpected block table shape: {bt.shape}"

    # Remove sequence
    manager.remove_sequence(seq2)
    assert len(manager._block_tables) == 1

    return True, 0.0


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    tests = [
        ("paged_attention", test_paged_attention),
        ("manager", test_manager),
    ]

    all_pass = True
    for name, test_fn in tests:
        ok, diff = test_fn()
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name}: {status} (max diff = {diff:.2e})")

    if all_pass:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.")
