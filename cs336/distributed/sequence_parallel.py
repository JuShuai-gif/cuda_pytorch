"""
Sequence Parallelism for long-context training.

Implements:
- SequenceParallel: Split sequence dimension across GPUs
- RingAttention: Ring-based distributed attention for ultra-long sequences
- SequenceParallelEmbedding: Embedding lookup with sequence dimension split
- SequenceParallelCrossEntropyLoss: Parallel loss computation over vocabulary

Key concepts:
- Standard Data Parallelism splits batch dimension → sequence length limited by GPU memory
- Sequence Parallelism splits sequence dimension → enables 2x-8x longer sequences
- RingAttention decomposes attention computation into a ring of GPUs
  where each GPU computes a block of the attention matrix

RingAttention algorithm:
1. Each GPU holds a chunk of Q, K, V along the sequence dimension
2. In each ring step, each GPU sends its K, V to the next GPU
3. Each GPU computes attention for its Q chunk with the received K, V
4. Partial results are accumulated across ring steps
5. After world_size steps, each GPU has full attention output for its Q chunk

Communication: world_size * 2 * (K_chunk_size + V_chunk_size)
Memory: each GPU stores 1/world_size of Q, K, V

For RingAttention cross-entropy loss:
- Each GPU computes logits for its sequence chunk
- All-Reduce the sum of losses across sequence chunks

Reference:
    Ring Attention with Blockwise Transformers for Near-Infinite Context
    (Liu et al., 2023)
    DeepSpeed Ulysses / Sequence Parallelism
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sequence Parallel wrapper
# ---------------------------------------------------------------------------


class SequenceParallel:
    """
    Utility for splitting/rescattering tensors along the sequence dimension.

    The sequence dimension (dim=1 for [batch, seq, hidden]) is split
    across sequence-parallel ranks.

    Operations:
    - scatter: Split full sequence tensor into per-rank chunks
    - gather: Reconstruct full sequence tensor from all ranks
    """

    def __init__(
        self,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
    ):
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        if sp_size > 1 and sp_group is None and dist.is_initialized():
            self.sp_group = dist.group.WORLD

    def scatter(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Split tensor along dim across sequence-parallel ranks.

        Each rank gets its evenly-divided chunk.

        Args:
            tensor: Full tensor.
            dim: Dimension to split on (default 1 = sequence dim).

        Returns:
            This rank's chunk of the tensor.
        """
        if self.sp_size <= 1:
            return tensor
        chunks = tensor.chunk(self.sp_size, dim=dim)
        return chunks[self.sp_rank]

    def gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Gather all chunks and concatenate along dim.

        Each rank provides its chunk; result is the full concatenated tensor.

        Args:
            tensor: This rank's chunk.
            dim: Dimension to gather along.

        Returns:
            Full concatenated tensor.
        """
        if self.sp_size <= 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.sp_size)]
        if self.sp_group is not None:
            dist.all_gather(gathered, tensor, group=self.sp_group)
        return torch.cat(gathered, dim=dim)

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum-reduce tensor across sequence-parallel ranks."""
        if self.sp_size <= 1:
            return tensor
        result = tensor.clone()
        if self.sp_group is not None:
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.sp_group)
        return result


# ---------------------------------------------------------------------------
# RingAttention: Distributed self-attention over sequence dimension
# ---------------------------------------------------------------------------


@dataclass
class RingAttentionContext:
    """State for managing a RingAttention forward/backward pass."""

    q: torch.Tensor  # Query chunk (batch, local_seq, num_heads, head_dim)
    k_chunks: list[torch.Tensor] = field(default_factory=list)  # All K chunks received
    v_chunks: list[torch.Tensor] = field(default_factory=list)  # All V chunks received
    local_seq_len: int = 0
    global_seq_len: int = 0
    num_heads: int = 0
    head_dim: int = 0
    scale: float = 1.0
    mask: Optional[torch.Tensor] = None


class RingAttentionQKV:
    """
    Helper to store and rotate Q, K, V chunks during RingAttention.

    In each ring step:
    - Send current K, V chunk to next rank
    - Receive K, V chunk from previous rank
    - Compute attention with local Q and received K, V
    """

    def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.q = q
        self.k = k
        self.v = v

    def rotate_kv(self, sp_group: Any, sp_size: int, sp_rank: int) -> None:
        """
        Rotate K, V chunks one step around the ring.

        Sends local K, V to (rank+1)%sp_size and receives from (rank-1)%sp_size.
        Each rank's Q stays fixed while K, V circulate.
        """
        if sp_size <= 1 or sp_group is None:
            return

        next_rank = (sp_rank + 1) % sp_size
        prev_rank = (sp_rank - 1 + sp_size) % sp_size

        # Asynchronous send and receive for overlap
        recv_k = torch.empty_like(self.k)
        recv_v = torch.empty_like(self.v)

        send_k_req = dist.isend(self.k, dst=next_rank, group=sp_group)
        send_v_req = dist.isend(self.v, dst=next_rank, group=sp_group)
        recv_k_req = dist.irecv(recv_k, src=prev_rank, group=sp_group)
        recv_v_req = dist.irecv(recv_v, src=prev_rank, group=sp_group)

        send_k_req.wait()
        send_v_req.wait()
        recv_k_req.wait()
        recv_v_req.wait()

        self.k = recv_k
        self.v = recv_v


class RingAttention(nn.Module):
    """
    Ring Attention: distributed self-attention over sequence dimension.

    The sequence length is split across SP ranks. Each rank computes
    attention between its Q chunk and the K, V chunks that circulate
    around the ring.

    Algorithm per rank:
        1. Split Q, K, V into chunks along sequence dimension
        2. For step in range(sp_size):
           a. Compute attention: output_i += softmax(Q_i @ K_curr) @ V_curr
           b. Rotate K, V to next rank
        3. Output_i is the full attention output for rank i's sequence chunk

    Communication: (sp_size - 1) * 2 * head_size * batch * seq_chunk * num_heads bytes
    (K and V chunks circulate around the ring)

    Memory reduction: Each GPU stores 1/sp_size of the attention matrix
    """

    def __init__(
        self,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Ring attention forward pass.

        Args:
            query: (batch, seq_chunk, num_heads, head_dim) - local Q chunk
            key: (batch, seq_chunk, num_heads, head_dim) - local K chunk
            value: (batch, seq_chunk, num_heads, head_dim) - local V chunk
            mask: Optional attention mask [batch, 1, seq_chunk, global_seq] or broadcastable
            scale: Optional scaling factor (default: 1/sqrt(head_dim))

        Returns:
            (batch, seq_chunk, num_heads, head_dim) - attention output for local Q
        """
        if self.sp_size <= 1:
            # Standard attention (no parallelism)
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=scale,
            )

        head_dim = query.size(-1)
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        batch_size, local_seq, num_heads = query.size(0), query.size(1), query.size(2)
        kv = RingAttentionQKV(query, key, value)

        # Online softmax for numerical stability in ring attention
        output = torch.zeros_like(query)
        m = torch.full(
            (batch_size, num_heads, local_seq, 1),
            float("-inf"),
            device=query.device,
            dtype=query.dtype,
        )
        l = torch.zeros(
            (batch_size, num_heads, local_seq, 1),
            device=query.device,
            dtype=query.dtype,
        )

        q_for_attn = query.transpose(1, 2)  # (batch, num_heads, local_seq, head_dim)

        for step in range(self.sp_size):
            k_curr = kv.k.transpose(1, 2)  # (batch, num_heads, local_seq, head_dim)
            v_curr = kv.v.transpose(1, 2)

            # Compute attention scores for this ring step
            scores = torch.matmul(q_for_attn, k_curr.transpose(-2, -1)) * scale
            # (batch, num_heads, local_seq, local_seq)

            # Apply mask if provided (mask covers the global sequence)
            if mask is not None:
                # Extract mask for current rotation
                mask_chunk = mask[:, :, :, step * local_seq : (step + 1) * local_seq]
                scores = scores + mask_chunk

            # Online softmax update
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            exp_scores = torch.exp(scores - m_new)
            exp_prev = torch.exp(m - m_new)

            l = l * exp_prev + exp_scores.sum(dim=-1, keepdim=True)
            output = output * exp_prev + torch.matmul(exp_scores, v_curr)
            m = m_new

            # Rotate K, V
            kv.rotate_kv(self.sp_group, self.sp_size, self.sp_rank)

        # Normalize by softmax denominator
        output = output / l
        # Restore original shape: (batch, seq_chunk, num_heads, head_dim)
        output = output.transpose(1, 2)

        return output


# ---------------------------------------------------------------------------
# Sequence Parallel Embedding
# ---------------------------------------------------------------------------


class SequenceParallelEmbedding(nn.Module):
    """
    Embedding layer with sequence parallelism.

    The embedding table is replicated; each rank handles a chunk
    of the input sequence. The output is the embedding lookup for
    that sequence chunk.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Embedding dimension.
        padding_idx: Padding token index.
        sp_size: Sequence parallel world size.
        sp_rank: This rank's index.
        sp_group: Process group for sequence parallelism.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
    ):
        super().__init__()
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.sp_group = sp_group

        # Full embedding table (replicated, not huge for typical vocab sizes)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for the local sequence chunk.

        Input ids are pre-split along sequence dimension by data loader.

        Args:
            input_ids: (batch, seq_chunk). Local sequence chunk.

        Returns:
            (batch, seq_chunk, hidden_size). Local embedding output.
        """
        return self.embedding(input_ids)


# ---------------------------------------------------------------------------
# Sequence Parallel Cross-Entropy Loss
# ---------------------------------------------------------------------------


class SequenceParallelCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with sequence parallelism.

    Each rank computes logits for its sequence chunk and local
    vocabulary partition. The final loss is the average across
    all ranks' contributions.

    Two modes:
    1. Vocab-parallel: Each rank has vocabulary chunk, All-Reduce loss
    2. Sequence-parallel: Each rank has sequence chunk, All-Reduce loss

    For the LM head with vocabulary parallelism:
    - Each rank computes logits for its vocab partition
    - Mask out tokens not in this partition
    - Sum losses across ranks via All-Reduce
    """

    def __init__(
        self,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with sequence and vocabulary parallelism.

        Args:
            logits: (batch, seq_chunk, local_vocab_size). Local logits.
            labels: (batch, seq_chunk). Token IDs in [0, global_vocab_size).
            vocab_start: Start of local vocabulary partition.
            vocab_end: End of local vocabulary partition.

        Returns:
            Scalar loss tensor (average across all ranks if reduction='mean').
        """
        if vocab_end is None:
            vocab_end = vocab_start + logits.size(-1)

        # Shift labels to local vocabulary range
        local_labels = labels - vocab_start

        # Mask: only compute loss for tokens in this partition
        valid_mask = (local_labels >= 0) & (local_labels < logits.size(-1))
        valid_mask = valid_mask & (labels != self.ignore_index)

        if not valid_mask.any():
            # No valid tokens in this rank's partition; return zero
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            local_labels = local_labels.clamp(0, logits.size(-1) - 1)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                local_labels.view(-1),
                ignore_index=self.ignore_index,
                reduction="sum" if self.reduction == "mean" else "none",
            )

        # All-Reduce to get total loss across all SP ranks
        if self.sp_size > 1 and self.sp_group is not None:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.sp_group)

        if self.reduction == "mean":
            total_valid = valid_mask.sum().float()
            if self.sp_size > 1 and self.sp_group is not None:
                dist.all_reduce(total_valid, op=dist.ReduceOp.SUM, group=self.sp_group)
            loss = loss / total_valid.clamp(min=1)

        return loss
