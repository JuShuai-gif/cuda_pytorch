"""Pruning methods and their FLOPs-vs-speedup implications."""

from __future__ import annotations

import torch


def magnitude_prune_unstructured(w: torch.Tensor, sparsity: float):
    """Zero out the smallest |weights| so that ``sparsity`` fraction are zero.

    Returns (pruned_weight, mask).  The pruned weight is still a dense tensor;
    a plain matmul does not skip the zeros, which is the whole point of the
    "FLOPs reduction != real speedup" lesson.
    """
    flat = w.abs().flatten()
    k = int(flat.numel() * sparsity)  # number of elements to zero
    if k <= 0:
        return w.clone(), torch.ones_like(w, dtype=torch.bool)
    threshold = torch.kthvalue(flat, k).values
    mask = w.abs() >= threshold
    return (w * mask), mask


def structured_row_prune(w: torch.Tensor, sparsity: float):
    """Remove whole rows (input channels) with the smallest L2 norm.

    Returns (pruned_weight, kept_indices).  The pruned weight is a smaller
    dense tensor, so a matmul genuinely shrinks - this is real speedup.
    """
    row_norm = w.norm(dim=1)
    k = int(w.shape[0] * (1 - sparsity))
    if k >= w.shape[0]:
        return w.clone(), torch.arange(w.shape[0])
    kept = torch.topk(row_norm, k).indices
    return w[kept].contiguous(), kept


def sparsity(w: torch.Tensor) -> float:
    """Fraction of zero elements."""
    return (w == 0).float().mean().item()


def to_2to4(w: torch.Tensor):
    """Convert to NVIDIA 2:4 structured sparsity (2 nonzeros per 4 elements)."""
    return torch.sparse.to_sparse_semi_structured(w)
