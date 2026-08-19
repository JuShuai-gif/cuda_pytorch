"""PagedAttention KV-cache block management.

The problem PagedAttention solves: with contiguous KV cache, each request must
reserve ``max_output_len`` contiguous slots up front, which fragments memory
when actual output lengths vary.  PagedAttention allocates KV cache in fixed
blocks (e.g. 16 tokens) and lets each request grab blocks on demand, so memory
waste is at most one block per request instead of (max_len - actual_len).

This module simulates both allocators over a fixed KV-cache budget and reports
the waste and the maximum number of concurrent requests each can serve.
"""

from __future__ import annotations


def simulate_contiguous(requests, block_size: int, total_blocks: int, max_len: int) -> dict:
    """Contiguous allocator: each request reserves max_len tokens up front."""
    per_req_blocks = (max_len + block_size - 1) // block_size
    max_concurrent = total_blocks // per_req_blocks
    used = 0
    wasted = 0
    for r in requests:
        if used + per_req_blocks > total_blocks:
            break
        used += per_req_blocks
        # Waste = reserved - actually used.
        actual = (r.input_len + r.output_len + block_size - 1) // block_size
        wasted += (per_req_blocks - actual) * block_size
    return {"max_concurrent": max_concurrent, "served": used // per_req_blocks,
            "wasted_tokens": wasted, "waste_ratio": wasted / (used * block_size)}


def simulate_paged(requests, block_size: int, total_blocks: int) -> dict:
    """Paged allocator: each request grabs blocks on demand."""
    free = total_blocks
    served = 0
    wasted = 0
    for r in requests:
        need = (r.input_len + r.output_len + block_size - 1) // block_size
        if need > free:
            break
        free -= need
        served += 1
        # Waste = partial block tail (internal fragmentation) only.
        wasted += need * block_size - (r.input_len + r.output_len)
    return {"max_concurrent": served, "served": served,
            "wasted_tokens": wasted, "waste_ratio": wasted / (total_blocks * block_size)}
