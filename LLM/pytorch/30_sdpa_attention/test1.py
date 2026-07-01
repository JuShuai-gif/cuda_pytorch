"""SDPA / FlashAttention demo: backend selection, causal, memory comparison.

Companion script for sdpa_attention/sdpa_attention.md.
  1. basic SDPA:            auto backend selection
  2. sdpa_kernel:           force specific backend
  3. causal attention:      is_causal=True vs explicit mask
  4. memory comparison:     SDPA vs manual attention
  5. MultiheadAttention:    the full module

Run:
    python test1.py                # full demo (needs CUDA)
    python test1.py basic          # basic SDPA
    python test1.py backend        # backend selection
    python test1.py causal         # causal attention
    python test1.py memory         # memory comparison
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return False
    return True


# ============ 1. Basic SDPA ============
def exp_basic():
    if not _cuda():
        return
    print("=" * 60)
    print("1. Basic SDPA: auto backend selection")
    print("=" * 60)

    B, H, S, D = 4, 8, 1024, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    out = F.scaled_dot_product_attention(q, k, v)
    print(f"  Input:  q={list(q.shape)} k={list(k.shape)} v={list(v.shape)}")
    print(f"  Output: {list(out.shape)}")

    # Check which backend was used
    flash_enabled = torch.backends.cuda.flash_sdp_enabled()
    mem_efficient_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_enabled = torch.backends.cuda.math_sdp_enabled()
    print(f"  Flash SDP enabled:       {flash_enabled}")
    print(f"  Mem-efficient enabled:   {mem_efficient_enabled}")
    print(f"  Math SDP enabled:        {math_enabled}")
    print(f"  → fp16 + CUDA + seq=1024 → FlashAttention v2/v3 selected")
    print()


# ============ 2. Backend selection ============
def exp_backend():
    if not _cuda():
        return
    print("=" * 60)
    print("2. sdpa_kernel: force specific backend")
    print("=" * 60)

    B, H, S, D = 2, 4, 512, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # Flash only
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        out_flash = F.scaled_dot_product_attention(q, k, v)
    print(f"  FLASH_ATTENTION: output shape={list(out_flash.shape)}")

    # Efficient (xformers) only
    try:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        ):
            out_eff = F.scaled_dot_product_attention(q, k, v)
        print(f"  EFFICIENT:       output shape={list(out_eff.shape)}")
    except Exception as e:
        print(f"  EFFICIENT:       {type(e).__name__}")

    # Math only (always works)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_math = F.scaled_dot_product_attention(q, k, v)
    print(f"  MATH:            output shape={list(out_math.shape)}")

    # Verify all backends give same result
    all_close = torch.allclose(out_flash, out_math, atol=1e-3)
    print(f"  All backends match: {all_close}")
    print()


# ============ 3. Causal attention ============
def exp_causal():
    if not _cuda():
        return
    print("=" * 60)
    print("3. Causal attention: is_causal vs explicit mask")
    print("=" * 60)

    B, H, S, D = 2, 4, 8, 16
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # Method 1: is_causal=True (FlashAttention native support)
    out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # Method 2: explicit causal mask
    mask = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), diagonal=1)
    out_mask = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    print(f"  is_causal=True:  {out_causal[0, 0, 0].tolist()}")
    print(f"  explicit mask:   {out_mask[0, 0, 0].tolist()}")
    print(f"  Match:           {torch.allclose(out_causal, out_mask, atol=1e-4)}")
    print(
        f"  → is_causal=True is faster (FlashAttention native, no mask materialization)"
    )
    print()


# ============ 4. Memory comparison ============
def exp_memory():
    if not _cuda():
        return
    print("=" * 60)
    print("4. Memory: SDPA vs manual attention")
    print("=" * 60)

    def manual_attention(q, k, v):
        scale = q.size(-1) ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.softmax(dim=-1)
        return attn @ v

    B, H, S, D = 2, 4, 2048, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # SDPA
    torch.cuda.reset_peak_memory_stats()
    out_sdpa = F.scaled_dot_product_attention(q, k, v)
    peak_sdpa = torch.cuda.max_memory_allocated() / 1e6

    # Manual (creates SxS attn matrix)
    torch.cuda.reset_peak_memory_stats()
    out_manual = manual_attention(q, k, v)
    peak_manual = torch.cuda.max_memory_allocated() / 1e6

    print(f"  SDPA peak memory:    {peak_sdpa:.1f} MB")
    print(f"  Manual peak memory: {peak_manual:.1f} MB")
    print(f"  SDPA / Manual:      {peak_sdpa / peak_manual * 100:.0f}%")
    print(f"  → SDPA avoids materializing [S×S] attention matrix (O(S^2) → O(S))")
    print(f"  → Manual needs {S * S * 2 / 1e6:.1f} MB just for the attention matrix")
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "backend": exp_backend,
    "causal": exp_causal,
    "memory": exp_memory,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[sdpa_attention demo] DONE")


if __name__ == "__main__":
    main()
