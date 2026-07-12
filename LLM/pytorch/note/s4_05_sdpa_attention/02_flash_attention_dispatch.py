"""SDPA Attention case study 1: FlashAttention dispatch and selection.

Companion script for sdpa_attention/ directory. Covers:
  1. SDPA backend selection
  2. FlashAttention vs Memory Efficient vs Math
  3. Dispatch key routing

Run:
    python 02_flash_attention_dispatch.py
"""

import sys

import torch


def exp_backend_selection():
    print("=" * 60)
    print("1. SDPA backend selection: FlashAttention vs others")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    # SDPA backends: FlashAttention, MemoryEfficient, Math (default fallback)
    from torch.backends.cuda import SDPBackend, sdp_kernel

    print(f"  Available SDPA backends:")
    for backend in SDPBackend:
        check = torch.backends.cuda.sdp_kernel.__members__
        print(f"    {backend.name:25s} = {backend.value}")

    # Query: what backend would be used?
    q = torch.randn(1, 8, 128, 64, device="cuda")
    k = torch.randn(1, 8, 128, 64, device="cuda")
    v = torch.randn(1, 8, 128, 64, device="cuda")

    try:
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            print(f"\n  FlashAttention selected: shape={list(out.shape)}")
    except Exception as e:
        print(f"\n  FlashAttention failed: {e}")

    # Manual control of which backend to use
    try:
        out_default = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print(f"  Default SDPA: shape={list(out_default.shape)}")
    except Exception as e:
        print(f"  Default SDPA failed: {e}")
    print()


def exp_causal_mask():
    print("=" * 60)
    print("2. FlashAttention with causal mask")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    q = torch.randn(1, 8, 128, 64, device="cuda")
    k = torch.randn(1, 8, 128, 64, device="cuda")
    v = torch.randn(1, 8, 128, 64, device="cuda")

    is_causal = True
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    print(f"  Causal self-attention: shape={list(out.shape)}")
    print(f"  FlashAttention v2 supports is_causal=True as fused operation")
    print()


EXPERIMENTS = {
    "backend": exp_backend_selection,
    "causal": exp_causal_mask,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[sdpa_attention case 1] DONE")


if __name__ == "__main__":
    main()
