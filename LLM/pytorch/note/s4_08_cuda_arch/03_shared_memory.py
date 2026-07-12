"""GPU Arch case study 3: shared memory demo.

Run: python 03_shared_memory.py
"""

import sys, time, torch

def exp_shared_mem_sizes():
    print("=" * 60)
    print("1. Shared memory capacities")
    print("=" * 60)
    if not torch.cuda.is_available(): return
    p = torch.cuda.get_device_properties(0)
    smem_block = p.max_shared_memory_per_block / 1024
    smem_block_optin = p.max_shared_memory_per_block_optin / 1024
    smem_sm = p.max_shared_memory_per_multi_processor / 1024
    print(f"  Per block:            {smem_block:.0f} KB")
    print(f"  Per block (opt-in):   {smem_block_optin:.0f} KB (driver opt-in)")
    print(f"  Per SM:               {smem_sm:.0f} KB")
    print(f"  FlashAttention uses shared memory for QK^T tile")

def exp_l1_cache():
    print("=" * 60)
    print("2. L1 cache vs shared memory")
    print("=" * 60)
    print("  GPU L1 cache and shared memory share the same on-chip SRAM")
    print("  Default: 128KB = L1 (64KB) + shared (64KB) per SM")
    print("  Opt-in:   128KB = L1 (28KB) + shared (100KB)")
    print("  Triton can configure this per-kernel")

EXPERIMENTS = {"smem": exp_shared_mem_sizes, "l1": exp_l1_cache}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cuda_arch case 3] DONE")

if __name__ == "__main__": main()
