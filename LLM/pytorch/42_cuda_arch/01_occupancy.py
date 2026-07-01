"""GPU Arch case study: occupancy calculator and SM.

Run: python 01_occupancy.py
"""

import sys, time, torch

def exp_gpu_properties():
    print("=" * 60)
    print("1. GPU SM / memory properties")
    print("=" * 60)
    if not torch.cuda.is_available(): return
    p = torch.cuda.get_device_properties(0)
    for attr in ["name", "multi_processor_count", "total_memory",
                  "max_threads_per_multi_processor", "max_shared_memory_per_block_optin",
                  "regs_per_multiprocessor"]:
        val = getattr(p, attr, "?")
        if isinstance(val, int) and val > 1024**2:
            val = f"{val/1024**3:.1f} GB"
        elif isinstance(val, int) and val > 1024:
            val = f"{val/1024:.0f} KB"
        print(f"  {attr}: {val}")

def exp_occupancy_calc():
    print("=" * 60)
    print("2. Occupancy calculator")
    print("=" * 60)
    if not torch.cuda.is_available(): return
    p = torch.cuda.get_device_properties(0)
    threads = int(input("  Threads per block (default 256): ") or 256)
    regs = int(input("  Registers per thread (default 64): ") or 64)
    smem = int(input("  Shared mem per block KB (default 8): ") or 8) * 1024
    max_b = min(p.regs_per_multiprocessor // (threads * regs),
                p.max_shared_memory_per_block_optin // smem,
                p.max_threads_per_multi_processor // threads)
    occ = max_b * threads / p.max_threads_per_multi_processor * 100
    print(f"  Active blocks/SM: {max_b}")
    print(f"  Occupancy: {occ:.0f}%")

EXPERIMENTS = {"props": exp_gpu_properties, "occ": exp_occupancy_calc}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cuda_arch] DONE")

if __name__ == "__main__": main()
