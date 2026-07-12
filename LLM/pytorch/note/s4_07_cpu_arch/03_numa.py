"""CPU Arch case study 3: NUMA awareness.

Run: python 03_numa.py
"""

import sys, torch

def exp_numa_check():
    print("=" * 60)
    print("1. NUMA topology detection")
    print("=" * 60)
    print("  lscpu | grep NUMA")
    print("  numactl --hardware")
    print(f"  PyTorch thread count: {torch.get_num_threads()}")
    print(f"  For NUMA-aware training:")
    print(f"    taskset -c 0-15 python train.py   # bind to socket 0 cores")
    print(f"    numactl --membind=0 python train.py # bind memory to socket 0")

def exp_pin_memory_numa():
    print("=" * 60)
    print("2. Pin memory NUMA implications")
    print("=" * 60)
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, pin_memory=True)
        print(f"  pin_memory locks pages -> avoids NUMA migration")
        print(f"  GPU DMA reads pinned pages directly -> bypass CPU")

EXPERIMENTS = {"numa": exp_numa_check, "pin": exp_pin_memory_numa}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cpu_arch case 3] DONE")

if __name__ == "__main__": main()
