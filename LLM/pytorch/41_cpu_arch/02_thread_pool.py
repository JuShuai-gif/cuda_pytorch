"""CPU Arch case study 2: threading and parallel.

Run: python 02_thread_pool.py
"""

import sys, time, torch


def exp_thread_control():
    print("=" * 60)
    print("1. Thread count and parallel_for")
    print("=" * 60)
    for n in [1, 2, 4, 8]:
        torch.set_num_threads(n)
        A = torch.randn(1024, 1024)
        B = torch.randn(1024, 1024)
        t0 = time.perf_counter()
        C = A @ B
        t = time.perf_counter() - t0
        gflops = (2 * 1024**3) / t / 1e9
        print(f"  threads={n}: {t*1000:.0f}ms ({gflops:.0f} GFLOPS)")
    torch.set_num_threads(torch.get_num_threads())


def exp_interop():
    print("=" * 60)
    print("2. intra_op vs inter_op parallelism")
    print("=" * 60)
    n_intra = torch.get_num_threads()
    n_inter = torch.get_num_interop_threads()
    print(f"  intra_op threads:  {n_intra}  (parallelize single op)")
    print(f"  inter_op threads:  {n_inter}  (parallelize multiple ops)")
    print(f"  OMP_NUM_THREADS controls intra_op")
    print(f"  torch.set_num_threads() controls intra_op at Python level")


EXPERIMENTS = {"threads": exp_thread_control, "interop": exp_interop}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cpu_arch case 2] DONE")

if __name__ == "__main__": main()
