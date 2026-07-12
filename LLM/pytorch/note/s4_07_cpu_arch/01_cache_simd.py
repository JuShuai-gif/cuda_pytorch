"""CPU Arch case study: cache line, SIMD, alignment.

Run: python 01_cache_simd.py
"""

import sys, time, torch


def exp_cache_line():
    print("=" * 60)
    print("1. Row-major vs column-major: cache line impact")
    print("=" * 60)
    N = 4096
    x = torch.randn(N, N)
    t0 = time.perf_counter()
    s_row = x.sum(dim=1)  # stride-1 contiguous
    t_row = time.perf_counter() - t0
    x_t = x.t().contiguous()
    t1 = time.perf_counter()
    s_col = x_t.sum(dim=1)
    t_col = time.perf_counter() - t1
    print(f"  Row-major (stride=1): {t_row*1000:.1f}ms")
    print(f"  Col-accessed tight:    {t_col*1000:.1f}ms")


def exp_simd_detect():
    print("=" * 60)
    print("2. SIMD vectorization check")
    print("=" * 60)
    A = torch.randn(2048, 2048)
    B = torch.randn(2048, 2048)
    t0 = time.perf_counter()
    C = A @ B
    t = time.perf_counter() - t0
    gflops = (2 * 2048**3) / t / 1e9
    print(f"  matmul 2048x2048: {t*1000:.1f}ms ({gflops:.0f} GFLOPS)")
    print(f"  Uses AVX/AVX-512 under MKL/OpenBLAS")


def exp_alignment():
    print("=" * 60)
    print("3. Memory alignment")
    print("=" * 60)
    for shape in [(1024,), (1023,), (100, 100)]:
        x = torch.randn(*shape)
        aligned = x.data_ptr() % 64 == 0
        print(f"  shape={list(shape)}, ptr={x.data_ptr():#x}, 64-byte aligned={aligned}")


EXPERIMENTS = {"cache": exp_cache_line, "simd": exp_simd_detect, "align": exp_alignment}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cpu_arch] DONE")

if __name__ == "__main__": main()
