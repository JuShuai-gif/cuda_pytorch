"""GPU Arch case study 2: tensor core benchmark.

Run: python 02_tensor_core.py
"""

import sys, time, torch

def exp_tensor_core():
    print("=" * 60)
    print("1. Tensor Core: fp16 vs fp32 matmul")
    print("=" * 60)
    if not torch.cuda.is_available(): return
    N = 4096
    for dt in [torch.float16, torch.float32]:
        A = torch.randn(N, N, device="cuda", dtype=dt)
        B = torch.randn(N, N, device="cuda", dtype=dt)
        for _ in range(5): C = A @ B
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): C = A @ B
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 20
        tflops = (2 * N**3) / t / 1e12
        print(f"  {str(dt):20s} {t*1000:.1f}ms ({tflops:.1f} TFLOPS)")

def exp_tensor_core_requirements():
    print("=" * 60)
    print("2. Tensor Core requirements")
    print("=" * 60)
    print("  - fp16 or bf16 inputs")
    print("  - M, N, K multiples of 8 (for fp16) or 16 (for bf16)")
    print("  - cuBLAS auto-detects and uses Tensor Cores")
    print("  - torch.compile + Inductor uses Triton matmul (also Tensor Core)")

EXPERIMENTS = {"tc": exp_tensor_core, "req": exp_tensor_core_requirements}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cuda_arch case 2] DONE")

if __name__ == "__main__": main()
