"""cuDNN Backend case study: algorithm selection.

Run: python 01_algorithm.py
"""

import sys, time, torch

def exp_algo_selection():
    print("=" * 60)
    print("1. cuDNN algorithm selection (benchmark=True)")
    print("=" * 60)
    if not torch.cuda.is_available(): return
    torch.backends.cudnn.benchmark = True
    conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
    x = torch.randn(16, 64, 56, 56, device="cuda")
    for _ in range(3): _ = conv(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): _ = conv(x)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 50
    print(f"  Conv2d(64,64,3): {t*1000:.3f}ms (algorithm from cache)")

def exp_cudnn_toggle():
    print("=" * 60)
    print("2. Toggle cuDNN: native vs library")
    print("=" * 60)
    if not torch.cuda.is_available(): return
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    x = torch.randn(4, 3, 32, 32, device="cuda")
    torch.backends.cudnn.enabled = True
    t0 = time.perf_counter()
    for _ in range(50): _ = conv(x)
    torch.cuda.synchronize()
    t_cudnn = (time.perf_counter() - t0) / 50
    torch.backends.cudnn.enabled = False
    t1 = time.perf_counter()
    for _ in range(50): _ = conv(x)
    torch.cuda.synchronize()
    t_native = (time.perf_counter() - t1) / 50
    torch.backends.cudnn.enabled = True
    print(f"  cuDNN:  {t_cudnn*1000:.3f}ms")
    print(f"  Native: {t_native*1000:.3f}ms")
    print(f"  Speedup: {t_native/t_cudnn:.1f}x" if t_cudnn > 0 else "")

EXPERIMENTS = {"algo": exp_algo_selection, "toggle": exp_cudnn_toggle}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[cudnn_backend] DONE")

if __name__ == "__main__": main()
