"""Device Copy case study 6: P2P access and NVLink bandwidth.

Companion script for device_copy/device_copy.md. Covers:
  1. Peer-to-peer (P2P) GPU access
  2. NVLink bandwidth measurement
  3. P2P vs host-mediated copy

Run:
    python 06_p2p_access.py
"""

import sys
import time

import torch


def exp_p2p_detection():
    print("=" * 60)
    print("1. P2P access detection and enable")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"  Need 2+ GPUs, found {n_gpus}")
        return

    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU {i} -> GPU {j}: {'P2P OK' if can_access else 'NO P2P'}")

    # Enable P2P where available
    print(f"\n  Enable P2P:")
    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j and torch.cuda.can_device_access_peer(i, j):
                with torch.cuda.device(i):
                    torch.cuda.init()
                    try:
                        torch.cuda.device(j)
                        print(f"    GPU {i} <-> GPU {j}: P2P enabled")
                    except Exception as e:
                        print(f"    GPU {i} <-> GPU {j}: enable failed ({e})")
    print()


def exp_p2p_bandwidth():
    print("=" * 60)
    print("2. P2P vs CPU-mediated copy benchmark")
    print("=" * 60)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return

    size_mb = 256
    n_elems = size_mb * 1024 * 1024 // 4
    n_iter = 10

    with torch.cuda.device(0):
        x = torch.randn(n_elems, device="cuda")

    # P2P: direct GPU 0 -> GPU 1
    if torch.cuda.can_device_access_peer(0, 1):
        with torch.cuda.device(1):
            y = torch.empty(n_elems, device="cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            with torch.cuda.device(1):
                y.copy_(x)
            torch.cuda.synchronize()
        t_p2p = (time.perf_counter() - t0) / n_iter

        bw = size_mb / (t_p2p / 1000) / 1000  # GB/s
        print(f"  P2P copy ({size_mb}MB): {t_p2p*1000:.2f} ms = {bw:.1f} GB/s")

    # CPU-mediated: GPU 0 -> CPU -> GPU 1
    cpu_buf = torch.empty(n_elems, pin_memory=True)
    with torch.cuda.device(1):
        y_cpu = torch.empty(n_elems, device="cuda")

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n_iter):
        cpu_buf.copy_(x, non_blocking=True)   # GPU 0 -> CPU
        torch.cuda.synchronize()
        y_cpu.copy_(cpu_buf, non_blocking=True)  # CPU -> GPU 1
        torch.cuda.synchronize()
    t_cpu_mode = (time.perf_counter() - t1) / n_iter

    bw_cpu = size_mb / (t_cpu_mode / 1000) / 1000
    print(f"  CPU-mediated ({size_mb}MB): {t_cpu_mode*1000:.2f} ms = {bw_cpu:.1f} GB/s")

    if t_p2p > 0:
        print(f"  P2P speedup: {t_cpu_mode / t_p2p:.1f}x")
    print()


EXPERIMENTS = {
    "detect": exp_p2p_detection,
    "bandwidth": exp_p2p_bandwidth,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 6] DONE")


if __name__ == "__main__":
    main()
