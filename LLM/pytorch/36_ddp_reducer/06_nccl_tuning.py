"""DDP Reducer case study 6: NCCL environment tuning.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. NCCL env variables for performance
  2. NVLink vs IB routing
  3. Topology-aware all-reduce

Run:
    python 06_nccl_tuning.py
"""

import sys

import torch


def exp_nccl_env():
    print("=" * 60)
    print("1. NCCL environment variables")
    print("=" * 60)

    env_vars = {
        "NCCL_DEBUG": "INFO (set to WARN or INFO for debugging)",
        "NCCL_IB_DISABLE": "0/1: enable/disable InfiniBand",
        "NCCL_SOCKET_IFNAME": "eth0/ib0: network interface for TCP",
        "NCCL_IB_HCA": "mlx5_0: InfiniBand HCA device",
        "NCCL_NET_GDR_LEVEL": "5: GPUDirect RDMA (0=disabled, 5=full)",
        "NCCL_ALGO": "Ring/Tree: all-reduce algorithm",
        "NCCL_CROSS_NIC": "1: use multiple NICs per node",
        "NCCL_TOPO_FILE": "/path/to/topo.xml: topology file",
        "NCCL_BUFFSIZE": "4194304: buffer size in bytes",
        "NCCL_NSOCKS_PERTHREAD": "4: sockets per thread",
        "NCCL_SOCKET_NTHREADS": "2: socket threads",
    }

    for var, desc in env_vars.items():
        print(f"  {var:25s}: {desc}")

    print(f"\n  Quick performance check:")
    if torch.cuda.is_available():
        if torch.distributed.is_available():
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    n_gpus = dist.get_world_size()
                    print(f"    World size: {n_gpus}")
            except Exception:
                pass
        print(f"    GPU count: {torch.cuda.device_count()}")
    print()


def exp_nvlink_detection():
    print("=" * 60)
    print("2. NVLink topology detection")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    print(f"  NVLink detection from Python:")
    print(f"    nvidia-smi nvlink -s    # NVLink status")
    print(f"    nvidia-smi topo -m       # Topology matrix")
    print(f"")

    n_gpus = torch.cuda.device_count()
    print(f"  P2P access matrix ({n_gpus} GPUs):")
    for i in range(min(n_gpus, 4)):
        row = []
        for j in range(min(n_gpus, 4)):
            if i != j:
                p2p = torch.cuda.can_device_access_peer(i, j)
                row.append("P2P" if p2p else "---")
            else:
                row.append(" X ")
        print(f"    GPU{i}: " + " ".join(row))

    if n_gpus >= 2:
        # Enable P2P (must be symmetric)
        for i in range(n_gpus):
            for j in range(n_gpus):
                if i != j and torch.cuda.can_device_access_peer(i, j):
                    with torch.cuda.device(i):
                        torch.cuda.device(j)  # Check access
    print()


def exp_allreduce_bench():
    print("=" * 60)
    print("3. All-reduce bandwidth estimation")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    # Simulated all-reduce bandwidth (single GPU, no actual distributed)
    import time

    n_gpus = torch.cuda.device_count()
    sizes_mb = [1, 10, 100]

    print(f"  All-reduce bandwidth estimates:")
    print(f"  (Actual needs multi-GPU with NCCL backend)")
    print(f"")

    for size_mb in sizes_mb:
        n_elems = size_mb * 1024 * 1024 // 4
        tensor = torch.randn(n_elems, device="cuda")

        # Just measure GPU bandwidth for reference
        t0 = time.perf_counter()
        _ = tensor * 2
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  {size_mb:3d} MB elementwise: {elapsed*1000:.3f} ms (GPU compute bandwidth)")

    print(f"\n  NCCL all-reduce is bounded by:")
    print(f"    NVLink: ~300 GB/s (intra-node)")
    print(f"    IB HDR: ~200 GB/s (inter-node)")
    print(f"    EDR:    ~100 GB/s (inter-node)")
    print()


EXPERIMENTS = {
    "env": exp_nccl_env,
    "nvlink": exp_nvlink_detection,
    "bench": exp_allreduce_bench,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 6] DONE")


if __name__ == "__main__":
    main()
