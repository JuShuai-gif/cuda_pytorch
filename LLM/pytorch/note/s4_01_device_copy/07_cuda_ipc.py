"""Device Copy case study 7: CUDA IPC and tensor sharing.

Companion script for device_copy/device_copy.md. Covers:
  1. CUDA IPC memory sharing
  2. Tensor sharing across processes
  3. torch.multiprocessing CUDA sharing

Run:
    python 07_cuda_ipc.py
"""

import sys

import torch


def exp_cuda_ipc():
    print("=" * 60)
    print("1. CUDA IPC: inter-process tensor sharing")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    # CUDA IPC allows sharing GPU tensors between processes
    # without copying through CPU

    tensor = torch.randn(1024, 1024, device="cuda")
    handle = tensor.storage()._share_cuda_()

    print(f"  Created IPC handle:")
    print(f"    Original tensor ptr: {tensor.data_ptr()}")
    print(f"    Handle type: {type(handle).__name__}")

    # In another process, reconstruct from handle:
    # other_tensor = torch.cuda.FloatStorage._new_shared_cuda(handle)
    # The two tensors share the same GPU memory

    print(f"\n  CUDA IPC sharing:")
    print(f"    1. Process A: handle = tensor.share_memory_()")
    print(f"    2. Process B: tensor = handle.reconstruct()")
    print(f"    3. Both point to SAME GPU physical memory")
    print(f"")
    print(f"  Benefits:")
    print(f"    - Zero-copy between processes on same node")
    print(f"    - DataLoader uses this for pin_memory transfers")
    print(f"")
    print(f"  Limitations:")
    print(f"    - Same physical GPU only (no cross-device IPC)")
    print(f"    - Memory must be managed (no automatic GC)")
    print(f"    - CUDA IPC is technically CUDA mem pool sharing")
    print()


def exp_multiprocess_sharing():
    print("=" * 60)
    print("2. torch.multiprocessing CUDA tensor sharing")
    print("=" * 60)

    print(f"  PyTorch multiprocessing sharing strategies:")
    print(f"    'file_descriptor': shared memory via fd (default)")
    print(f"    'file_system':     shared memory via /dev/shm files")
    print(f"")

    print(f"  For CUDA tensors:")
    print(f"    torch.multiprocessing.set_sharing_strategy('file_system')")
    print(f"    - Stores CUDA tensor IPC handle in shared memory")
    print(f"    - Child process reads handle, reconstructs GPU tensor")
    print(f"    - Zero GPU-to-GPU copy")
    print(f"")

    print(f"  DataLoader with num_workers:")
    print(f"    - pin_memory=True + num_workers > 0")
    print(f"    - Main process receives pinned CPU tensor")
    print(f"    - IPC shared from worker to main")
    print()


EXPERIMENTS = {
    "ipc": exp_cuda_ipc,
    "multi": exp_multiprocess_sharing,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 7] DONE")


if __name__ == "__main__":
    main()
