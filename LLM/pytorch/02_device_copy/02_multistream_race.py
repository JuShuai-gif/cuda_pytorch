"""Device Copy case study 2: multi-stream copy and data race debug.

Companion script for device_copy/device_copy.md. Covers:
  1. Multi-stream execution demo
  2. Stream sync with events
  3. Data race pattern with non_blocking

Run:
    python 02_multistream_race.py
"""

import sys

import torch


def exp_stream_isolation():
    print("=" * 60)
    print("1. Multi-stream execution isolation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # Stream 1: matrix multiply
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = torch.randn(2048, 2048, device="cuda")
    d = torch.randn(2048, 2048, device="cuda")

    with torch.cuda.stream(s1):
        result1 = a @ b  # on s1

    with torch.cuda.stream(s2):
        result2 = c @ d  # on s2

    # Both streams run concurrently
    torch.cuda.synchronize()

    print(f"  Stream 1 matmul done: shape={list(result1.shape)}")
    print(f"  Stream 2 matmul done: shape={list(result2.shape)}")
    print(f"  -> Both kernels ran concurrently")

    # Stream synchronization with events
    s3 = torch.cuda.Stream()
    with torch.cuda.stream(s3):
        x = torch.randn(512, 512, device="cuda")
        y = torch.randn(512, 512, device="cuda")
        z = x @ y
        event = s3.record_event()

    # Wait for s3 to finish before proceeding
    torch.cuda.current_stream().wait_event(event)
    print(f"\n  After wait_event: s3 computation confirmed complete")
    print()


def exp_race_pattern():
    print("=" * 60)
    print("2. Danger: data race with non_blocking copy")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    # UNSAFE: writing to src while non_blocking copy is in progress
    x = torch.randn(4096, 4096)  # pageable (NOT pinned)
    y = torch.zeros(4096, 4096, device="cuda")

    print(f"  Unsafe pattern (pageable memory + non_blocking):")
    print(f"    1. y.copy_(x, non_blocking=True)  # start H2D")
    print(f"    2. x += 1                           # modify source!")
    print(f"    3. ... CUDA may read stale or partly-modified data")

    # Safer: use pinned memory
    x_pinned = torch.randn(4096, 4096, pin_memory=True)

    print(f"\n  Safer pattern (pinned memory + non_blocking):")
    print(f"    1. y.copy_(x_pinned, non_blocking=True)")
    print(f"    2. Other CPU work (don't modify x_pinned)")
    print(f"    3. torch.cuda.synchronize()  # wait for copy to finish")

    # Demonstrate: writing to x_pinned while copy in progress IS still unsafe
    y_copy = torch.zeros(4096, 4096, device="cuda")
    y_copy.copy_(x_pinned, non_blocking=True)
    x_pinned[0] = 999.0  # modifying while copy in progress = UB

    torch.cuda.synchronize()
    # The GPU copy may or may not have the modified value
    first_pinned = y_copy[0, 0].item()
    print(f"\n  After concurrent modify: y_copy[0,0] = {first_pinned}")
    print(f"  -> This may or may NOT be 999.0 (race condition!)")
    print()


def exp_device_guard():
    print("=" * 60)
    print("3. DeviceGuard: device isolation in multi-GPU")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print("  [SKIP] Need 2+ GPUs")
        print(f"  DeviceGuard pattern (RAII):")
        print(f"    with torch.cuda.device(gpu_id):")
        print(f"        x = torch.randn(3, device='cuda')  # on gpu_id")
        print(f"        # CUDAGuard auto-restores previous device")
        return

    # DeviceGuard: RAII pattern for device context
    print(f"  Available GPUs: {n_gpus}")

    with torch.cuda.device(0):
        x0 = torch.randn(3)
        print(f"  default stream GPU 0: {torch.cuda.current_stream().cuda_stream}")

        with torch.cuda.device(1):
            x1 = torch.randn(3)
            stream1 = torch.cuda.current_stream().cuda_stream

        # Back to GPU 0
        print(f"  back on GPU 0: {torch.cuda.current_stream().cuda_stream}")

    print(f"  -> DeviceGuard auto-restores previous device")
    print(f"  -> Same pattern applies to StreamGuard for CUDA streams")
    print()


EXPERIMENTS = {
    "stream": exp_stream_isolation,
    "race": exp_race_pattern,
    "guard": exp_device_guard,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[device_copy case 2] DONE")


if __name__ == "__main__":
    main()
