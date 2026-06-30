"""CUDA Stream, Event & CUDA Graph demo.

Companion script for cuda_stream/cuda_stream.md. Covers:
  1. stream:         create non-default stream, async execution
  2. event:          record event, wait_event, elapsed_time
  3. cuda graph:     capture, replay, memory pool
  4. synchronize:    stream-level vs device-level sync

Run:
    python test1.py                # full demo (needs CUDA)
    python test1.py stream         # stream demo
    python test1.py event          # event & timing demo
    python test1.py graph          # CUDA Graph demo
"""

import sys
import time

import torch


def _check_cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return False
    return True


# ============ 1. Stream: async execution on multiple streams ============
def exp_stream():
    if not _check_cuda():
        return

    print("=" * 60)
    print("1. Stream: async execution on multiple streams")
    print("=" * 60)

    N = 64 * 1024 * 1024 // 4  # 64M elements
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # Stream 1: copy a large tensor
    x1 = torch.randn(N, device="cuda")
    with torch.cuda.stream(s1):
        y1 = x1 * 2 + 1

    # Stream 2: another computation in parallel
    x2 = torch.randn(N, device="cuda")
    with torch.cuda.stream(s2):
        y2 = x2 * 3 - 1

    # Both streams must finish before we read results
    torch.cuda.synchronize()

    print(f"  Stream 1 result: mean={y1.mean().item():.3f}")
    print(f"  Stream 2 result: mean={y2.mean().item():.3f}")
    print("  -> Two streams execute in parallel on the GPU")
    print()

    # Verify default stream
    default = torch.cuda.current_stream()
    print(f"  Default stream:  id={default.stream_id}, device={default.device_index}")
    print(f"  Stream 1:        id={s1.stream_id}, device={s1.device_index}")
    print(f"  Stream 2:        id={s2.stream_id}, device={s2.device_index}")

    # Streams from pool: IDs are reused
    s3 = torch.cuda.Stream()
    print(f"  New stream:      id={s3.stream_id}")
    print()


# ============ 2. Event: timing & synchronization ============
def exp_event():
    if not _check_cuda():
        return

    print("=" * 60)
    print("2. Event: GPU timing & cross-stream sync")
    print("=" * 60)

    # Measure GPU kernel time (not CPU launch time)
    x = torch.randn(4096, 4096, device="cuda")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        x = x @ x
    end.record()

    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    print(f"  10 matmul (4096x4096): {elapsed:.3f} ms ({elapsed / 10:.3f} ms each)")

    # Cross-stream sync: make stream A wait for stream B
    s_a = torch.cuda.Stream()
    s_b = torch.cuda.Stream()

    with torch.cuda.stream(s_a):
        data_a = torch.randn(1024, device="cuda") * 2
        event_a = s_a.record_event()

    # Current stream waits for s_a's event before proceeding
    torch.cuda.current_stream().wait_event(event_a)
    # Now we can safely read data_a

    torch.cuda.synchronize()
    print(f"  data_a mean: {data_a.mean().item():.3f}")
    print("  -> wait_event ensures stream ordering without CPU sync")
    print()


# ============ 3. CUDA Graph ============
def exp_graph():
    if not _check_cuda():
        return

    print("=" * 60)
    print("3. CUDA Graph: capture & replay")
    print("=" * 60)

    # Input must have fixed address across replays
    x = torch.randn(4096, 4096, device="cuda")

    graph = torch.cuda.CUDAGraph()

    # --- Capture ---
    with torch.cuda.graph(graph):
        y = x * 2 + 1
        z = y.relu()

    print(f"  Graph captured: 2 ops (mul+add+relu)")

    # --- Replay (no CPU launch overhead) ---
    x.copy_(torch.randn(4096, 4096, device="cuda"))
    torch.cuda.synchronize()

    n_warmup = 5
    n_iter = 100

    # Warmup
    for _ in range(n_warmup):
        graph.replay()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        graph.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Eager baseline
    for _ in range(n_warmup):
        (x * 2 + 1).relu()
    torch.cuda.synchronize()

    t2 = time.perf_counter()
    for _ in range(n_iter):
        (x * 2 + 1).relu()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    graph_time = (t1 - t0) * 1000 / n_iter
    eager_time = (t3 - t2) * 1000 / n_iter

    print(f"  Graph replay:  {graph_time:.4f} ms/iter")
    print(f"  Eager launch:  {eager_time:.4f} ms/iter")
    if eager_time > 0:
        print(
            f"  Speedup:       {eager_time / graph_time:.2f}x (eliminates CPU launch overhead)"
        )
    print("  -> For small kernels, CUDA Graph eliminates ~5-10us launch per op")

    # Memory pool sharing
    x2 = torch.randn(2048, 2048, device="cuda")
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2, pool=graph.pool()):
        y2 = x2 * 3
    print(f"\n  Graph 2 (shared pool): created with graph.pool()")

    # Graph reset
    graph2.reset()
    print(f"  Graph 2 reset: memory freed")
    graph.reset()
    print()


EXPERIMENTS = {
    "stream": exp_stream,
    "event": exp_event,
    "graph": exp_graph,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[cuda_stream demo] DONE")


if __name__ == "__main__":
    main()
