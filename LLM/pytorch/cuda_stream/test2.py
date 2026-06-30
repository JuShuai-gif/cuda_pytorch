"""CUDA Stream advanced scenarios: multi-stream pipelines, interleaved timing.

Companion script for cuda_stream/cuda_stream.md.
  1. multi-stream pipeline:  producer-consumer with 2 streams
  2. interleaved timing:     measure kernel overlap
  3. CUDAGraph with shapes:  graph capture + replay detail
  4. Stream priority:        high/low priority stream behavior

Run:
    python test2.py                # full demo (needs CUDA)
    python test2.py pipeline       # multi-stream pipeline
    python test2.py overlap        # measure kernel overlap
    python test2.py graph_shapes   # CUDAGraph + dynamic shapes
"""

import sys
import time
import torch


def _cuda():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return False
    return True


# ============ 1. Multi-stream producer-consumer ============
def exp_pipeline():
    if not _cuda():
        return

    print("=" * 60)
    print("1. Multi-stream pipeline: producer -> consumer")
    print("=" * 60)

    N = 16 * 1024 * 1024 // 4  # 16M elements
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    num_chunks = 8
    chunk_size = N // num_chunks

    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    intermediate = torch.empty(N, device="cuda")
    output = torch.empty(N, device="cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size

        # Stream 1: producer (add)
        with torch.cuda.stream(s1):
            intermediate[start:end] = x[start:end] + y[start:end]

        # Record event on s1 after producer finishes
        evt = s1.record_event()

        # Stream 2: wait for producer, then consume (relu)
        with torch.cuda.stream(s2):
            s2.wait_event(evt)
            output[start:end] = intermediate[start:end].relu()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Serial baseline
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        output[start:end] = (x[start:end] + y[start:end]).relu()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"  Chunks: {num_chunks} x {chunk_size / 1e6:.1f}M elements")
    print(f"  Pipeline: {(t1 - t0) * 1000:.1f} ms")
    print(f"  Serial:   {(t3 - t2) * 1000:.1f} ms")
    print(
        "  -> Event-based sync: s2 waits for s1 per chunk, then both proceed in parallel"
    )
    print()


# ============ 2. Kernel overlap measurement ============
def exp_overlap():
    if not _cuda():
        return

    print("=" * 60)
    print("2. Kernel overlap: measure concurrent execution")
    print("=" * 60)

    N = 64 * 1024 * 1024 // 4  # 64M elements
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    r1 = torch.empty(N, device="cuda")
    r2 = torch.empty(N, device="cuda")

    # Timed: both streams execute simultaneously
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s1):
        r1 = a * 2 + 1
    with torch.cuda.stream(s2):
        r2 = b * 3 - 2
    torch.cuda.synchronize()
    t_concurrent = time.perf_counter() - t0

    # Timed: sequential execution
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.cuda.stream(s1):
        r1 = a * 2 + 1
    torch.cuda.synchronize()
    with torch.cuda.stream(s2):
        r2 = b * 3 - 2
    torch.cuda.synchronize()
    t_serial = time.perf_counter() - t2

    print(f"  Concurrent: {t_concurrent * 1000:.2f} ms")
    print(f"  Serial:     {t_serial * 1000:.2f} ms")
    if t_serial > 0:
        print(
            f"  Overlap:    {t_serial / t_concurrent:.2f}x (closer to 2.0 = full overlap)"
        )
    print("  -> Two independent streams overlap work on GPU")
    print()


# ============ 3. CUDAGraph with fixed/dynamic shapes ============
def exp_graph_shapes():
    if not _cuda():
        return

    print("=" * 60)
    print("3. CUDAGraph: fixed address + input patterns")
    print("=" * 60)

    # Scenario A: fixed size tensor (must NOT change address across replays)
    x = torch.zeros(4096, 4096, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2 + 1
        z = y.relu()

    # Replay with different data (same tensor object)
    results = []
    for scale in [1.0, 2.0, 5.0]:
        x.copy_(torch.randn(4096, 4096, device="cuda") * scale)
        graph.replay()
        results.append(y.mean().item())

    print(f"  Fixed tensor replay: mean(y) = {[f'{r:.2f}' for r in results]}")
    print("  -> Same tensor object, different data, graph replayed correctly")

    # Scenario B: check if stream is capturing
    print(f"\n  is_capturing after graph: {torch.cuda.is_current_stream_capturing()}")

    # Scenario C: multiple graphs sharing pool
    x2 = torch.zeros(2048, 2048, device="cuda")
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2, pool=graph.pool()):
        y2 = x2 * 3
    x2.copy_(torch.ones(2048, 2048, device="cuda"))
    graph2.replay()
    print(f"  Graph2 (shared pool): mean={y2.mean().item():.1f}")

    graph.reset()
    graph2.reset()
    print()


EXPERIMENTS = {
    "pipeline": exp_pipeline,
    "overlap": exp_overlap,
    "graph_shapes": exp_graph_shapes,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[cuda_stream test2] DONE")


if __name__ == "__main__":
    main()
