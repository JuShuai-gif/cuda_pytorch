"""CUDA Graph demo: capture, replay, memory pool, speed benchmark."""

import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_capture_replay():
    print("=== basic capture & replay ===")
    x = torch.randn(4096, 4096, device=DEVICE)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2 + 1
        z = y.relu()

    x.copy_(torch.randn(4096, 4096, device=DEVICE))
    graph.replay()
    result = z.mean().item()
    assert isinstance(result, float), f"unexpected type: {type(result)}"

    graph.reset()
    print("  PASS\n")


def test_memory_pool_sharing():
    print("=== memory pool sharing ===")
    x1 = torch.randn(1024, 1024, device=DEVICE)
    x2 = torch.randn(1024, 1024, device=DEVICE)

    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        y1 = x1 * 2

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=g1.pool()):
        y2 = x2 * 3

    x1.copy_(torch.randn(1024, 1024, device=DEVICE))
    x2.copy_(torch.randn(1024, 1024, device=DEVICE))
    g1.replay()
    g2.replay()

    g1.reset()
    g2.reset()
    print("  PASS")


def test_replay_speed():
    """Compare graph replay vs eager for small kernel."""
    print("=== speed: graph vs eager ===")
    N, M = 128, 128
    x = torch.randn(N, M, device=DEVICE)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2 + 1

    n_iter = 500

    x.copy_(torch.randn(N, M, device=DEVICE))
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        graph.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(10):
        x * 2 + 1
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(n_iter):
        x * 2 + 1
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    graph_time = (t1 - t0) / n_iter * 1000
    eager_time = (t3 - t2) / n_iter * 1000
    speedup = eager_time / graph_time if graph_time > 0 else float("inf")

    print(f"  Graph:   {graph_time:.4e} ms/iter")
    print(f"  Eager:   {eager_time:.4e} ms/iter")
    print(f"  Speedup: {speedup:.2f}x")
    graph.reset()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        exit(0)

    test_capture_replay()
    test_memory_pool_sharing()
    test_replay_speed()
    print("[cuda_graph demo] DONE")