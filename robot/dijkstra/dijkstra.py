from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


def generate_random_graph(num_nodes, density=0.3, seed=42):
    rng = np.random.default_rng(seed)
    graph = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    for i in range(num_nodes):
        graph[i, i] = 0.0
        for j in range(num_nodes):
            if i != j and rng.random() < density:
                graph[i, j] = rng.uniform(1.0, 10.0)
    return graph


def dijkstra_cpu(graph, source):
    n = graph.shape[0]
    dist = np.full(n, np.inf, dtype=np.float32)
    visited = np.zeros(n, dtype=bool)
    dist[source] = 0.0

    for _ in range(n):
        u = -1
        min_dist = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i

        if u == -1:
            break

        visited[u] = True

        for v in range(n):
            if not visited[v] and graph[u, v] < np.inf:
                new_dist = dist[u] + graph[u, v]
                if new_dist < dist[v]:
                    dist[v] = new_dist

    return dist


def compile_dijkstra_extension():
    cuda_source = (Path(__file__).parent / "dijkstra_kernel.cu").read_text()
    cpp_source = """
void dijkstra_cuda_step(
    torch::Tensor dist,
    torch::Tensor graph,
    torch::Tensor updated_flag,
    int num_nodes
);
"""
    return load_inline(
        name="dijkstra_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["dijkstra_cuda_step"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


def dijkstra_cuda(graph, source):
    ext = compile_dijkstra_extension()
    DEVICE = "cuda"
    n = graph.shape[0]

    dist = np.full(n, np.inf, dtype=np.float32)
    dist[source] = 0.0

    dist_tensor = torch.from_numpy(dist).to(DEVICE)
    graph_tensor = torch.from_numpy(graph).to(DEVICE)
    updated_flag = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    for _ in range(n - 1):
        ext.dijkstra_cuda_step(dist_tensor, graph_tensor, updated_flag, n)
        torch.cuda.synchronize()
        if updated_flag.cpu().item() == 0:
            break

    return dist_tensor.cpu().numpy()


def main():
    print("=" * 60)
    print("Dijkstra Shortest Path - Parallel Bellman-Ford Relaxation")
    print("Note: GPU version is O(VE) while CPU Dijkstra is O(V^2).")
    print("GPU wins only for graphs where V is small but E is very large,")
    print("or when solving all-pairs shortest paths simultaneously.")
    print("=" * 60)

    configurations = [
        (500, 0.5, "small-dense"),
        (1000, 0.3, "medium"),
        (2000, 0.1, "large-sparse"),
    ]
    SOURCE = 0

    for num_nodes, density, label in configurations:
        graph_np = generate_random_graph(num_nodes, density, 42)

        print(f"\n{'=' * 60}")
        print(f"Graph [{label}] Nodes={num_nodes} Density={density}")
        print("=" * 60)

        print("\n[CUDA] Running...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist_cuda = dijkstra_cuda(graph_np, SOURCE)
        torch.cuda.synchronize()
        cuda_time = time.perf_counter() - start

        print("[CPU] Running...")
        start = time.perf_counter()
        dist_cpu = dijkstra_cpu(graph_np, SOURCE)
        cpu_time = time.perf_counter() - start

        cuda_finite = np.where(np.isfinite(dist_cuda), dist_cuda, 0)
        cpu_finite = np.where(np.isfinite(dist_cpu), dist_cpu, 0)
        max_diff = np.max(np.abs(cuda_finite - cpu_finite))
        inf_match = np.all(np.isinf(dist_cuda) == np.isinf(dist_cpu))

        print(f"\n  CUDA: time={cuda_time:.4f}s")
        print(f"  CPU:  time={cpu_time:.4f}s")
        print(f"  Speedup: {cpu_time / cuda_time:.2f}x")
        print(f"  Correctness: max_diff={max_diff:.2e} inf_match={inf_match}")


if __name__ == "__main__":
    main()
