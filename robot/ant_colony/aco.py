from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


def generate_tsp_cities(num_cities, seed=42):
    rng = np.random.default_rng(seed)
    cities = rng.uniform(0, 100, (num_cities, 2)).astype(np.float32)
    return cities


def compute_distance_matrix(cities):
    n = cities.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = cities[i, 0] - cities[j, 0]
                dy = cities[i, 1] - cities[j, 1]
                dist[i, j] = math.sqrt(dx * dx + dy * dy)
    return dist


def tour_length(tour, dist):
    length = 0.0
    for i in range(len(tour)):
        length += dist[tour[i], tour[(i + 1) % len(tour)]]
    return length


def aco_cpu(
    num_cities,
    num_ants,
    max_iterations,
    alpha=1.0,
    beta=3.0,
    evaporation_rate=0.1,
    q=100.0,
):
    rng = np.random.default_rng(42)
    cities = generate_tsp_cities(num_cities, 42)
    distances = compute_distance_matrix(cities)
    pheromone = np.ones((num_cities, num_cities), dtype=np.float32) * 0.1

    best_length = float("inf")
    best_tour = None

    for it in range(max_iterations):
        pheromone_delta = np.zeros((num_cities, num_cities), dtype=np.float32)
        tours = np.zeros((num_ants, num_cities), dtype=np.int32)
        tour_lengths = np.zeros(num_ants, dtype=np.float32)

        for ant in range(num_ants):
            visited = np.zeros(num_cities, dtype=bool)
            start = ant % num_cities
            current = start
            visited[current] = True
            tours[ant, 0] = current
            length = 0.0

            for step in range(1, num_cities):
                probs = np.zeros(num_cities, dtype=np.float64)
                for c in range(num_cities):
                    if not visited[c]:
                        tau = pheromone[current, c] ** alpha
                        eta = (1.0 / (distances[current, c] + 1e-10)) ** beta
                        probs[c] = tau * eta
                probs /= probs.sum()
                next_city = rng.choice(num_cities, p=probs)
                length += distances[current, next_city]
                current = next_city
                visited[current] = True
                tours[ant, step] = current

            # Return to start
            length += distances[current, start]
            tour_lengths[ant] = length

            if length < best_length:
                best_length = length
                best_tour = tours[ant].copy()

            # Deposit pheromone
            deposit = q / length
            for step in range(num_cities):
                frm = tours[ant, step]
                to = tours[ant, (step + 1) % num_cities]
                pheromone_delta[frm, to] += deposit
                pheromone_delta[to, frm] += deposit

        pheromone = (1 - evaporation_rate) * pheromone + pheromone_delta
        pheromone = np.maximum(pheromone, 1e-10)

    return best_tour, best_length


def compile_aco_extension():
    cuda_source = (Path(__file__).parent / "aco_kernel.cu").read_text()
    cpp_source = """
void aco_iteration_cuda(
    torch::Tensor distances,
    torch::Tensor pheromone,
    torch::Tensor pheromone_delta,
    torch::Tensor tours,
    torch::Tensor tour_lengths,
    torch::Tensor best_length,
    torch::Tensor best_idx,
    torch::Tensor best_tour,
    torch::Tensor curand_states,
    int num_cities, int num_ants,
    float alpha, float beta,
    float evaporation_rate, float q
);
void aco_init_curand(
    torch::Tensor curand_states,
    int num_ants,
    unsigned long long seed
);
"""
    return load_inline(
        name="aco_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["aco_iteration_cuda", "aco_init_curand"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


def aco_cuda(
    num_cities,
    num_ants,
    max_iterations,
    alpha=1.0,
    beta=3.0,
    evaporation_rate=0.1,
    q=100.0,
):
    ext = compile_aco_extension()
    DEVICE = "cuda"

    cities = generate_tsp_cities(num_cities, 42)
    distances = compute_distance_matrix(cities)

    dist_tensor = torch.from_numpy(distances).to(DEVICE)
    pheromone = torch.full(
        (num_cities, num_cities), 0.1, dtype=torch.float32, device=DEVICE
    )
    pheromone_delta = torch.zeros(
        (num_cities, num_cities), dtype=torch.float32, device=DEVICE
    )
    tours = torch.zeros((num_ants, num_cities), dtype=torch.int32, device=DEVICE)
    tour_lengths = torch.empty(num_ants, dtype=torch.float32, device=DEVICE)
    best_length = torch.empty(1, dtype=torch.float32, device=DEVICE)
    best_idx = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    best_tour = torch.zeros(num_cities, dtype=torch.int32, device=DEVICE)

    curand_state_size = 48 * num_ants
    curand_states = torch.empty(curand_state_size, dtype=torch.uint8, device=DEVICE)
    ext.aco_init_curand(curand_states, num_ants, 42)

    global_best_length = float("inf")
    global_best_tour = None

    for it in range(max_iterations):
        ext.aco_iteration_cuda(
            dist_tensor,
            pheromone,
            pheromone_delta,
            tours,
            tour_lengths,
            best_length,
            best_idx,
            best_tour,
            curand_states,
            num_cities,
            num_ants,
            alpha,
            beta,
            evaporation_rate,
            q,
        )

        length_val = best_length.cpu().item()
        if length_val < global_best_length:
            global_best_length = length_val
            global_best_tour = best_tour.cpu().numpy().copy()

    return global_best_tour, global_best_length


def main():
    NUM_CITIES = 100
    NUM_ANTS = 512
    ITERATIONS = 50

    print("=" * 60)
    print("Ant Colony Optimization - TSP")
    print(f"Cities: {NUM_CITIES}, Ants: {NUM_ANTS}, Iterations: {ITERATIONS}")
    print("=" * 60)

    print("\n[CUDA ACO] Running...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    best_tour_cuda, best_length_cuda = aco_cuda(NUM_CITIES, NUM_ANTS, ITERATIONS)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start
    print(f"  Best tour length: {best_length_cuda:.4f}")
    print(f"  Time: {cuda_time:.4f}s")

    print("\n[CPU ACO] Running...")
    start = time.perf_counter()
    best_tour_cpu, best_length_cpu = aco_cpu(NUM_CITIES, NUM_ANTS, ITERATIONS)
    cpu_time = time.perf_counter() - start
    print(f"  Best tour length: {best_length_cpu:.4f}")
    print(f"  Time: {cpu_time:.4f}s")

    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'CUDA':>15} {'CPU':>15}")
    print("-" * 60)
    print(f"{'Time':<25} {cuda_time:>14.4f}s {cpu_time:>14.4f}s")
    print(f"{'Best Length':<25} {best_length_cuda:>14.4f} {best_length_cpu:>14.4f}")
    print(f"{'Speedup':<25} {cpu_time / cuda_time:>14.2f}x {'':>15}")
    print("-" * 60)


if __name__ == "__main__":
    main()
