from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


def rastrigin_cpu(x):
    dim = len(x)
    return 10.0 * dim + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in x)


def ga_cpu(pop_size, dim, generations, bounds=(-5.12, 5.12)):
    rng = np.random.default_rng(42)
    bound_min, bound_max = bounds

    crossover_rate = 0.85
    mutation_rate = 0.05
    mutation_std = 0.5
    tournament_size = 5
    elite_count = max(1, pop_size // 10)

    population = rng.uniform(bound_min, bound_max, (pop_size, dim)).astype(np.float32)
    fitness = np.array([rastrigin_cpu(p) for p in population])

    for gen in range(generations):
        new_population = np.empty_like(population)

        elite_indices = np.argsort(fitness)[:elite_count]
        new_population[:elite_count] = population[elite_indices]

        for i in range(elite_count, pop_size):
            candidates = rng.integers(0, pop_size, tournament_size)
            best_idx = candidates[np.argmin(fitness[candidates])]

            partner_candidates = rng.integers(0, pop_size, tournament_size)
            partner_idx = partner_candidates[np.argmin(fitness[partner_candidates])]

            child = np.empty(dim, dtype=np.float32)
            for d in range(dim):
                if rng.random() < crossover_rate:
                    alpha = rng.random()
                    child[d] = (
                        alpha * population[best_idx, d]
                        + (1 - alpha) * population[partner_idx, d]
                    )
                else:
                    child[d] = population[best_idx, d]

                if rng.random() < mutation_rate:
                    child[d] += rng.normal(0, mutation_std)
                child[d] = np.clip(child[d], bound_min, bound_max)

            new_population[i] = child

        population = new_population
        fitness = np.array([rastrigin_cpu(p) for p in population])

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]


def compile_ga_extension():
    cuda_source = (Path(__file__).parent / "ga_kernel.cu").read_text()
    cpp_source = """
void ga_generation_cuda(
    torch::Tensor population,
    torch::Tensor fitness,
    torch::Tensor new_population,
    torch::Tensor selected_indices,
    torch::Tensor elite_indices_gpu,
    torch::Tensor curand_states,
    int pop_size, int dim,
    float crossover_rate, float mutation_rate,
    float mutation_std, float bound_min, float bound_max,
    int tournament_size, int elite_count
);
void ga_init_curand(
    torch::Tensor curand_states,
    int total_threads,
    unsigned long long seed
);
"""
    return load_inline(
        name="ga_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["ga_generation_cuda", "ga_init_curand"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


def ga_cuda(pop_size, dim, generations, bounds=(-5.12, 5.12)):
    ext = compile_ga_extension()
    DEVICE = "cuda"
    bound_min, bound_max = bounds

    crossover_rate = 0.85
    mutation_rate = 0.05
    mutation_std = 0.5
    tournament_size = 5
    elite_count = max(1, pop_size // 10)

    rng = np.random.default_rng(42)
    pop_np = rng.uniform(bound_min, bound_max, (pop_size, dim)).astype(np.float32)

    population = torch.from_numpy(pop_np).to(DEVICE)
    new_population = torch.empty_like(population)
    fitness = torch.empty(pop_size, dtype=torch.float32, device=DEVICE)
    selected_indices = torch.empty(pop_size, dtype=torch.int32, device=DEVICE)
    elite_indices_gpu = torch.empty(elite_count, dtype=torch.int32, device=DEVICE)

    curand_state_size = 48 * pop_size
    curand_states = torch.empty(curand_state_size, dtype=torch.uint8, device=DEVICE)
    ext.ga_init_curand(curand_states, pop_size, 42)

    for gen in range(generations):
        ext.ga_generation_cuda(
            population,
            fitness,
            new_population,
            selected_indices,
            elite_indices_gpu,
            curand_states,
            pop_size,
            dim,
            crossover_rate,
            mutation_rate,
            mutation_std,
            bound_min,
            bound_max,
            tournament_size,
            elite_count,
        )

        # Find elite indices on CPU (only fitness values, which is small)
        fit_cpu = fitness.cpu().numpy()
        elite_idx_cpu = np.argsort(fit_cpu)[:elite_count].astype(np.int32)
        elite_indices_gpu.copy_(torch.from_numpy(elite_idx_cpu).to(DEVICE))

        # Swap buffers
        population, new_population = new_population, population

    fitness_cpu = fitness.cpu().numpy()
    pop_cpu = population.cpu().numpy()
    best_idx = np.argmin(fitness_cpu)
    return pop_cpu[best_idx], float(fitness_cpu[best_idx])


def main():
    POP_SIZE = 8192
    DIM = 30
    GENERATIONS = 100
    BOUNDS = (-5.12, 5.12)

    print("=" * 60)
    print("Genetic Algorithm - Rastrigin Function")
    print(f"Pop: {POP_SIZE}, Dim: {DIM}, Generations: {GENERATIONS}")
    print("=" * 60)

    print("\n[CUDA GA] Running...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    best_cuda, fitness_cuda = ga_cuda(POP_SIZE, DIM, GENERATIONS, BOUNDS)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start
    print(f"  Best fitness: {fitness_cuda:.6f}")
    print(f"  Time: {cuda_time:.4f}s")

    print("\n[CPU GA] Running...")
    start = time.perf_counter()
    best_cpu, fitness_cpu = ga_cpu(POP_SIZE, DIM, GENERATIONS, BOUNDS)
    cpu_time = time.perf_counter() - start
    print(f"  Best fitness: {fitness_cpu:.6f}")
    print(f"  Time: {cpu_time:.4f}s")

    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'CUDA':>15} {'CPU':>15}")
    print("-" * 60)
    print(f"{'Time':<25} {cuda_time:>14.4f}s {cpu_time:>14.4f}s")
    print(f"{'Best Fitness':<25} {fitness_cuda:>14.6f} {fitness_cpu:>14.6f}")
    print(f"{'Speedup':<25} {cpu_time / cuda_time:>14.2f}x {'':>15}")
    print("-" * 60)


if __name__ == "__main__":
    main()
