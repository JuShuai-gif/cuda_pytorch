from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


def rastrigin_cpu(x):
    dim = len(x)
    return 10.0 * dim + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in x)


def pso_cpu_sync(num_particles, dim, iterations, bounds=(-5.12, 5.12)):
    rng = np.random.default_rng(42)
    bound_min, bound_max = bounds

    positions = rng.uniform(bound_min, bound_max, (num_particles, dim)).astype(
        np.float32
    )
    velocities = rng.uniform(-1, 1, (num_particles, dim)).astype(np.float32)
    personal_best_pos = positions.copy()
    personal_best_fitness = np.array([rastrigin_cpu(p) for p in positions])

    best_idx = np.argmin(personal_best_fitness)
    global_best_pos = personal_best_pos[best_idx].copy()
    global_best_fitness = personal_best_fitness[best_idx]

    w, c1, c2 = 0.7298, 1.49618, 1.49618
    v_max = (bound_max - bound_min) * 0.2

    for it in range(iterations):
        # Evaluate all particles first
        fitness = np.array([rastrigin_cpu(p) for p in positions])

        # Update personal bests and global best
        for i in range(num_particles):
            if fitness[i] < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness[i]
                personal_best_pos[i] = positions[i].copy()
                if fitness[i] < global_best_fitness:
                    global_best_fitness = fitness[i]
                    global_best_pos = positions[i].copy()

        # Update all particles using the same global best
        for i in range(num_particles):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_pos[i] - positions[i])
                + c2 * r2 * (global_best_pos - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], bound_min, bound_max)

    return global_best_pos, global_best_fitness


def compile_pso_extension():
    cuda_source = (Path(__file__).parent / "pso_kernel.cu").read_text()
    cpp_source = """
void pso_iteration_cuda(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor personal_best_pos,
    torch::Tensor personal_best_fitness,
    torch::Tensor global_best_pos,
    torch::Tensor global_best_fitness,
    torch::Tensor current_fitness,
    torch::Tensor block_best_fitness,
    torch::Tensor block_best_idx,
    torch::Tensor global_best_idx,
    torch::Tensor curand_states,
    int num_particles, int dim,
    float w, float c1, float c2,
    float v_max, float bound_min, float bound_max
);
void pso_init_curand(
    torch::Tensor curand_states,
    int num_particles,
    unsigned long long seed
);
"""
    return load_inline(
        name="pso_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["pso_iteration_cuda", "pso_init_curand"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


def pso_cuda(num_particles, dim, iterations, bounds=(-5.12, 5.12)):
    ext = compile_pso_extension()
    DEVICE = "cuda"
    bound_min, bound_max = bounds

    w, c1, c2 = 0.7298, 1.49618, 1.49618
    v_max = (bound_max - bound_min) * 0.2

    rng = np.random.default_rng(42)
    positions_np = rng.uniform(bound_min, bound_max, (num_particles, dim)).astype(
        np.float32
    )
    velocities_np = rng.uniform(-1, 1, (num_particles, dim)).astype(np.float32)
    init_fitness_np = np.array(
        [rastrigin_cpu(p) for p in positions_np], dtype=np.float32
    )

    threads = 256
    blocks = (num_particles + threads - 1) // threads

    positions = torch.from_numpy(positions_np).to(DEVICE)
    velocities = torch.from_numpy(velocities_np).to(DEVICE)
    current_fitness = torch.empty(num_particles, dtype=torch.float32, device=DEVICE)
    personal_best_pos = positions.clone()
    personal_best_fitness = torch.from_numpy(init_fitness_np).to(DEVICE)

    best_idx = torch.argmin(personal_best_fitness).item()
    global_best_pos = personal_best_pos[best_idx].clone().to(DEVICE)
    global_best_fitness = personal_best_fitness[best_idx].clone().to(DEVICE)
    global_best_idx = torch.tensor([best_idx], dtype=torch.int32, device=DEVICE)

    block_best_fitness = torch.empty(blocks, dtype=torch.float32, device=DEVICE)
    block_best_idx = torch.empty(blocks, dtype=torch.int32, device=DEVICE)

    curand_state_size = 48 * num_particles
    curand_states = torch.empty(curand_state_size, dtype=torch.uint8, device=DEVICE)
    ext.pso_init_curand(curand_states, num_particles, 42)

    torch.cuda.synchronize()

    for it in range(iterations):
        ext.pso_iteration_cuda(
            positions,
            velocities,
            personal_best_pos,
            personal_best_fitness,
            global_best_pos,
            global_best_fitness,
            current_fitness,
            block_best_fitness,
            block_best_idx,
            global_best_idx,
            curand_states,
            num_particles,
            dim,
            w,
            c1,
            c2,
            v_max,
            bound_min,
            bound_max,
        )

    torch.cuda.synchronize()
    return global_best_pos.cpu().numpy(), global_best_fitness.item()


def main():
    NUM_PARTICLES = 8192
    DIM = 30
    ITERATIONS = 100
    BOUNDS = (-5.12, 5.12)

    print("=" * 60)
    print("Particle Swarm Optimization - Rastrigin Function")
    print(f"Particles: {NUM_PARTICLES}, Dim: {DIM}, Iterations: {ITERATIONS}")
    print("=" * 60)

    print("\n[CUDA PSO] Running...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    best_pos_cuda, best_fitness_cuda = pso_cuda(NUM_PARTICLES, DIM, ITERATIONS, BOUNDS)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start
    print(f"  Best fitness: {best_fitness_cuda:.6f}")
    print(f"  Time: {cuda_time:.4f}s")

    print("\n[CPU PSO Sync] Running...")
    start = time.perf_counter()
    best_pos_cpu, best_fitness_cpu = pso_cpu_sync(
        NUM_PARTICLES, DIM, ITERATIONS, BOUNDS
    )
    cpu_time = time.perf_counter() - start
    print(f"  Best fitness: {best_fitness_cpu:.6f}")
    print(f"  Time: {cpu_time:.4f}s")

    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'CUDA':>15} {'CPU':>15}")
    print("-" * 60)
    print(f"{'Time':<25} {cuda_time:>14.4f}s {cpu_time:>14.4f}s")
    print(f"{'Best Fitness':<25} {best_fitness_cuda:>14.6f} {best_fitness_cpu:>14.6f}")
    print(f"{'Speedup':<25} {cpu_time / cuda_time:>14.2f}x {'':>15}")
    print("-" * 60)


if __name__ == "__main__":
    main()
