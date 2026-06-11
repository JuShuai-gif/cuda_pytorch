#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>

__device__ float rastrigin(const float* x, int dim) {
    float sum = 10.0f * dim;
    for (int d = 0; d < dim; d++) {
        sum += x[d] * x[d] - 10.0f * cosf(2.0f * M_PI * x[d]);
    }
    return sum;
}

__global__ void init_curand_kernel(
    curandState* states,
    int total_threads,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_threads) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void evaluate_fitness_kernel(
    const float* population,
    float* fitness,
    int pop_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pop_size) {
        fitness[idx] = rastrigin(&population[idx * dim], dim);
    }
}

__global__ void tournament_selection_kernel(
    int* selected_indices,
    const float* fitness,
    curandState* states,
    int pop_size,
    int tournament_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];
    int best_idx = -1;
    float best_fitness = INFINITY;

    for (int t = 0; t < tournament_size; t++) {
        int candidate = (int)(curand_uniform(&local_state) * pop_size);
        if (candidate >= pop_size) candidate = pop_size - 1;
        if (fitness[candidate] < best_fitness) {
            best_fitness = fitness[candidate];
            best_idx = candidate;
        }
    }

    selected_indices[idx] = best_idx;
    states[idx] = local_state;
}

__global__ void crossover_mutation_kernel(
    float* new_population,
    const float* old_population,
    const int* selected_indices,
    curandState* states,
    int pop_size,
    int dim,
    float crossover_rate,
    float mutation_rate,
    float mutation_std,
    float bound_min,
    float bound_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];

    int parent1_idx = selected_indices[idx];
    int partner_idx = selected_indices[(idx + pop_size / 2) % pop_size];

    for (int d = 0; d < dim; d++) {
        float gene;
        if (curand_uniform(&local_state) < crossover_rate) {
            float alpha = curand_uniform(&local_state);
            gene = alpha * old_population[parent1_idx * dim + d]
                 + (1.0f - alpha) * old_population[partner_idx * dim + d];
        } else {
            gene = old_population[parent1_idx * dim + d];
        }

        if (curand_uniform(&local_state) < mutation_rate) {
            float noise = curand_normal(&local_state) * mutation_std;
            gene += noise;
        }

        gene = fminf(fmaxf(gene, bound_min), bound_max);
        new_population[idx * dim + d] = gene;
    }

    states[idx] = local_state;
}

__global__ void copy_elites_kernel(
    float* new_population,
    const float* old_population,
    const int* elite_indices,
    int dim,
    int elite_count
) {
    int elite_id = blockIdx.x;
    int d = threadIdx.x;
    if (elite_id >= elite_count || d >= dim) return;

    int src_idx = elite_indices[elite_id];
    new_population[elite_id * dim + d] = old_population[src_idx * dim + d];
}

void ga_generation_cuda(
    torch::Tensor population,
    torch::Tensor fitness,
    torch::Tensor new_population,
    torch::Tensor selected_indices,
    torch::Tensor elite_indices_gpu,
    torch::Tensor curand_states,
    int pop_size,
    int dim,
    float crossover_rate,
    float mutation_rate,
    float mutation_std,
    float bound_min,
    float bound_max,
    int tournament_size,
    int elite_count
) {
    int threads = 256;
    int blocks = (pop_size + threads - 1) / threads;

    evaluate_fitness_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        population.data_ptr<float>(),
        fitness.data_ptr<float>(),
        pop_size, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    tournament_selection_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        selected_indices.data_ptr<int>(),
        fitness.data_ptr<float>(),
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        pop_size, tournament_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    crossover_mutation_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        new_population.data_ptr<float>(),
        population.data_ptr<float>(),
        selected_indices.data_ptr<int>(),
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        pop_size, dim,
        crossover_rate, mutation_rate, mutation_std,
        bound_min, bound_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Copy elites
    int elite_blocks = elite_count;
    int elite_threads = dim;
    copy_elites_kernel<<<elite_blocks, elite_threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        new_population.data_ptr<float>(),
        population.data_ptr<float>(),
        elite_indices_gpu.data_ptr<int>(),
        dim, elite_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ga_init_curand(
    torch::Tensor curand_states,
    int total_threads,
    unsigned long long seed
) {
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    init_curand_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        total_threads, seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
