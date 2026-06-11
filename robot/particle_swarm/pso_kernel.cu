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
    int n,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void evaluate_and_find_best_kernel(
    const float* positions,
    float* current_fitness,
    float* block_best_fitness,
    int* block_best_idx,
    int num_particles,
    int dim
) {
    __shared__ float s_fit[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Evaluate fitness
    float my_fit = rastrigin(&positions[gid * dim], dim);
    if (gid < num_particles) {
        current_fitness[gid] = my_fit;
    }

    // Block-level reduction to find best in this block
    s_fit[tid] = (gid < num_particles) ? my_fit : INFINITY;
    s_idx[tid] = (gid < num_particles) ? gid : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            if (s_fit[tid + s] < s_fit[tid]) {
                s_fit[tid] = s_fit[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && s_idx[tid] >= 0) {
        block_best_fitness[blockIdx.x] = s_fit[tid];
        block_best_idx[blockIdx.x] = s_idx[tid];
    }
}

// Second reduction: find global best across blocks, update if better
__global__ void reduce_blocks_kernel(
    const float* block_best_fitness,
    const int* block_best_idx,
    float* global_best_fitness,
    int* global_best_idx,
    int num_blocks
) {
    __shared__ float s_fit[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;

    s_fit[tid] = (tid < num_blocks) ? block_best_fitness[tid] : INFINITY;
    s_idx[tid] = (tid < num_blocks) ? block_best_idx[tid] : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            if (s_fit[tid + s] < s_fit[tid]) {
                s_fit[tid] = s_fit[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && s_idx[0] >= 0) {
        float current_global = *global_best_fitness;
        if (s_fit[0] < current_global) {
            *global_best_fitness = s_fit[0];
            *global_best_idx = s_idx[0];
        }
    }
}

// Copy best particle's position to global_best_pos
__global__ void copy_global_best_pos_kernel(
    const float* positions,
    float* global_best_pos,
    const int* global_best_idx,
    int dim
) {
    int best_idx = *global_best_idx;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) {
        global_best_pos[d] = positions[best_idx * dim + d];
    }
}

__global__ void update_particles_kernel(
    float* positions,
    float* velocities,
    float* personal_best_pos,
    float* personal_best_fitness,
    const float* global_best_pos,
    const float* current_fitness,
    curandState* states,
    int num_particles,
    int dim,
    float w, float c1, float c2,
    float v_max,
    float bound_min,
    float bound_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    curandState local_state = states[idx];
    float cur_fit = current_fitness[idx];
    float p_best_fit = personal_best_fitness[idx];

    // Update personal best
    if (cur_fit < p_best_fit) {
        personal_best_fitness[idx] = cur_fit;
        for (int d = 0; d < dim; d++) {
            personal_best_pos[idx * dim + d] = positions[idx * dim + d];
        }
    }

    // Update velocity and position
    for (int d = 0; d < dim; d++) {
        float r1 = curand_uniform(&local_state);
        float r2 = curand_uniform(&local_state);

        float v = velocities[idx * dim + d];
        float p_best = personal_best_pos[idx * dim + d];
        float g_best = global_best_pos[d];
        float pos = positions[idx * dim + d];

        v = w * v + c1 * r1 * (p_best - pos) + c2 * r2 * (g_best - pos);

        if (v > v_max) v = v_max;
        if (v < -v_max) v = -v_max;

        velocities[idx * dim + d] = v;
        pos += v;

        if (pos > bound_max) pos = bound_max;
        if (pos < bound_min) pos = bound_min;

        positions[idx * dim + d] = pos;
    }

    states[idx] = local_state;
}

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
    int num_particles,
    int dim,
    float w, float c1, float c2,
    float v_max,
    float bound_min,
    float bound_max
) {
    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;

    // Step 1: evaluate fitness + block reduction
    evaluate_and_find_best_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        positions.data_ptr<float>(),
        current_fitness.data_ptr<float>(),
        block_best_fitness.data_ptr<float>(),
        block_best_idx.data_ptr<int>(),
        num_particles, dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 2: reduce across blocks to find global best index
    reduce_blocks_kernel<<<1, 256, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        block_best_fitness.data_ptr<float>(),
        block_best_idx.data_ptr<int>(),
        global_best_fitness.data_ptr<float>(),
        global_best_idx.data_ptr<int>(),
        blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 3: copy global best position
    int pos_blocks = (dim + threads - 1) / threads;
    copy_global_best_pos_kernel<<<pos_blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        positions.data_ptr<float>(),
        global_best_pos.data_ptr<float>(),
        global_best_idx.data_ptr<int>(),
        dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 4: update velocities, positions, and personal bests
    update_particles_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        personal_best_pos.data_ptr<float>(),
        personal_best_fitness.data_ptr<float>(),
        global_best_pos.data_ptr<float>(),
        current_fitness.data_ptr<float>(),
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        num_particles, dim,
        w, c1, c2, v_max, bound_min, bound_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pso_init_curand(
    torch::Tensor curand_states,
    int num_particles,
    unsigned long long seed
) {
    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;
    init_curand_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        num_particles, seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
