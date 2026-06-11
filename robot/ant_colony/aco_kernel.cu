#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>

#define MAX_CITIES 2048

__global__ void init_curand_kernel(
    curandState* states,
    int num_ants,
    unsigned long long seed,
    int base_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_ants) {
        curand_init(seed, idx * base_seed, 0, &states[idx]);
    }
}

__global__ void construct_tours_kernel(
    curandState* states,
    int* tours,
    float* tour_lengths,
    const float* distances,
    const float* pheromone,
    int num_cities,
    int num_ants,
    float alpha,
    float beta
) {
    int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_idx >= num_ants) return;

    curandState local_state = states[ant_idx];
    int start_city = ant_idx % num_cities;

    // Visited flags per ant (using shared memory for the visited array)
    extern __shared__ char shared_visited[];
    char* visited = shared_visited + threadIdx.x * num_cities;

    for (int c = 0; c < num_cities; c++) {
        visited[c] = 0;
    }

    int current = start_city;
    visited[current] = 1;
    tours[ant_idx * num_cities + 0] = current;
    float total_length = 0.0f;

    for (int step = 1; step < num_cities; step++) {
        float probabilities[MAX_CITIES];
        float prob_sum = 0.0f;

        for (int c = 0; c < num_cities; c++) {
            if (visited[c]) {
                probabilities[c] = 0.0f;
            } else {
                float tau = powf(pheromone[current * num_cities + c], alpha);
                float eta = powf(1.0f / (distances[current * num_cities + c] + 1e-10f), beta);
                probabilities[c] = tau * eta;
                prob_sum += probabilities[c];
            }
        }

        // Roulette wheel selection
        float r = curand_uniform(&local_state) * prob_sum;
        float cumulative = 0.0f;
        int next_city = -1;

        for (int c = 0; c < num_cities; c++) {
            if (!visited[c]) {
                cumulative += probabilities[c];
                if (cumulative >= r) {
                    next_city = c;
                    break;
                }
            }
        }

        // Fallback: pick first unvisited
        if (next_city == -1) {
            for (int c = 0; c < num_cities; c++) {
                if (!visited[c]) { next_city = c; break; }
            }
        }

        total_length += distances[current * num_cities + next_city];
        current = next_city;
        visited[current] = 1;
        tours[ant_idx * num_cities + step] = current;
    }

    // Return to start
    total_length += distances[current * num_cities + start_city];
    tour_lengths[ant_idx] = total_length;
    states[ant_idx] = local_state;
}

__global__ void update_pheromone_kernel(
    float* pheromone,
    const float* pheromone_delta,
    float evaporation_rate,
    int num_cities
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_cities * num_cities;

    if (idx < total) {
        pheromone[idx] = (1.0f - evaporation_rate) * pheromone[idx] + pheromone_delta[idx];
        if (pheromone[idx] < 1e-10f) pheromone[idx] = 1e-10f;
    }
}

__global__ void deposit_pheromone_kernel(
    float* pheromone,
    const int* tours,
    const float* tour_lengths,
    int num_cities,
    int num_ants,
    float q
) {
    int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_idx >= num_ants) return;

    float deposit = q / tour_lengths[ant_idx];

    for (int step = 0; step < num_cities; step++) {
        int from = tours[ant_idx * num_cities + step];
        int to = tours[ant_idx * num_cities + (step + 1) % num_cities];
        atomicAdd(&pheromone[from * num_cities + to], deposit);
        atomicAdd(&pheromone[to * num_cities + from], deposit);
    }
}

__global__ void find_best_tour_kernel(
    const float* tour_lengths,
    float* best_length,
    int* best_idx,
    int num_ants
) {
    __shared__ float shared_len[256];
    __shared__ int shared_idx[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_len[tid] = (idx < num_ants) ? tour_lengths[idx] : INFINITY;
    shared_idx[tid] = (idx < num_ants) ? idx : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_len[tid + s] < shared_len[tid]) {
                shared_len[tid] = shared_len[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && shared_idx[tid] >= 0 && shared_len[tid] < *best_length) {
        float candidate = shared_len[tid];
        int* best_len_int = (int*)best_length;
        int cand_int = __float_as_int(candidate);
        int old = atomicMin(best_len_int, cand_int);
        if (old > cand_int) {
            *best_idx = shared_idx[tid];
        }
    }
}

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
    int num_cities,
    int num_ants,
    float alpha,
    float beta,
    float evaporation_rate,
    float q
) {
    int threads = 256;
    int ant_blocks = (num_ants + threads - 1) / threads;
    int city_blocks = (num_cities * num_cities + threads - 1) / threads;

    // Reset pheromone delta
    pheromone_delta.zero_();

    // Construct tours
    size_t shared_mem = threads * num_cities * sizeof(char);
    construct_tours_kernel<<<ant_blocks, threads, shared_mem,
        torch::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        tours.data_ptr<int>(),
        tour_lengths.data_ptr<float>(),
        distances.data_ptr<float>(),
        pheromone.data_ptr<float>(),
        num_cities, num_ants, alpha, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Find best tour
    float inf = INFINITY;
    cudaMemcpy(best_length.data_ptr(), &inf, sizeof(float), cudaMemcpyHostToDevice);
    best_idx.zero_();
    find_best_tour_kernel<<<ant_blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        tour_lengths.data_ptr<float>(),
        best_length.data_ptr<float>(),
        best_idx.data_ptr<int>(),
        num_ants);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Deposit pheromone
    deposit_pheromone_kernel<<<ant_blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        pheromone_delta.data_ptr<float>(),
        tours.data_ptr<int>(),
        tour_lengths.data_ptr<float>(),
        num_cities, num_ants, q);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Evaporate + update pheromone
    update_pheromone_kernel<<<city_blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        pheromone.data_ptr<float>(),
        pheromone_delta.data_ptr<float>(),
        evaporation_rate, num_cities);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void aco_init_curand(
    torch::Tensor curand_states,
    int num_ants,
    unsigned long long seed
) {
    int threads = 256;
    int blocks = (num_ants + threads - 1) / threads;
    init_curand_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<curandState*>(curand_states.data_ptr()),
        num_ants, seed, 31337);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
