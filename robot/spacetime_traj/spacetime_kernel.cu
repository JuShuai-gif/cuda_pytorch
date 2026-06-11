#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>

__device__ float eval_poly5(const float* c, float t) {
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
}
__device__ float eval_poly5_dot(const float* c, float t) {
    return c[1] + t * (2.0f * c[2] + t * (3.0f * c[3] + t * (4.0f * c[4] + t * 5.0f * c[5])));
}
__device__ float eval_poly5_ddot(const float* c, float t) {
    return 2.0f * c[2] + t * (6.0f * c[3] + t * (12.0f * c[4] + t * 20.0f * c[5]));
}
__device__ float eval_poly5_dddot(const float* c, float t) {
    return 6.0f * c[3] + t * (24.0f * c[4] + t * 60.0f * c[5]);
}
__device__ float eval_poly4(const float* c, float t) {
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])));
}
__device__ float eval_poly4_dot(const float* c, float t) {
    return c[1] + t * (2.0f * c[2] + t * (3.0f * c[3] + t * 4.0f * c[4]));
}
__device__ float eval_poly4_ddot(const float* c, float t) {
    return 2.0f * c[2] + t * (6.0f * c[3] + t * 12.0f * c[4]);
}
__device__ float eval_poly4_dddot(const float* c, float t) {
    return 6.0f * c[3] + t * 24.0f * c[4];
}

__global__ void evaluate_trajectory_kernel(
    const float* coeffs,          // [num_candidates, 11]: a0..a5, b0..b4
    const float* T_values,        // [num_candidates]: time horizon per candidate
    const float* v_targets,       // [num_candidates]: target velocity per candidate
    const float* obstacles,       // [num_obstacles, 3]: s, d, radius
    float* total_cost,            // [num_candidates] output
    float* cost_components,       // [num_candidates * 9] detailed breakdown
    int num_candidates,
    int num_obstacles,
    int num_time_steps,
    float w_jerk, float w_lat_accel, float w_lon_accel,
    float w_ref_dev, float w_obstacle, float w_vel_target,
    float w_time, float w_curvature, float w_centripetal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    const float* lat_c  = &coeffs[idx * 11];
    const float* lon_c  = &coeffs[idx * 11 + 6];
    float T             = T_values[idx];
    float v_target      = v_targets[idx];

    if (T <= 0.0f) T = 1.0f;

    float jerk_int       = 0.0f;
    float lat_accel_int  = 0.0f;
    float lon_accel_int  = 0.0f;
    float ref_dev_int    = 0.0f;
    float obstacle_cost  = 0.0f;
    float max_curvature  = 0.0f;
    float max_centrip    = 0.0f;

    for (int k = 0; k < num_time_steps; k++) {
        float t = T * (float)k / (float)(num_time_steps - 1);

        float d       = eval_poly5(lat_c, t);
        float d_dot   = eval_poly5_dot(lat_c, t);
        float d_ddot  = eval_poly5_ddot(lat_c, t);
        float d_dddot = eval_poly5_dddot(lat_c, t);

        float s       = eval_poly4(lon_c, t);
        float s_dot   = eval_poly4_dot(lon_c, t);
        float s_ddot  = eval_poly4_ddot(lon_c, t);
        float s_dddot = eval_poly4_dddot(lon_c, t);

        jerk_int      += d_dddot * d_dddot + s_dddot * s_dddot;
        lat_accel_int += d_ddot * d_ddot;
        lon_accel_int += s_ddot * s_ddot;
        ref_dev_int   += d * d;

        float denom = 1.0f + d_dot * d_dot;
        float kappa = fabsf(d_ddot) / (denom * sqrtf(denom) + 1e-6f);
        if (kappa > max_curvature) max_curvature = kappa;
        float centrip = s_dot * s_dot * kappa;
        if (centrip > max_centrip) max_centrip = centrip;

        for (int o = 0; o < num_obstacles; o++) {
            float obs_s = obstacles[o * 3 + 0];
            float obs_d = obstacles[o * 3 + 1];
            float obs_r = obstacles[o * 3 + 2];
            float ds = s - obs_s;
            float dd = d - obs_d;
            float dist = sqrtf(ds * ds + dd * dd);
            if (dist < obs_r * 3.0f) {
                obstacle_cost += expf(-dist / (obs_r + 1e-6f));
            }
        }
    }

    float inv_n = 1.0f / (float)num_time_steps;
    jerk_int      *= inv_n;
    lat_accel_int *= inv_n;
    lon_accel_int *= inv_n;
    ref_dev_int   *= inv_n;
    obstacle_cost *= inv_n;

    float s_dot_final = eval_poly4_dot(lon_c, T);
    float vel_dev = (s_dot_final - v_target) * (s_dot_final - v_target);

    float total = 0.0f;
    total += w_jerk       * jerk_int;
    total += w_lat_accel  * lat_accel_int;
    total += w_lon_accel  * lon_accel_int;
    total += w_ref_dev    * ref_dev_int;
    total += w_obstacle   * obstacle_cost;
    total += w_vel_target * vel_dev;
    total += w_time       * T;
    if (max_curvature > 0.3f)  total += w_curvature   * max_curvature;
    if (max_centrip > 3.0f)    total += w_centripetal * max_centrip;

    total_cost[idx] = total;

    int base = idx * 9;
    cost_components[base + 0] = jerk_int;
    cost_components[base + 1] = lat_accel_int;
    cost_components[base + 2] = lon_accel_int;
    cost_components[base + 3] = ref_dev_int;
    cost_components[base + 4] = obstacle_cost;
    cost_components[base + 5] = max_curvature;
    cost_components[base + 6] = max_centrip;
    cost_components[base + 7] = vel_dev;
    cost_components[base + 8] = T;
}

__global__ void find_best_candidate_kernel(
    const float* total_cost,
    float* block_best_cost,
    int* block_best_idx,
    int num_candidates
) {
    __shared__ float s_cost[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    s_cost[tid] = (gid < num_candidates) ? total_cost[gid] : INFINITY;
    s_idx[tid] = (gid < num_candidates) ? gid : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            if (s_cost[tid + s] < s_cost[tid]) {
                s_cost[tid] = s_cost[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && s_idx[tid] >= 0) {
        block_best_cost[blockIdx.x] = s_cost[tid];
        block_best_idx[blockIdx.x] = s_idx[tid];
    }
}

__global__ void global_reduce_kernel(
    const float* block_best_cost,
    const int* block_best_idx,
    float* global_best_cost,
    int* global_best_idx,
    int num_blocks
) {
    __shared__ float s_cost[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;

    s_cost[tid] = (tid < num_blocks) ? block_best_cost[tid] : INFINITY;
    s_idx[tid] = (tid < num_blocks) ? block_best_idx[tid] : -1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            if (s_cost[tid + s] < s_cost[tid]) {
                s_cost[tid] = s_cost[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *global_best_cost = s_cost[0];
        *global_best_idx = s_idx[0];
    }
}

void spacetime_eval_cuda(
    torch::Tensor coeffs,          // [n, 11]
    torch::Tensor T_values,        // [n]
    torch::Tensor v_targets,       // [n]
    torch::Tensor obstacles,       // [m, 3]
    torch::Tensor total_cost,      // [n]
    torch::Tensor cost_components, // [n * 9]
    torch::Tensor block_best_cost, // [num_blocks]
    torch::Tensor block_best_idx,  // [num_blocks]
    torch::Tensor global_best_cost,// [1]
    torch::Tensor global_best_idx, // [1]
    int num_candidates,
    int num_obstacles,
    int num_time_steps,
    float w_jerk, float w_lat_accel, float w_lon_accel,
    float w_ref_dev, float w_obstacle, float w_vel_target,
    float w_time, float w_curvature, float w_centripetal
) {
    int threads = 256;
    int blocks = (num_candidates + threads - 1) / threads;

    evaluate_trajectory_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        coeffs.data_ptr<float>(),
        T_values.data_ptr<float>(),
        v_targets.data_ptr<float>(),
        obstacles.data_ptr<float>(),
        total_cost.data_ptr<float>(),
        cost_components.data_ptr<float>(),
        num_candidates, num_obstacles, num_time_steps,
        w_jerk, w_lat_accel, w_lon_accel,
        w_ref_dev, w_obstacle, w_vel_target,
        w_time, w_curvature, w_centripetal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    find_best_candidate_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        total_cost.data_ptr<float>(),
        block_best_cost.data_ptr<float>(),
        block_best_idx.data_ptr<int>(),
        num_candidates);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    global_reduce_kernel<<<1, 256, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        block_best_cost.data_ptr<float>(),
        block_best_idx.data_ptr<int>(),
        global_best_cost.data_ptr<float>(),
        global_best_idx.data_ptr<int>(),
        blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
