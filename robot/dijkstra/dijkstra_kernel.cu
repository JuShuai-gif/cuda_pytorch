#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void relax_kernel(
    float* dist,
    const float* graph,
    int* updated_flag,
    int num_nodes
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    float du = dist[u];
    if (isinf(du)) return;

    int base = u * num_nodes;
    int local_updated = 0;

    for (int v = 0; v < num_nodes; v++) {
        float w = graph[base + v];
        if (!isinf(w) && w > 0.0f) {
            float new_dist = du + w;
            if (new_dist < dist[v]) {
                int* dist_int = (int*)(&dist[v]);
                int new_int = __float_as_int(new_dist);
                int old_int;
                do {
                    old_int = atomicMin(dist_int, new_int);
                } while (__float_as_int(new_dist) < old_int);
                local_updated = 1;
            }
        }
    }

    if (local_updated) {
        *updated_flag = 1;
    }
}

void dijkstra_cuda_step(
    torch::Tensor dist,
    torch::Tensor graph,
    torch::Tensor updated_flag,
    int num_nodes
) {
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    updated_flag.zero_();

    relax_kernel<<<blocks, threads, 0,
        torch::cuda::getCurrentCUDAStream()>>>(
        dist.data_ptr<float>(),
        graph.data_ptr<float>(),
        updated_flag.data_ptr<int>(),
        num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
