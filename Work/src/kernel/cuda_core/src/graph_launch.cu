// CUDA Graph probe: normal launch vs captured graph replay.
//
// A launch-bound workload is many tiny kernels: each launch costs a few
// microseconds of host overhead while the GPU work is near zero.  Capturing
// the sequence into a CUDA graph and replaying it with cudaGraphLaunch turns
// N launches into one, collapsing the host-side cost.  Device time stays
// similar because the GPU work is identical -- that gap (wall collapses,
// event barely moves) is the launch-bound signature.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda_common.h"

using namespace cuda_lab;

constexpr int N = 1024;  // small working set (tiny kernels)

__global__ void scale_add(float* x, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = a * x[i] + b;
}

int main() {
    print_device_info();

    const int n_ops = 64;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float* x = nullptr;
    CUDA_CHECK(cudaMalloc(&x, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(x, 0, N * sizeof(float)));

    auto run_normal = [&]() {
        for (int i = 0; i < n_ops; i++) {
            scale_add<<<blocks, threads>>>(x, 1.0001f, 1.0f);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Build the graph on a dedicated stream (capture must own its stream).
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));
    CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < n_ops; i++) {
        scale_add<<<blocks, threads, 0, capture_stream>>>(x, 1.0001f, 1.0f);
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    auto run_graph = [&]() {
        CUDA_CHECK(cudaGraphLaunch(exec, 0));  // launch on default stream
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Warm up.
    run_normal();
    run_graph();

    const int iters = 100;
    std::vector<double> normal_wall, normal_event, graph_wall, graph_event;
    EventTimer timer;
    for (int k = 0; k < iters; k++) {
        WallTimer w;
        w.start();
        run_normal();
        normal_wall.push_back(w.ms());
    }
    for (int k = 0; k < iters; k++) {
        timer.start();
        for (int i = 0; i < n_ops; i++) {
            scale_add<<<blocks, threads>>>(x, 1.0001f, 1.0f);
        }
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        normal_event.push_back(timer.ms());
    }
    for (int k = 0; k < iters; k++) {
        WallTimer w;
        w.start();
        run_graph();
        graph_wall.push_back(w.ms());
    }
    for (int k = 0; k < iters; k++) {
        timer.start();
        CUDA_CHECK(cudaGraphLaunch(exec, 0));
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        graph_event.push_back(timer.ms());
    }

    JsonReport r;
    r.begin();
    r.put("experiment", "cuda_graph");
    r.put("n_ops", static_cast<long long>(n_ops));
    r.put("normal_wall_ms_mean", mean(normal_wall));
    r.put("graph_wall_ms_mean", mean(graph_wall));
    r.put("normal_event_ms_mean", mean(normal_event));
    r.put("graph_event_ms_mean", mean(graph_event));
    r.put("wall_speedup_x", mean(normal_wall) / mean(graph_wall));
    r.put("event_speedup_x", mean(normal_event) / mean(graph_event));
    std::printf("%s", r.end().c_str());

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(capture_stream));
    CUDA_CHECK(cudaFree(x));
    return 0;
}
