// Host-to-device copy probe: pinned vs pageable, blocking vs async.
//
// On a discrete GPU, pageable host memory must be staged through pinned
// buffers before DMA, and cudaMemcpy blocks the host until the copy finishes.
// cudaMemcpyAsync enqueues the copy and returns immediately, letting the host
// overlap other work.  On this Jetson/Thor unified-memory SoC the pageable vs
// pinned gap is smaller (no PCIe crossing), but the blocking vs async launch
// behavior is still measurable.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_common.h"

using namespace cuda_lab;

int main() {
    print_device_info();

    const size_t bytes = 64ull * 1024 * 1024;  // 64 MB
    const size_t n = bytes / sizeof(float);

    // Pageable host buffer (malloc).
    float* pageable = static_cast<float*>(std::malloc(bytes));
    // Pinned host buffer (cudaHostAlloc).
    float* pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
    // Device buffer.
    float* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, bytes));

    for (size_t i = 0; i < n; i++) pageable[i] = pinned[i] = 1.0f;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const int iters = 50;

    auto bench = [&](const char* label, bool use_pinned, bool async) {
        float* src = use_pinned ? pinned : pageable;
        std::vector<double> wall_ms, event_ms;
        EventTimer timer;

        // Warm up (first allocation / page setup is not representative).
        for (int k = 0; k < 5; k++) {
            if (async) {
                CUDA_CHECK(cudaMemcpyAsync(device, src, bytes, cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpy(device, src, bytes, cudaMemcpyHostToDevice));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Wall time: what the host actually experiences (sync included).
        for (int k = 0; k < iters; k++) {
            WallTimer w;
            w.start();
            if (async) {
                CUDA_CHECK(cudaMemcpyAsync(device, src, bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                CUDA_CHECK(cudaMemcpy(device, src, bytes, cudaMemcpyHostToDevice));
            }
            wall_ms.push_back(w.ms());
        }

        // Device time: CUDA-event measured copy duration.
        for (int k = 0; k < iters; k++) {
            timer.start(stream);
            if (async) {
                CUDA_CHECK(cudaMemcpyAsync(device, src, bytes, cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpy(device, src, bytes, cudaMemcpyHostToDevice));
            }
            timer.stop(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            event_ms.push_back(timer.ms());
        }

        double gbps = (static_cast<double>(bytes) / (mean(wall_ms) * 1e-3)) / 1e9;
        JsonReport r;
        r.begin();
        r.put("experiment", "h2d_copy");
        r.put("variant", label);
        r.put("bytes", static_cast<long long>(bytes));
        r.put("wall_ms_mean", mean(wall_ms));
        r.put("event_ms_mean", mean(event_ms));
        r.put("gbps_wall", gbps);
        std::printf("%s", r.end().c_str());
    };

    bench("pageable_sync", /*pinned=*/false, /*async=*/false);
    bench("pinned_sync", true, false);
    bench("pinned_async", true, true);

    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(pinned));
    std::free(pageable);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
