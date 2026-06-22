#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

constexpr int N = 1 << 20; // 1M elements
constexpr int THREADS_PER_BLOCK = 256;

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    // Host vectors
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device vectors
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Copy result back
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": "
                      << h_c[i] << " != " << h_a[i] + h_b[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "PASSED: vector addition of " << N
                  << " elements verified.\n";
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ok ? 0 : 1;
}
