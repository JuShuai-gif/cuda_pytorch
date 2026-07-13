

#include <stdio.h>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;                 // a million floats (1,048,576)
    size_t bytes = n * sizeof(float);

    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes),
          *c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) a[i] = b[i] = 1.0f;

    float *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    vadd<<<4096, 256>>>(da, db, dc, n);   // 4096 * 256 = n threads, one per float

    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);
    printf("c[0]=%f c[n-1]=%f\n", c[0], c[n-1]);
}