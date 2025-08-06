
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

void sumArrayCPU(int *a, int *b, int *c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
    }
}

__global__ void sumArrayGPU(int *a, int *b, int *c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    int div = 0;
    initDevice(0);
    bool bResult = false;

    int size = 1 << 20;
    printf("with array size %d\n", size);

    int blocksize = 1024;
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    printf("grid %d block %d \n", grid.x, block.x);

    size_t bytes = sizeof(int) * size;

    int *a_h = (int *)malloc(bytes);
    int *b_h = (int *)malloc(bytes);
    int *tmp = (int *)malloc(bytes);

    int *tmp_from_gpu = (int *)malloc(bytes);

    initialData_int(a_h, size);
    initialData_int(b_h, size);

    printf("a_h[0] %d\n", a_h[0]);
    printf("b_h[0] %d\n", b_h[0]);

    memset(tmp, 0, bytes);

    double iStart{}, iElaps{};

    double iStart1{}, iElaps1{};

    int gpu_sum{0};

    // 设备内存
    int *idata_dev = NULL;
    int *odata_dev = NULL;

    int *a_d = NULL;
    int *b_d = NULL;
    int *c_d = NULL;
    CHECK(cudaMalloc((void **)&a_d, bytes));
    CHECK(cudaMalloc((void **)&b_d, bytes));
    CHECK(cudaMalloc((void **)&c_d, bytes));

    CHECK(cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice));

    int cpu_sum = 0;
    iStart = cpuSecond();

    sumArrayCPU(a_h, b_h, tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    iStart1 = cpuSecond();

    sumArrayGPU<<<block, grid>>>(a_d, b_d, c_d, size);
    CHECK(cudaMemcpy(tmp_from_gpu, c_d, bytes, cudaMemcpyDeviceToHost));
    printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);
    iElaps1 = cpuSecond() - iStart1;

    checkResultInt(tmp, tmp_from_gpu, size);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(tmp);
    free(tmp_from_gpu);
    return 0;
}
