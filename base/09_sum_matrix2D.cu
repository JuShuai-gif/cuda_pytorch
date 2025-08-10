
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

void sumMatrix2D_CPU(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    float *a = MatA;
    float *b = MatB;
    float *c = MatC;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            c[j] = a[j] + b[j];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

__global__ void sumMatrix2D_GPU(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ixy = ix + iy * ny;
    if (ix < nx && iy < ny) {
        MatC[ixy] = MatA[ixy] + MatB[ixy];
    }
}

int main(int argc, char **argv) {
    initDevice(0);
    int nx = 1 << 12;
    int ny = 1 << 12;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(nxy);

    // 分配CPU内存
    float *A_host = (float *)malloc(nBytes);
    float *B_host = (float *)malloc(nBytes);
    float *C_host = (float *)malloc(nBytes);
    float *cpu_from_gpu = (float *)malloc(nBytes);

    initialData(A_host, nxy);
    initialData(B_host, nxy);

    memset(C_host, 0, nxy);
    memset(cpu_from_gpu, 0, nxy);

    // gpu 内存分配
    float *A_dev = NULL;
    float *B_dev = NULL;
    float *C_dev = NULL;
    CHECK(cudaMalloc((void **)&A_dev, nBytes));
    CHECK(cudaMalloc((void **)&B_dev, nBytes));
    CHECK(cudaMalloc((void **)&C_dev, nBytes));

    CHECK(cudaMemcpy(A_dev, A_host, nxy, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nxy, cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;

    // cpu compute
    // cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
    double iStart = cpuSecond();
    sumMatrix2D_CPU(A_host, B_host, C_host, nx, ny);
    double iElaps = cpuSecond() - iStart;
    printf("CPU Execution Time elapsed %f sec\n", iElaps);

    // 2d block and 2d grid
    dim3 block_0(dimx, dimy);
    dim3 grid_0((nx - 1) / block_0.x + 1, (ny - 1) / block_0.y + 1);

    iStart = cpuSecond();
    sumMatrix2D_GPU<<<block_0, grid_0>>>(A_dev, B_dev, C_dev, nx, ny);
    CHECK(cudaDeviceSynchronize())

    iElaps = cpuSecond() - iStart;

    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
           grid_0.x, grid_0.y, block_0.x, block_0.y, iElaps);

    CHECK(cudaMemcpy(cpu_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, cpu_from_gpu, nxy);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(A_host);
    free(B_host);
    free(C_host);
    free(cpu_from_gpu);
    cudaDeviceReset();
    return 0;
}
