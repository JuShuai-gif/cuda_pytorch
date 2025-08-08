#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

__global__ void printThreadIndex(float* A,const int nx,const int ny){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int idx = iy * nx + ix;
    printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
            "global index %2d ival %f\n",threadIdx.x,threadIdx.y,
            blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}


int main(int argc,char** argv){
    initDevice(0);
    int nx = 8,ny=6;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);

    // Malloc
    float* A_host = (float*)malloc(nBytes);
    initialData(A_host,nxy);
    printMatrix(A_host,nx,ny);

    // cudaMalloc
    float* A_dev = NULL;
    CHECK(cudaMalloc((void**)&A_dev,nBytes));

    cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyDeviceToHost);

    dim3 block(4,2);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    printThreadIndex<<<grid,block>>>(A_dev,nx,ny);

    CHECK(cudaDeviceSynchronize());

    cudaFree(A_dev);
    free(A_host);

    cudaDeviceReset();
    return 0;
}





