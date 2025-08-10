#include <cuda_runtime.h>
#include <stdio.h>


__global__ void nesthelloworld(int isize,int idepth){
    unsigned int tid = threadIdx.x;

    printf("depth : %d blockIdx: %d,threadIdx: %d\n",idepth,blockIdx.x,threadIdx.x);
    if (isize == 1) {
        return;
    }

    int nthread = (isize >> 1);
    if (tid == 0 && nthread > 0) {
        int child_depth = idepth + 1;
        nesthelloworld<<<1, nthread>>>(nthread, child_depth);
        printf("-----------> nested execution depth: %d\n",idepth);
    }
}

int main(int argc,char* argv[])
{
    int size=64;
    int block_x=2;
    dim3 block(block_x,1);
    dim3 grid((size-1)/block.x+1,1);
    nesthelloworld<<<grid,block>>>(size,0);
    cudaGetLastError();
    cudaDeviceReset();
    return 0;
}











