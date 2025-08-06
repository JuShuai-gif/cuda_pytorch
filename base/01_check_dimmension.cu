#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void)
{
    /*
    threadIdx: 当前线程在其所属线程块中的三维索引
    blockIdx: 当前线程块在整个网格中的三维索引
    blockDim: 线程块的三维大小
    gridDim: 网格的三维大小
    */
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
        gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
                            blockIdx.x,blockIdx.y,blockIdx.z,
                            blockDim.x,blockDim.y,blockDim.z,
                            gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc,char ** argv)
{
    int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1)/block.x);
    printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
    checkIndex<<<block,grid>>>();
    cudaDeviceReset();
    return 0;

}










