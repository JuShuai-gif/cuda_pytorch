#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
/*
一般情况下，线程块的大小设置为32的倍数，这样可以避免延迟
块运行在SM上，块又分为好多warp，SM有最大线程数
块最多支持线程是1024

nblock = (grid + block - 1)/block
*/


int main(int argc,char** argv){
    int nElem = 1024;
    dim3 block(1024);
    dim3 grid((nElem-1)/block.x + 1);

    printf("grid.x %d block.x %d\n",grid.x,block.x);

    block.x = 512;
    // 等价于 grid.x = (nElem + block.x -1)/block.x
    grid.x = (nElem - 1)/block.x + 1;
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    block.x = 256;
    grid.x = (nElem - 1)/block.x + 1;
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    block.x = 128;
    grid.x = (nElem - 1)/block.x + 1;
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    cudaDeviceReset();
    return 0;
}




















