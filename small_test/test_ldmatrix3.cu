#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"

__global__ void helloFromGPU (void) {
  __shared__ half aTile[2*8*8];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  // 下面的代码是把smem中的2*8*8的矩阵，初始化数值！
  if (tidx == 0) {
    for (int i = 0; i < 2*8*8; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx * 8;
  uint32_t my_register[2];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [ %2 ];\n"
  : "=r"(my_register[0]), "=r"(my_register[1])
  : "r"(smem)
  );

  if (tidx == 4) {
    for (int i = 0; i < 2; i++) {
        half * tmp = (half*)(&(my_register[i]));
        printf("%f\n", (float)(tmp[0]));
        printf("%f\n", (float)(tmp[1]));
    }
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
helloFromGPU <<<grid, block>>>();

cudaDeviceReset();
return 0;
}