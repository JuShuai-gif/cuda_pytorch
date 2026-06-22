#include <stdio.h>
#include <iostream>
#include <cuda_bf16.h>


__global__ void test_ldmatrix()
{
    __shared__ __nv_bfloat16 aTile[8*8];

    int tid = threadIdx.x;

    if (tid == 0) {
        for (int i = 0; i < 64; i++)
            aTile[i] = __float2bfloat16((float)i);
    }

    __syncthreads();

    int row = tid % 8;

    uint32_t reg_a;

    uint32_t smem =
        __cvta_generic_to_shared(&aTile[row * 8]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
        : "=r"(reg_a)
        : "r"(smem)
    );

    if (tid == 1)
    {
        __nv_bfloat16 *ptr = (__nv_bfloat16 *)&reg_a;

        printf("Thread %d: %f %f\n",
               tid,
               __bfloat162float(ptr[0]),
               __bfloat162float(ptr[1]));
    }
}

__global__ void helloFromGPU (void)
{
  __shared__ uint32_t aTile[4*8*4];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  // 下面的代码是把smem中的4*8*4的矩阵，初始化数值！
  if (tidx == 0) {
    for (int i = 0; i < 4*8*4; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx % 16 * 8 + tidx / 16 * 4;
  uint32_t a[4];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
  : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) 
  : "r"(smem)
  );

  if (tidx == 1) {
    printf("%d \n", (a[0])); printf("%d \n", (a[1]));
    printf("%d \n", (a[2])); printf("%d \n", (a[3]));
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
test_ldmatrix <<<grid, block>>>();

cudaDeviceReset();
return 0;
}