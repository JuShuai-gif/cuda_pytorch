#include "helper_cuda ptx.h"

#include <cstdint>

__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(int m,
                                               int n,
                                               int k,
                                               const float alpha,
                                               const float* A,
                                               int lda,
                                               const float* B,
                                               int ldb,
                                               const float beta,
                                               float* C,
                                               int ldc) 
{
    // Operands A, B, C: row-major format

    // Abbreviations:
    // ldg - 加载全局内存
    // lds - 加载共享内存
    // stg - 存储全局内存
    // sts - 存储共享内存
    // cvta - 转换地址

    


}