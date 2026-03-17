#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void test_ldmatrix(__nv_bfloat16 *check1, __nv_bfloat16 *check2) {
    __shared__ __nv_bfloat16 smem_a[64]; // 8x8 matrix in row-major
    __shared__ __nv_bfloat16 smem_b[64]; // 8x8 matrix in col-major

    int tid = threadIdx.x;

    // Initialize matrix A (row-major): [1, 2, 3, ..., 64]
    // Each thread initializes 2 elements
    check1[tid * 2] = smem_a[tid * 2] = __float2bfloat16((tid * 2) + 1.0f);
    check1[tid * 2 + 1] = smem_a[tid * 2 + 1] = __float2bfloat16((tid * 2) + 2.0f);

    // printf("tid=%d idx=(%d,%d) value=(%.1f, %.1f)\n",
    //    tid,
    //    tid * 2,
    //    tid * 2 + 1,
    //    __bfloat162float(check1[tid * 2]),
    //    __bfloat162float(check1[tid * 2 + 1]));


    // // Initialize matrix B in col-major layout
    // // For col-major, column 0 is [1, 9, 17, 25, 33, 41, 49, 57]
    // // Memory layout: [1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, ...]
    // for (int i = 0; i < 2; i++) {
    //     int idx = tid * 2 + i;
    //     int col = idx / 8;
    //     int row = idx % 8;
    //     int value = row * 8 + col + 1; // Element at (row, col)
    //     check2[idx] = smem_b[idx] = __float2bfloat16(value);
    // }
    // 行主序
    check2[tid * 2] = smem_b[tid * 2] = __float2bfloat16((tid * 2) + 1.0f);
    check2[tid * 2 + 1] = smem_b[tid * 2 + 1] = __float2bfloat16((tid * 2) + 2.0f);


    __syncthreads();

    // Single register to hold loaded data (1 register = 2 bf16 values)
    uint32_t reg_a;
    uint32_t reg_b;

    // ldmatrix without trans - loads from row-major layout
    // Each thread accesses based on its lane ID
    // 把 shared memory 指针转换为 32-bit shared memory address，用于 PTX 指令访问
    uint32_t smem_addr_a = __cvta_generic_to_shared(&smem_a[(tid % 8) * 8]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                    : "=r"(reg_a)
                    : "r"(smem_addr_a));

    // ldmatrix with trans - loads with transpose
    uint32_t smem_addr_b = __cvta_generic_to_shared(&smem_b[(tid % 8) * 8]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                    : "=r"(reg_b)
                    : "r"(smem_addr_b));

    // Thread 0 prints its register
    if (tid <10 ) {
        __nv_bfloat16 *ptr_a = (__nv_bfloat16 *)&reg_a;
        __nv_bfloat16 *ptr_b = (__nv_bfloat16 *)&reg_b;

        printf("Thread 0 - ldmatrix without trans: %f, %f\n",__bfloat162float(ptr_a[0]), __bfloat162float(ptr_a[1]));

        printf("Thread 0 - ldmatrix with trans: %f, %f\n",__bfloat162float(ptr_b[0]), __bfloat162float(ptr_b[1]));
    }
}





int main() {
    // Check for Tensor Core support (compute capability >= 7.0)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 7) {
        printf("This program requires compute capability 7.0 or higher\n");
        return 1;
    }

    printf("Running ldmatrix test on %s (compute %d.%d)\n\n", prop.name ,
        prop.major, prop.minor);

    __nv_bfloat16 *h_check1 = (__nv_bfloat16 *)malloc(64 * sizeof(__nv_bfloat16));
    __nv_bfloat16 *h_check2 = (__nv_bfloat16 *)malloc(64 * sizeof(__nv_bfloat16));

    __nv_bfloat16 *d_check1{};

    __nv_bfloat16 *d_check2{};

    cudaMalloc((void **)&d_check1, 64 * sizeof(__nv_bfloat16));
    cudaMalloc((void **)&d_check2, 64 * sizeof(__nv_bfloat16));

    // Launch kernel with 1 warp (32 threads)
    test_ldmatrix<<<1, 32>>>(d_check1, d_check2);

    cudaMemcpy(h_check1, d_check1, 64 * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_check2, d_check2, 64 * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("行主序结果: \n");
    for (int i = 0; i < 64; ++i) {
        if (i % 8 == 0 && i != 0)
        printf("\n");
        int m = i % 8;
        int n = i / 8;
        printf("a[%2d] = %.0f\t", i, __bfloat162float(h_check1[m + n * 8]));
    }

    printf("\n");
    printf("\n");
    printf("列主序结果: \n");
    for (int i = 0; i < 64; ++i) {
        if (i % 8 == 0 && i != 0)
            printf("\n");
        int m = i % 8;
        int n = i / 8;
        printf("b[%2d] = %.0f\t", i, __bfloat162float(h_check2[m * 8 + n]));
    }

    free(h_check1);
    free(h_check2);
    cudaFree(d_check1);
    cudaFree(d_check2);

    return 0;
}














