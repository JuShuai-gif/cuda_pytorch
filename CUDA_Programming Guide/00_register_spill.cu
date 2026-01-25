#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 设备端内核函数
__global__ void kernel_bad(float *out) {
    int tid = threadIdx.x;

    float arr[128];                 // ① 大局部数组
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        arr[i] = tid * 0.1f + i;
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        sum += arr[i];              // ② 长活跃区间
    }

    out[tid] = sum;
}

// 主机端验证函数
void verify_results(float *h_out, int n) {
    printf("Verifying results for first 5 threads:\n");
    for (int tid = 0; tid < 5 && tid < n; ++tid) {
        // 手动计算期望值
        float expected = 0.0f;
        for (int i = 0; i < 128; ++i) {
            expected += tid * 0.1f + i;
        }
        
        printf("Thread %d: computed = %.2f, expected = %.2f, diff = %.6f\n",
               tid, h_out[tid], expected, fabs(h_out[tid] - expected));
    }
}

int main() {
    // 设置线程块和网格大小
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = 1;
    const int TOTAL_THREADS = BLOCK_SIZE * GRID_SIZE;
    
    printf("Starting CUDA kernel test...\n");
    printf("Configuration: Grid size = %d, Block size = %d\n", GRID_SIZE, BLOCK_SIZE);
    printf("Each thread uses 128-element array (512 bytes)\n");
    
    // 分配主机内存
    float *h_out = new float[TOTAL_THREADS];
    
    // 分配设备内存
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, TOTAL_THREADS * sizeof(float)));
    
    // 启动内核
    printf("Launching kernel...\n");
    kernel_bad<<<GRID_SIZE, BLOCK_SIZE>>>(d_out);
    
    // 检查内核执行错误
    CHECK_CUDA(cudaGetLastError());
    
    // 等待内核完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 将结果从设备复制到主机
    CHECK_CUDA(cudaMemcpy(h_out, d_out, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 验证结果
    verify_results(h_out, TOTAL_THREADS);
    
    // 计算总和用于验证
    float total_sum = 0.0f;
    for (int i = 0; i < TOTAL_THREADS; ++i) {
        total_sum += h_out[i];
    }
    printf("\nTotal sum across all threads: %.2f\n", total_sum);
    
    // 清理
    delete[] h_out;
    CHECK_CUDA(cudaFree(d_out));
    
    // 重置设备
    CHECK_CUDA(cudaDeviceReset());
    
    printf("Test completed successfully!\n");
    return 0;
}