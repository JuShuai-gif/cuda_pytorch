/**
 * @file smem_bandwidth.cu
 * @brief 共享内存带宽基准测试
 * 
 * 本测试用于测量GPU共享内存的访问带宽
 * 共享内存是每个SM上最快的内存,延迟极低
 * 
 * 关键技术点:
 * 1. 使用st.shared.v4.b32内联汇编进行共享内存存储
 * 2. 使用clock测量精确周期数
 * 3. 计算理论带宽和实际带宽
 * 4. 支持不同GPU架构(Kepler, Maxwell+)
 * 
 * 共享内存特点:
 * - 延迟极低 (~1-2周期)
 * - 容量小 (通常48KB/SM)
 * - 需要手动管理 bank's冲突
 */

#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
const int BLOCK = 256;

// 要计时的LDS指令数量
const int ROUND = 512;

/**
 * @brief 共享内存带宽测试内核
 * 
 * 测量共享内存的写入带宽:
 * 1. 声明共享内存数组
 * 2. 使用内联汇编进行共享内存存储
 * 3. 使用clock测量精确周期数
 * 4. 每个warp记录开始和结束时间
 * 
 * @param ret 返回值指针
 * @param clk_start 开始时钟数组
 * @param clk_stop 结束时钟数组
 */
__global__ void smem_bandwidth_kernel(
    int *ret, uint32_t *clk_start, uint32_t *clk_stop) {
    // 共享内存数组,大小为BLOCK + ROUND
    __shared__ int4 smem[BLOCK + ROUND];

    uint32_t tid = threadIdx.x;

    uint32_t start;
    uint32_t stop;
    uint32_t smem_addr;
    int4 reg = make_int4(tid, tid + 1, tid + 2, tid + 3);

    // 获取共享内存地址
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(smem_addr)
        : "l"(smem + tid));

    // 同步并记录开始时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory");

    // 执行ROUND次共享内存写入
#pragma unroll
    for (int i = 0; i < ROUND; ++i) {
        asm volatile(
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "r"(smem_addr + i * (uint32_t)sizeof(int4)) "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w)
            : "memory");
    }

    // 同步并记录结束时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory");

    // 每个warp的第一个线程记录时间
    if (threadIdx.x % 32 == 0) {
        clk_start[threadIdx.x / 32] = start;
        clk_stop[threadIdx.x / 32] = stop;
    }

    // 虚拟读取,防止编译器优化
    int tmp = ((int *)smem)[tid];
    if (tmp < 0) {
        *ret = tmp;
    }
}

/**
 * @brief 主函数 - 共享内存带宽基准测试
 * 
 * 测试流程:
 * 1. 获取GPU设备属性
 * 2. 根据GPU架构设置共享内存bank大小
 * 3. 预热指令缓存
 * 4. 执行带宽测试
 * 5. 计算测量带宽和理论带宽
 * 6. 计算整个芯片的理论总带宽
 * 
 * 带宽计算:
 * - 测量带宽 = 共享内存总访问量 / 周期数
 * - 理论带宽 = 向上取整到32的倍数(对齐到warp)
 * - 芯片总带宽 = SM数 * 每SM带宽 * 时钟频率
 */
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 根据GPU架构设置共享内存配置
    if (prop.major == 3) {
        // Kepler GPU: 使用64-bit bank
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if (BLOCK < 256) {
            printf("thread block size is not enough to utilze all LSU.\n");
        }
    } else {
        if (BLOCK < 128) {
            printf("thread block size is not enough to utilze all LSU.\n");
        }
    }

    int *d_ret;
    uint32_t *d_clk_start;
    uint32_t *d_clk_stop;
    cudaMalloc(&d_ret, BLOCK * sizeof(int));
    cudaMalloc(&d_clk_start, BLOCK / 32 * sizeof(uint32_t));
    cudaMalloc(&d_clk_stop, BLOCK / 32 * sizeof(uint32_t));

    // 预热L0/L1指令缓存
    for (int i = 0; i < WARMUP; ++i) {
        smem_bandwidth_kernel<<<1, BLOCK>>>(d_ret, d_clk_start, d_clk_stop);
    }

    // 共享内存带宽基准测试
    smem_bandwidth_kernel<<<1, BLOCK>>>(d_ret, d_clk_start, d_clk_stop);

    uint32_t h_clk_start[BLOCK];
    uint32_t h_clk_stop[BLOCK];
    cudaMemcpy(h_clk_start, d_clk_start, BLOCK / 32 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clk_stop, d_clk_stop, BLOCK / 32 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // 找出最早开始和最晚结束的时间
    uint32_t start_min = ~0;
    uint32_t stop_max = 0;

    for (int i = 0; i < BLOCK / 32; ++i) {
        if (h_clk_start[i] < start_min) {
            start_min = h_clk_start[i];
        }
        if (h_clk_stop[i] > stop_max) {
            stop_max = h_clk_stop[i];
        }
    }

    uint32_t smem_size = BLOCK * ROUND * sizeof(int4);
    uint32_t duration = stop_max - start_min;
    float bw_measured = float(smem_size) / duration;
    // 向上取整到32
    uint32_t bw_theoretical = ((uint32_t)bw_measured + 31) / 32 * 32;

    printf("shared memory accessed: %u byte\n", smem_size);
    printf("duration: %u cycles\n", duration);
    printf("shared memory bandwidth per SM (measured): %f byte/cycle\n", bw_measured);
    printf("shared memory bandwidth per SM (theoretical): %u byte/cycle\n", bw_theoretical);

    uint32_t clk = prop.clockRate / 1000;
    uint32_t sm = prop.multiProcessorCount;
    float chip_bandwidth = float(sm) * bw_theoretical * clk / 1000;
    printf("standard clock frequency: %u MHz\n", clk);
    printf("SM: %u\n", sm);
    printf("whole chip shared memory bandwidth (theoretical): %f GB/s\n", chip_bandwidth);

    cudaFree(d_ret);
    cudaFree(d_clk_start);
    cudaFree(d_clk_stop);

    return 0;
}
