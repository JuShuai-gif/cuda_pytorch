/**
 * @file dram_latency.cu
 * @brief DRAM (全局内存) 访问延迟基准测试
 * 
 * 本测试用于测量GPU访问DRAM(全局内存)的延迟
 * 使用依赖链(dependent load)确保延迟无法被并行访问隐藏
 * 
 * 关键技术点:
 * 1. 使用bar.sync和clock测量精确的指令周期数
 * 2. 使用依赖加载确保测量的是真实延迟而非带宽
 * 3. 使用大stride避免L2缓存命中,确保访问DRAM
 * 4. 使用L2 flush清空缓存,保证每次测试都是DRAM访问
 * 
 * 延迟组成:
 * - L2未命中时的DRAM访问延迟 ~400-800周期(视GPU架构而定)
 * - TLB查找延迟
 * - 内存控制器延迟
 */

#include <cmath>
#include <cstdio>
#include <cstdint>

// 要计时的LDG指令数量
const int ROUND = 10;
// LDG指令之间的stride(字节)
// 必须大于L2缓存行大小(通常128B)以避免L2缓存命中
const int STRIDE = 1024;

// 用于刷清L2缓存的工作空间大小(128MB)
const int L2_FLUSH_SIZE = (1 << 20) * 128;

/**
 * @brief L2缓存刷清内核
 * 
 * 通过访问大量数据(128MB)将L2缓存填满,从而刷清原有的缓存数据
 * 这样可以确保后续的延迟测试访问的是真实DRAM而非缓存数据
 * 
 * @tparam BLOCK 每个block的线程数
 * @param x 输入数据指针
 * @param y 输出指针(用于防止优化)
 */
template <int BLOCK>
__global__ void flush_l2_kernel(const int *x, int *y) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    const int *x_ptr = x + blockIdx.x * BLOCK + warp_id * 32;
    int sum = 0;

#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int *ldg_ptr = x_ptr + (lane_id ^ i);

        asm volatile(
            "{.reg .s32 val;\n"
            " ld.global.cg.b32 val, [%1];\n"
            " add.s32 %0, val, %0;}\n"
            : "+r"(sum) : "l"(ldg_ptr));
    }

    if (sum != 0) {
        *y = sum;
    }
}

void flush_l2() {
    int *x;
    int *y;
    cudaMalloc(&x, L2_FLUSH_SIZE);
    cudaMalloc(&y, sizeof(int));
    cudaMemset(x, 0, L2_FLUSH_SIZE);

    int n = L2_FLUSH_SIZE / sizeof(int);
    flush_l2_kernel<128><<<n / 128, 128>>>(x, y);

    cudaFree(x);
    cudaFree(y);
}

/**
 * @brief DRAM延迟测试内核
 * 
 * 测量单个内存访问的延迟:
 * 1. 使用bar.sync确保所有线程同步
 * 2. 使用clock获取开始和结束时的GPU时钟周期
 * 3. 使用依赖加载链:每次加载的地址依赖上一次加载的结果
 *    这样确保每个加载必须等待上一个完成,无法并行隐藏延迟
 * 4. 使用ld.global.cg.b32指令(缓存加载)访问全局内存
 * 
 * @tparam ROUND 加载次数
 * @param stride stride数组指针
 * @param ret 返回值指针
 * @param clk 时钟周期数组
 */
template <int ROUND>
__global__ __launch_bounds__(32, 1) void dram_latency_kernel(const uint32_t *stride,
                                                             uint32_t *ret,
                                                             uint32_t *clk) {
    const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
    uint32_t val;

    // 预热TLB(Translation Lookaside Buffer),避免TLB miss影响测量
    asm volatile(
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(val)
        : "l"(ldg_ptr)
        : "memory");

    ldg_ptr += val;

    uint32_t start;
    uint32_t stop;

    // 同步并记录开始时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory");

    // 依赖加载链:每次加载的地址基于上次加载的结果
    // 这样确保延迟无法被并行加载隐藏
    // IADD/IMAD/XMAD的延迟远低于DRAM,可以忽略
#pragma unroll
    for (int i = 0; i < ROUND; ++i) {
        asm volatile(
            "ld.global.cg.b32 %0, [%1];\n"
            : "=r"(val)
            : "l"(ldg_ptr)
            : "memory");

        /*
         * dependent LDG instructions to make sure that
         * LDG latency can not be hidden by parallel LDG.
         *
         * IADD/IMAD/XMAD's latency is much lower than dram and can be ignored.
         */
        ldg_ptr += val;
    }

    // 同步并记录结束时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory");

    clk[threadIdx.x] = stop - start;

    // 虚拟写回,防止编译器优化掉加载
    if (val == 0) {
        *ret = val;
    }
}

/**
 * @brief 主函数 - DRAM延迟基准测试
 * 
 * 测试流程:
 * 1. 分配stride数组,每个元素存储步长值
 * 2. 预热L0/L1指令缓存
 * 3. 刷清L2缓存,确保后续访问DRAM
 * 4. 执行延迟测试,测量加载指令的时钟周期
 * 5. 计算并输出平均延迟(周期数/ROUND)
 * 
 * 使用pinned memory加速数据传输
 */
int main() {
    static_assert(STRIDE >= 32 * sizeof(uint32_t) && STRIDE % sizeof(uint32_t) == 0,
                  "invalid 'STRIDE'");

    const uint32_t STRIDE_MEM_SIZE = (ROUND + 1) * STRIDE;

    // 使用pinned memory(页锁定内存)提高数据传输效率
    uint32_t *h_stride;
    cudaMallocHost(&h_stride, STRIDE_MEM_SIZE);

    for (int i = 0; i < STRIDE_MEM_SIZE / sizeof(uint32_t); ++i) {
        h_stride[i] = STRIDE;
    }

    uint32_t *d_stride, *d_ret;
    cudaMalloc(&d_stride, STRIDE_MEM_SIZE);
    cudaMalloc(&d_ret, sizeof(uint32_t));
    cudaMemcpy(d_stride, h_stride, STRIDE_MEM_SIZE, cudaMemcpyHostToDevice);

    uint32_t *d_clk;
    cudaMalloc(&d_clk, 32 * sizeof(uint32_t));

    // 预热L0/L1指令缓存
    dram_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);

    // 刷清L2缓存
    flush_l2();

    // 执行DRAM延迟基准测试
    dram_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, d_clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("DRAM latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_stride);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_stride);

    return 0;
}
