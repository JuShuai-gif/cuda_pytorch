/**
 * @file l2cache_latency.cu
 * @brief L2缓存延迟基准测试
 * 
 * 本测试用于测量GPU L2缓存的访问延迟
 * L2缓存是所有SM共享的二级缓存,延迟比DRAM低很多
 * 
 * 关键技术点:
 * 1. 使用较小的stride(128B),确保访问在L2缓存中
 * 2. 使用依赖加载链确保延迟无法被隐藏
 * 3. 预热阶段将数据加载到L2缓存
 * 
 * 与DRAM延迟测试的区别:
 * - STRIDE更小(128B vs 1024B)
 * - 无需L2 flush(因为我们希望L2命中)
 * - 延迟结果远小于DRAM延迟
 */

#include <cmath>
#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
// 要计时的LDG指令数量
const int ROUND = 10;
// LDG指令之间的stride(字节)
// 128B 通常大于L1缓存行但小于L2缓存行,可确保L2命中
const int STRIDE = 128;

/**
 * @brief L2缓存延迟测试内核
 * 
 * 与DRAM延迟测试类似,但使用更小的stride:
 * 1. 使用依赖加载链测量延迟
 * 2. 使用ld.global.cg.b32指令
 * 3. 通过stride控制访问模式
 * 
 * @tparam ROUND 加载次数
 * @param stride stride数组
 * @param ret 返回值指针
 * @param clk 时钟周期数组
 */
template <int ROUND>
__global__ __launch_bounds__(32, 1) void l2_latency_kernel(const uint32_t *stride,
                                                           uint32_t *ret,
                                                           uint32_t *clk) {
    const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
    uint32_t val;

    // 第一次加载预热
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

    // 依赖加载链测量延迟
    // IADD/IMAD/XMAD的延迟远低于L2缓存,可以直接忽略
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
         * IADD/IMAD/XMAD's latency is much lower than
         * l2 cache and can be ignored.
         */
        ldg_ptr += val;
    }

    // 同步并记录结束时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory");

    clk[threadIdx.x] = stop - start;

    if (val == 0) {
        *ret = val;
    }
}

/**
 * @brief 主函数 - L2缓存延迟基准测试
 * 
 * 测试流程:
 * 1. 分配stride数组
 * 2. 预热L0/L1指令缓存和L2缓存
 * 3. 执行延迟测试
 * 4. 输出结果(周期数/ROUND)
 * 
 * 关键与DRAM延迟测试的区别:
 * - 不需要flush L2(我们希望L2命中)
 * - stride更小(128B vs 1024B)
 */
int main() {
    static_assert(STRIDE >= 32 * sizeof(uint32_t) && STRIDE % sizeof(uint32_t) == 0,
                  "invalid 'STRIDE'");

    const uint32_t STRIDE_MEM_SIZE = (ROUND + 1) * STRIDE;

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

    // 预热L0/L1指令缓存和L2缓存
    for (int i = 0; i < WARMUP; ++i) {
        l2_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);
    }

    // L2缓存延迟基准测试
    l2_latency_kernel<ROUND><<<1, 32>>>(d_stride, d_ret, d_clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, d_clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("l2 cache latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_stride);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_stride);

    return 0;
}
