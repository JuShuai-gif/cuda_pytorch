/**
 * @file smem_latency.cu
 * @brief 共享内存延迟基准测试
 * 
 * 本测试用于测量GPU共享内存的访问延迟
 * 共享内存是GPU上最快的可编程内存
 * 
 * 关键技术点:
 * 1. 使用ld.shared.b32内联汇编进行共享内存加载
 * 2. 使用依赖加载链确保延迟无法被隐藏
 * 3. 使用clock测量精确周期数
 * 
 * 共享内存延迟特点:
 * - 延迟极低 (~1-2周期)
 * - 不受缓存层次影响
 * - bank冲突会显著增加延迟
 */

#include <cstdio>
#include <cstdint>

const int WARMUP = 100;
// 要计时的LDS指令数量
const int ROUND = 50;

/**
 * @brief 共享内存延迟测试内核
 * 
 * 测量共享内存的加载延迟:
 * 1. 声明小型共享内存数组(6个uint32_t)
 * 2. 初始化共享内存数据
 * 3. 使用依赖加载链测量延迟
 * 4. 使用ld.shared.b32指令进行加载
 * 
 * @param addr 地址数组
 * @param ret 返回值指针
 * @param clk 时钟周期数组
 */
__global__ __launch_bounds__(16, 1) void smem_latency_kernel(const uint32_t *addr, uint32_t *ret, uint32_t *clk) {
    // 小型共享内存数组
    __shared__ uint32_t smem[6];

    // 初始化共享内存
    smem[threadIdx.x] = addr[threadIdx.x];

    uint32_t start;
    uint32_t stop;
    uint32_t smem_addr;

    // 获取共享内存地址
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(smem_addr)
        : "l"(smem + threadIdx.x));

    // 同步并记录开始时钟
    asm volatile(
        "bat.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory");

    // 依赖加载链测量延迟
#pragma unroll
    for (int i = 0; i < ROUND; ++i) {
        asm volatile(
            "ld.shared.b32 %0, [%0];\n"
            : "+r"(smem_addr) : : "memory");
    }

    // 同步并记录结束时钟
    asm volatile(
        "bat.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory");

    clk[threadIdx.x] = stop - start;

    // 虚拟写回,防止编译器优化
    if (smem_addr == ~0x0) {
        *ret = smem_addr;
    }
}

/**
 * @brief 主函数 - 共享内存延迟基准测试
 * 
 * 测试流程:
 * 1. 创建地址数组,每个元素为偏移量
 * 2. 预热指令缓存
 * 3. 执行延迟测试
 * 4. 输出结果(周期数/ROUND)
 * 
 * 注意:
 * - 使用16个线程,每个线程访问不同的共享内存位置
 * - 共享内存非常快,延迟只有1-2个周期
 */
int main() {
    uint32_t *h_addr;
    cudaMallocHost(&h_addr, 16 * sizeof(uint32_t));

    for (int i = 0; i < 16; ++i) {
        h_addr[i] = i * sizeof(uint32_t);
    }

    uint32_t *d_addr, *d_ret;
    cudaMalloc(&d_addr, 16 * sizeof(uint32_t));
    cudaMalloc(&d_ret, sizeof(uint32_t));
    cudaMemcpy(d_addr, h_addr, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t *d_clk;
    cudaMalloc(&d_clk, 16 * sizeof(uint32_t));

    // 预热L0/L1指令缓存
    for (int i = 0; i < WARMUP; ++i) {
        smem_latency_kernel<<<1, 16>>>(d_addr, d_ret, d_clk);
    }

    // 共享内存延迟基准测试
    smem_latency_kernel<<<1, 16>>>(d_addr, d_ret, d_clk);

    uint32_t h_clk[16];
    cudaMemcpy(h_clk, d_clk, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("shared memory latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_addr);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_addr);

    return 0;
}
