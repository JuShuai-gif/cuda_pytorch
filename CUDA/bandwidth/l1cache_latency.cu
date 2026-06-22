/**
 * @file l1cache_latency.cu
 * @brief L1缓存延迟基准测试
 * 
 * 本测试用于测量GPU L1缓存的访问延迟
 * L1缓存是GPU最快速的缓存,位于每个SM上
 * 
 * 关键技术点:
 * 1. 使用ld.global.nc.b32 (non-coherent global to L1 cache) 指令
 * 2. 通过指针链式解引用访问L1缓存
 * 3. 只使用少数线程(4个)减少竞争
 * 
 * L1缓存特点:
 * - 延迟极低 (~5-10周期)
 * - 容量较小 (通常16-128KB)
 * - 每个SM独立
 */

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <sys/types.h>

// 预热迭代次数
const int WARMUP = 100;
// 使用的线程数(较少线程减少竞争)
const int THREAD = 4;

// 加载指令重复次数
const int ROUND = 50;

/**
 * @brief L1缓存延迟测试内核
 * 
 * 使用指针链式解引用访问L1缓存:
 * 1. ldg_ptr存储一个指针地址
 * 2. 加载该地址的值(也是指针)
 * 3. 用新指针继续访问
 * 
 * 使用ld.global.nc.b64指令进行非一致性的全局加载,
 * 数据会进入L1缓存但不会写回主存
 * 
 * @tparam ROUND 加载次数
 * @param ptr 指针数组
 * @param ret 返回值指针
 * @param clk 时钟周期数组
 */
template <int ROUND>
__global__ __launch_bounds__(THREAD, 1) void l1_latency_kernel(void **ptr, void **ret, uint32_t *clk) {
    void **ldg_ptr = ptr + threadIdx.x;

    // 预热循环
    for (int i = 0; i < ROUND; ++i) {
        asm volatile(
            "ld.global.nc.b64 %0, [%0];\n"
            : "+l"(ldg_ptr) : : "memory");
    }

    uint32_t start;
    uint32_t stop;

    // 同步并记录开始时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start) : : "memory");

    // 实际测量延迟的循环
    for (int i = 0; i < ROUND; ++i) {
        asm volatile(
            "ld.global.nc.b64 %0,[%0];\n"
            : "+l"(ldg_ptr) : : "memory");
    }

    // 同步并记录结束时钟
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop) : : "memory");

    clk[threadIdx.x] = stop - start;

    if (ldg_ptr == nullptr) {
        *ret = ldg_ptr;
    }
}

/**
 * @brief 主函数 - L1缓存延迟基准测试
 * 
 * 测试流程:
 * 1. 创建指针数组,每个元素指向下一个元素,形成链式结构
 *    这样每次加载都会访问前一次加载的结果,确保缓存命中
 * 2. 预热指令缓存
 * 3. 执行延迟测试
 * 4. 输出结果(周期数/ROUND)
 * 
 * 关键设计:
 * - 指针链形成循环,确保数据始终在L1缓存中
 * - 只用4个线程,避免SM资源竞争
 */
int main() {
    void **d_ptr;
    void **d_ret;
    uint32_t *d_clk;
    cudaMalloc(&d_ptr, THREAD * sizeof(void *));
    cudaMalloc(&d_ret, sizeof(void *));
    cudaMalloc(&d_clk, THREAD * sizeof(uint32_t));

    void **h_ptr;
    cudaMallocHost(&h_ptr, THREAD * sizeof(void *));

    // 创建指针链: ptr[i] = d_ptr + i
    // 这样每次解引用都会访问下一个指针,形成循环访问模式
    for (int i = 0; i < THREAD; ++i) {
        h_ptr[i] = d_ptr + i;
    }

    cudaMemcpy(d_ptr, h_ptr, THREAD * sizeof(void *), cudaMemcpyHostToDevice);

    // 预热指令缓存
    for (int i = 0; i < WARMUP; ++i) {
        l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clk);
    }

    // L1缓存延迟基准测试
    l1_latency_kernel<ROUND><<<1, THREAD>>>(d_ptr, d_ret, d_clk);

    uint32_t h_clk[THREAD];
    cudaMemcpy(h_clk, d_clk, THREAD * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("l1 cache latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_ptr);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_ptr);

    return 0;
}
