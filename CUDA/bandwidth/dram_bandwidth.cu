/**
 * @file dram_bandwidth.cu
 * @brief DRAM (全局内存) 带宽基准测试
 * 
 * 本测试用于测量GPU到DRAM(全局内存)的读写带宽，包括:
 * - 纯读取带宽 (read)
 * - 纯写入带宽 (write)
 * - 同时读写带宽 (copy)
 * 
 * 测试方法: 使用CUDA事件(Event)计时,多次迭代取平均值以获得稳定结果
 * 使用ldg_cs/stg_cs内联汇编确保使用缓存加载/存储指令
 */

#include <cstdio>
#include <cstdint>

// 每次迭代的内存偏移量,用于避免缓存影响(16MB偏移)
const int MEMORY_OFFSET = (1u << 20) * 16;
// 基准测试迭代次数,用于减少计时误差
const int BENCH_ITER = 100;

// 每个block的线程数
const int BLOCK = 128;
// 向量加载展开因子,用于一次加载多个数据
const int LDG_UNROLL = 1;

/**
 * @brief 使用缓存加载指令读取全局内存
 * 
 * ldg.cs (Load Global with Cache Streaming) 是一种全局内存加载指令,
 * 数据会加载到L1/L2缓存中,适合读取一次后会被多次使用的数据
 * 使用内联汇编可以确保使用特定的指令,绕过编译器优化
 * 
 * @param ptr 读取地址
 * @return uint4 返回4个32位数据(16字节)
 */
__device__ __forceinline__ uint4 ldg_cs(const void *ptr) {
    uint4 ret;
    // ld.global：从全局内存 DRAM 加载数据
    // .cs Cache Streaming,使用缓存流式加载
    // .v4 一次加载4个元素
    // .b32 每个元素 32 位(4字节)
    // =r =输出约束，表示将结果写入寄存器
    // ret.x ret.y ret.z ret.w uint4的4个分量
    // %0 %1 %2 %3 编译器自动映射到这4个寄存器
    // l 地址约束，必须是64位地址
    // ptr 要读取的内存地址
    // %4 映射到第5个操作数
    /*
    等价的高级代码：
    uint4 ret;
    ret.x = ptr[0];   // 加载第1个32位
    ret.y = ptr[1];   // 加载第2个32位  
    ret.z = ptr[2];   // 加载第3个32位
    ret.w = ptr[3];   // 加载第4个32位

    .cs vs .cg的区别
    ld.global.cs Cache Streaming，数据进入L1/L2 缓存
    ld.global.cg Cache Global,数据进入L2，不进L1

    cs适合读多次的场景，cg适合只读一次的场景
    */
    asm volatile(
        "ld.global.cs.v4.b32 {%0,%1,%2,%3},[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr));
    return ret;
}

/**
 * @brief 使用缓存存储指令写入全局内存
 * 
 * stg.cs (Store Global with Cache Streaming) 是一种全局内存存储指令,
 * 数据会写入L1/L2缓存后再同步到DRAM
 * 
 * @param reg 要存储的寄存器数据
 * @param ptr 写入地址
 */
__device__ __forceinline__ void stg_cs(const uint4 &reg, void *ptr) {
    /*
    st.global.cs.v4.b32
    st.global 写入全局内存(DRAM)
    .cs Cache Streaming 通过缓存写入
    .v4 Vector 4 一次写入4个元素
    .b32 每个元素32位

    将 4x32bit = 16字节写入全局内存

    约束 | 含义 |
    |------|------|
    | "r"(reg.x) | 输入,32位寄存器,第0个操作数 |
    | "r"(reg.y) | 输入,32位寄存器,第1个操作数 |
    | "r"(reg.z) | 输入,32位寄存器,第2个操作数 |
    | "r"(reg.w) | 输入,32位寄存器,第3个操作数 |
    | "l"(ptr) | 输入,64位地址,第4个操作数 |

    等价的高级代码:
    ptr[0] = reg.x;
    ptr[1] = reg.y;
    ptr[2] = reg.z;
    ptr[3] = reg.w;
    */
    asm volatile(
        "st.global.cs.v4.b32 [%4],{%0,%1,%2,%3};" ::"r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w), "l"(ptr)
    );
}

/**
 * @brief 纯读取内核 - 只读取全局内存数据
 * 
 * 读取策略: 每个线程读取VEC_UNROLL个数据块,每个块大小为BLOCK
 * 这样可以增加内存访问的并行度,提高带宽利用率
 * 使用条件存储防止编译器优化掉读取操作
 * 
 * @tparam BLOCK 每个线程处理的block数
 * @tparam VEC_UNROLL 展开因子
 * @param x 输入数据指针
 * @param y 输出指针(仅用于防止读取被优化)
 */
template <int BLOCK, int VEC_UNROLL>
__global__ void read_kernel(const void *x, void *y) {
    uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

    const uint4 *ldg_ptr = (const uint4 *)x + idx;
    uint4 reg[VEC_UNROLL];

#pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        reg[i] = ldg_cs(ldg_ptr + i * BLOCK);
    }

#pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        if (reg[i].x != 0) {
            stg_cs(reg[i], (uint4 *)y + i);
        }
    }
}

/**
 * @brief 纯写入内核 - 只写入全局内存数据
 * 
 * 写入全0数据到全局内存,用于测量纯写入带宽
 * 使用stg_cs内联汇编确保使用缓存存储指令
 * 
 * @tparam BLOCK 每个线程处理的block数
 * @tparam VEC_UNROLL 展开因子
 * @param y 输出数据指针
 */
template <int BLOCK, int VEC_UNROLL>
__global__ void write_kernel(void *y) {
    uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

    uint4 *stg_ptr = (uint4 *)y + idx;

#pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        uint4 reg = make_uint4(0, 0, 0, 0);
        stg_cs(reg, stg_ptr + i * BLOCK);
    }
}

/**
 * @brief 复制内核 - 同时读写全局内存
 * 
 * 先从x读取数据,然后写入到y,模拟真实的内存复制操作
 * 同时测量读取和写入带宽,总带宽约为读取+写入
 * 
 * @tparam BLOCK 每个线程处理的block数
 * @tparam VEC_UNROLL 展开因子
 * @param x 输入数据指针
 * @param y 输出数据指针
 */
template <int BLOCK, int VEC_UNROLL>
__global__ void copy_kernel(const void *x, void *y) {
    uint32_t idx = blockIdx.x * BLOCK * VEC_UNROLL + threadIdx.x;

    const uint4 *ldg_ptr = (const uint4 *)x + idx;
    uint4 *stg_ptr = (uint4 *)y + idx;
    uint4 reg[VEC_UNROLL];

#pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        reg[i] = ldg_cs(ldg_ptr + i * BLOCK);
    }

#pragma unroll
    for (int i = 0; i < VEC_UNROLL; ++i) {
        stg_cs(reg[i], stg_ptr + i * BLOCK);
    }
}

/**
 * @brief DRAM带宽基准测试函数
 * 
 * 测试流程:
 * 1. 预热: 运行几次内核使GPU达到稳定状态
 * 2. 读取测试: 测量纯读取带宽
 * 3. 写入测试: 测量纯写入带宽
 * 4. 复制测试: 测量同时读写带宽
 * 
 * 使用MEMORY_OFFSET确保每次迭代访问不同的内存区域,避免缓存影响
 * 
 * @param size_in_byte 测试数据大小(字节)
 */
void benchmark(size_t size_in_byte) {
    printf("%luMB (r+w)\n", size_in_byte / (1 << 20));

    double size_gb = (double)size_in_byte / (1 << 30);

    size_t n = size_in_byte / sizeof(uint4);
    size_t grid = n / (BLOCK * LDG_UNROLL);

    static_assert(MEMORY_OFFSET % sizeof(uint4) == 0,
                  "invalid MEMORY_OFFSET");

    // 分配足够的内存: 测试数据 + 迭代偏移空间
    // 使用MEMORY_OFFSET * BENCH_ITER确保每次迭代使用不同缓存行
    char *ws;
    cudaMalloc(&ws, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

    // 初始化内存为0,确保读取测试的数据一致性
    cudaMemset(ws, 0, size_in_byte + MEMORY_OFFSET * BENCH_ITER);

    // 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.f;

    // warmup
    read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws, nullptr);
    write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws);
    copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(ws, ws + size_in_byte / 2);

    // read
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

    // write
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        write_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("write %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

    // copy
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        copy_kernel<BLOCK, LDG_UNROLL><<<grid / 2, BLOCK>>>(
            ws + i * MEMORY_OFFSET,
            ws + i * MEMORY_OFFSET + size_in_byte / 2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time_ms, start, stop);
    printf("copy %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

    printf("---------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(ws);
}

/**
 * @brief 主函数 - 驱动基准测试
 * 
 * 测试不同数据大小下的DRAM带宽:
 * - 从4MB开始,每次翻倍,最大到1GB
 * - 这样可以观察缓存层次对带宽的影响
 * 小数据可能受益于L2/L1缓存,大数据则更接近真实DRAM带宽
 */
int main() {
    size_t size = (1lu << 20) * 4;

    // 4MB~1GB,每次翻倍
    while (size <= (1lu << 30)) {
        benchmark(size);
        size *= 2;
    }

    return 0;
}
