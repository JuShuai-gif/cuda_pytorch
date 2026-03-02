/**
 * @file l2cache_bandwidth.cu
 * @brief L2缓存带宽基准测试
 * 
 * 本测试用于测量GPU L2缓存的访问带宽
 * L2缓存是所有SM共享的二级缓存,位于GPU芯片上
 * 
 * 关键技术点:
 * 1. 数据大小(2MB)小于L2缓存,确保数据在L2缓存中
 * 2. 预热阶段将数据加载到L2缓存
 * 3. 使用ldg_cg指令(缓存加载)访问数据
 * 4. 测量高迭代次数下的平均带宽
 * 
 * L2缓存特点:
 * - 延迟中等 (~30-50周期)
 * - 容量较大 (通常1-6MB)
 * - 所有SM共享
 */

#include <cstdio>

// 访问数据大小(字节),应小于L2缓存大小
// 2MB 通常小于大多数GPU的L2缓存
const int DATA_SIZE_IN_BYTE = (1lu << 20) * 2;
// LDG指令总数
const int N_LDG = (1lu << 20) * 512;

const int WARMUP_ITER = 200;
const int BENCH_ITER = 200;

/**
 * @brief 使用缓存加载指令读取全局内存
 * 
 * ld.global.cg.b32 是全局内存缓存加载指令
 * 数据会进入L2缓存(L1可选),适合读多写少的场景
 * 
 * @param ptr 读取地址
 * @return int 读取的32位数据
 */
__device__ __forceinline__ int ldg_cg(const void *ptr) {
    int ret;
    asm volatile(
        "ld.global.cg.b32 %0, [%1];"
        : "=r"(ret)
        : "l"(ptr));

    return ret;
}

/**
 * @brief L2缓存带宽测试内核
 * 
 * 使用多个线程同时访问L2缓存中的数据:
 * 1. 每个线程通过UNROLL展开加载多个数据
 * 2. 使用BLOCK * UNROLL确保足够的并行度
 * 3. 使用取模运算实现数据循环访问
 * 
 * @tparam BLOCK 每block线程数
 * @tparam UNROLL 展开因子
 * @tparam N_DATA 数据总量
 * @param x 输入数据指针
 * @param y 输出指针(用于防止优化)
 */
template <int BLOCK, int UNROLL, int N_DATA>
__global__ void kernel(const int *x, int *y) {
    int offset = (BLOCK * UNROLL * blockIdx.x + threadIdx.x) % N_DATA;
    const int *ldg_ptr = x + offset;
    int reg[UNROLL];

#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
    }

    int sum = 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        sum += reg[i];
    }

    if (sum != 0) {
        *y = sum;
    }
}

/**
 * @brief 主函数 - L2缓存带宽基准测试
 * 
 * 测试流程:
 * 1. 分配2MB数据(小于L2缓存)
 * 2. 预热200次,将数据加载到L2缓存
 * 3. 计时测量200次迭代的总时间
 * 4. 计算带宽: 数据总量 / 时间
 * 
 * 带宽计算:
 * - 每次迭代访问 N_LDG * sizeof(int) 字节
 * - 总数据量 = N_LDG * sizeof(int) * BENCH_ITER
 * - 带宽 = 总数据量 / 总时间(秒)
 */
int main() {
    const int N_DATA = DATA_SIZE_IN_BYTE / sizeof(int);

    const int UNROLL = 16;
    const int BLOCK = 128;

    static_assert(N_DATA >= UNROLL * BLOCK && N_DATA % (UNROLL * BLOCK) == 0,
                  "UNROLL or BLOCK is invalid");

    int *x, *y;
    cudaMalloc(&x, N_DATA * sizeof(int));
    cudaMalloc(&y, N_DATA * sizeof(int));
    cudaMemset(x, 0, N_DATA * sizeof(int));

    int grid = N_LDG / UNROLL / BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热:将数据加载到L2缓存
    for (int i = 0; i < WARMUP_ITER; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
    }

    // 实际基准测试
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
    }
    cudaEventRecord(stop);

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    double gbps = ((double)(N_LDG * sizeof(int)) / 1e9) / ((double)time_ms / BENCH_ITER / 1e3);
    printf("L2 cache bandwidth: %fGB/s\n", gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(x);
    cudaFree(y);

    return 0;
}