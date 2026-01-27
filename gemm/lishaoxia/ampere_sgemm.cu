#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j + p * n];
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}


// 把 C/C++ 中的 shared memory 指针转换成 PTX 需要的 32-bit shared 地址
/*
在PTX中：
- shared memory地址是32-bit
- C指针是generic address(64-bit)
*/
__device__ __forceinline__ 
uint32_t smem_u32addr(const void* smem_ptr){
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"            // 声明一个64位寄存器，用来存放shared地址的64位形式
        "cvta.to.shared.u64 u64addr, %1;\n" // 将 generic address -> shared address
        "cvt.u32.u64 %0, u64addr;}\n"       // shared 地址在硬件上是32-bit 截断为u32,%0 对应 addr
        : "=r"(addr)                        // 
        : "l"(smem_ptr)
    );
    return addr;
}

// 如果 guard 为真，聪global memory读取4字节(32bit),异步拷贝到shared memory
/*
@p cp.async.ca.shared.global.L2::128B [%0], [%1], 4;

| 字段              | 含义                  |
| --------------- | ------------------- |
| `@p`            | predicate 执行        |
| `cp.async`      | 异步 copy             |
| `ca`            | cache at all levels |
| `shared.global` | global → shared     |
| `L2::128B`      | L2 cache line hint  |
| `[%0]`          | shared 32-bit 地址    |
| `[%1]`          | global 64-bit 指针    |
| `4`             | 传输字节数               |
*/
__device__ __forceinline__ 
void ldgsts32(const uint32_t &smem_addr,
            const void*gmem_ptr,
            bool guard){
    asm volatile(
        "{.reg .pred p;\n"      // 声明 predicate 寄存器
        " step.ne.b32 p, %2, 0;\n"  // 用 predicate 代替 if 分支（避免 warp divergence）
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4  // CUDA于11.4
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4;}\n"
#else
        " @p cp.async.ca.shared.global [%0], [%1], 4;}\n"
#endif
        : : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard)
    );
}

// 带 size 的版本(支持越界安全)
/*
%2 = src_size
硬件语义：
- 如果 src_size < 4
- 不足部分自动填 0
- 不会触发非法访存

这是 FlashAttention / CUTLASS 常用的 边界处理方式
*/
__device__ __forceinline__
void ldgsts32(const uint32_t &smem_addr,
              const void *gmem_ptr,
              const uint32_t &src_size,
              bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %3, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4
        " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;}\n"
#else
        " @p cp.async.ca.shared.global [%0], [%1], 4, %2;}\n"
#endif
        : : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size), "r"((int)guard)
    );
}

// 等待当前 CTA 中所有 cp.async 完成
/*
cp.async.wait_all
    - 阻塞当前 warp
    - 确保 shared memory 数据可用

通常在：
    - pipeline stage 切换
    - 使用 shared 数据前

*/
__device__ __forceinline__ 
void ldgsts_commit(){
    asm volatile("cp.async.wait_all;\n"::);
}

// 条件性 global store（float）
__device__ __forceinline__ 
void stg32(const float &reg,void* ptr,bool guard){
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p,%2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr),"f"(reg), "r"((int)guard)
    );
}

/*
从 shared memory 连续读取 16 字节

映射到 4 个 float 寄存器

要求：

addr 16B 对齐
*/
__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile(
        "ld.shared.v4.f32 {%0, %1, %2, %3},[%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

/*
向 shared 写一个 float

无 predicate

常用于：

register → shared

reduction / staging
*/
__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

/*
一条指令写 16 字节

比 4 次 st.shared.f32：

指令数更少

带宽更高

要求：

addr 16B 对齐
*/
__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

// 这套代码的典型使用模式
/*
uint32_t smem = smem_u32addr(&smem_buf[offset]);

ldgsts32(smem, gmem_ptr, guard);
// 多个 cp.async

ldgsts_commit();   // 等待

lds128(r0, r1, r2, r3, smem);
*/

// C 寄存器分块 → 写回中间片段
struct StgFrag
{
    float data[4][4];   // 一个 4×4 FP32 子块，用于从 C_frag[16][8] 中切出一个 可写回的小 tile
    
    // tile_x ∈ {0,1} → n 方向
    // tile_y ∈ {0,1,2,3} → m 方向
    __device__ __forceinline__
    StgFrag(const float (&C_frag)[16][8],int tile_x,int tile_y){
        #pragma unroll
        for (int i = 0; i < 4; ++i){
            #pragma unroll
            for (int j = 0; j < 4; ++j){
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            } 
        }
    }
};

// 通用尾块写回（m / n 非整除）
__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
                float* C_stg_ptr,
                const float* C_lds_ptr,
                uint32_t C_sts_addr,
                uint32_t m,
                uint32_t n,
                uint32_t m_idx,
                uint32_t n_idx){
    __syncthreads();

    // STS128：寄存器 → shared memory
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // 每行 stride = 9×float4
        // 9 是 padding，避免 bank conflict（非常经典）
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 9 * sizeof(float4));
    }

    __syncthreads();
    
    // 36 = 9 * 4
    // m_guard 防止 m 越界
    // n_idx < n 防止 n 越界
    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 36],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n256k8
 * warp tile: m64n64k8
 * thread tile: m16n8k8
 * thread fragment:
 *     matrixA: 16x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                         256
 *                    --|---------------------------------------|
 *             B_tile  8|                                       |
 *                    --|---------------------------------------|
 *
 *  A_tile   | 8 |      |    64   |
 *         --|---|    --|---------|---------|---------|---------|
 *           |   |      |         |         |         |         |
 *           |   |    64|  warp0  |  warp1  |  warp2  |  warp3  |
 *           |   |      |         |         |         |         |
 *        128|   |    --|---------|---------|---------|---------|
 *           |   |      |         |         |         |         |
 *           |   |      |  warp4  |  warp5  |  warp6  |  warp7  |
 *           |   |      |         |         |         |         |
 *         --|---|      |---------|---------|---------|---------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */
// 说明：这是一个针对 Ampere 架构手写的 SGEMM（FP32）kernel
// Tile 形状：CTA = 128x256x8, Warp = 64x64x8, Thread = 16x8x8
// 核心目标：最大化 LDS/FFMA 吞吐，隐藏 LDG latency，避免 LDS broadcast 冲突
__global__ __launch_bounds__(256)
void ampere_sgemm_128x256x8_kernel(
    const float* A,// 输入矩阵 A，row-major, [m, k]
    const float* B,// 输入矩阵 B，row-major, [k, n]
    float* C,// 输出矩阵 C，row-major, [m, n]
    uint32_t m,// 矩阵 A/C 的行数
    uint32_t n,// 矩阵 B/C 的列数
    uint32_t k,// 矩阵 A 的列数 / B 的行数
    uint32_t B_ldg_step){   // B 指针在 K 方向前进一个 tile 的字节步长
    /*
    * ---------------- Shared Memory Layout ----------------
    * smem 总大小 32KB：
    * [0 , 16KB) : A tile double buffer
    * [16KB, 32KB) : B tile double buffer
    * 每个 tile 都是 K=8 的 slice
    * A: 132 x 8（padding=4 防止 bank conflict）
    * B: 256 x 8
    */
    __shared__ __align__(16*1024) char smem[32 * 1024];
    float* A_smem = reinterpret_cast<float*>(smem);
    float* B_smem = reinterpret_cast<float*>(smem + 16 * 1024);

    // ---------------- Register Fragments ----------------
    // A_frag: ping-pong buffer [2][16]
    // B_frag: ping-pong buffer [2][8]
    // C_frag: thread-level accumulator，16x8
    float A_frag[2][16];
    float B_frag[2][8];
    float C_frag[16][8];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }
    
    // ---------------- Thread / Warp ID ----------------
    const uint32_t lane_id = threadIdx.x % 32;// warp 内线程号
    const uint32_t warp_id = threadIdx.x / 32;// warp号(0-7)

    // ---------------- MMA Thread Mapping ----------------
    // 每个 warp 做 64x64
    // lane → (x,y) 子 tile，用于避免 LDS broadcast
    const uint32_t mma_tid_x = (lane_id / 2) % 8;   // N 方向
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);  // M 方向


    // ---------------- Global Load Pointer ----------------
    // A: 每个线程加载 4 个 FP32，跨 K 方向
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8) * k + threadIdx.x % 8);
    // B: 每个线程加载 2x FP32，跨 N 方向
    const char *B_ldg_ptr = (const char *)(
        B + (threadIdx.x / 128) * n + blockIdx.x * 256 + threadIdx.x % 128);

    // ---------------- A/B LDG Offset ----------------
    uint32_t A_ldg_offset[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        A_ldg_offset[i] = i * 32 * k * sizeof(float);
    }

    // B_ldg_offset
    uint32_t B_ldg_offset[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        B_ldg_offset[i] = i * 2 * n * sizeof(float);
    }
    
    // ---------------- Shared Memory Address ----------------
    // 使用 u32 addr，方便 xor 切换 double buffer
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8));
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 128) * 256 + (threadIdx.x % 128));
    
    // LDS：warp 内读取 A/B fragment
    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 4) * 64 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 4) * 64 + mma_tid_x * 4);

    // ---------------- Boundary Guard ----------------
    // A 的 M 方向 guard（4 行）
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 + i * 32;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }
    
    // B 的 N 方向 guard（左右 2 块）
    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n_idx = blockIdx.x * 256 + threadIdx.x % 128 + i * 128;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    // ---------------- K Tile Count ----------------
    // 先 load 第一个不满 8 的 tile
    uint32_t k_tiles = (k + 7) / 8 - 1;

    /* ================== First Tile Load ================== */
    {
        uint32_t first_k_tile = k - k_tiles*8;
        // Load A
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t src_size = threadIdx.x % 8 < first_k_tile ? 4 : 0;

            ldgsts32(A_sts_addr + i * 32 * sizeof(float),
                     A_ldg_ptr + A_ldg_offset[i],
                     src_size,
                     (A_ldg_guard & (1u << i)) != 0);
        }

        A_ldg_ptr += first_k_tile * sizeof(float);

        // Load B
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t src_size = i * 2 + threadIdx.x / 128 < first_k_tile ? 4 : 0;

            ldgsts32(B_sts_addr + i * 2 * 256 * sizeof(float),
                     B_ldg_ptr + B_ldg_offset[i],
                     src_size,
                     (B_ldg_guard & (1u << 0)) != 0);
            ldgsts32(B_sts_addr + (i * 2 * 256 + 128) * sizeof(float),
                     B_ldg_ptr + B_ldg_offset[i] + 128 * sizeof(float),
                     src_size,
                     (B_ldg_guard & (1u << 1)) != 0);
        }

        B_ldg_ptr += n * first_k_tile * sizeof(float);

        ldgsts_commit();
        __syncthreads();

        // double buffer switch
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x2000;
    }
    
    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(A_frag[0][8], A_frag[0][9], A_frag[0][10], A_frag[0][11],
           A_lds_addr + 32 * sizeof(float));
    lds128(A_frag[0][12], A_frag[0][13], A_frag[0][14], A_frag[0][15],
           A_lds_addr + 48 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                ldgsts_commit();
                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x2000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x2000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float);
                B_ldg_ptr += B_ldg_step;
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][8],
                   A_frag[(k_frag + 1) % 2][9],
                   A_frag[(k_frag + 1) % 2][10],
                   A_frag[(k_frag + 1) % 2][11],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 32) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][12],
                   A_frag[(k_frag + 1) % 2][13],
                   A_frag[(k_frag + 1) % 2][14],
                   A_frag[(k_frag + 1) % 2][15],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 48) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 256 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 256 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag < 4) {
                ldgsts32(A_sts_addr + k_frag * 32 * sizeof(float),
                         A_ldg_ptr + A_ldg_offset[k_frag],
                         (A_ldg_guard & (1u << k_frag)) != 0);

                ldgsts32(B_sts_addr + k_frag * 2 * 256 * sizeof(float),
                         B_ldg_ptr + B_ldg_offset[k_frag],
                         (B_ldg_guard & (1u << 0)) != 0);
                ldgsts32(B_sts_addr + (k_frag * 2 * 256 + 128) * sizeof(float),
                         B_ldg_ptr + B_ldg_offset[k_frag] + 128 * sizeof(float),
                         (B_ldg_guard & (1u << 1)) != 0);
            }

            // FFMA loop
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][8],
                   A_frag[(k_frag + 1) % 2][9],
                   A_frag[(k_frag + 1) % 2][10],
                   A_frag[(k_frag + 1) % 2][11],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 32) * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][12],
                   A_frag[(k_frag + 1) % 2][13],
                   A_frag[(k_frag + 1) % 2][14],
                   A_frag[(k_frag + 1) % 2][15],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 48) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 256 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 256 + 32) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 4096) +
                                       mma_tid_y * 4 * 9 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 4096) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 4 * 64;
    uint32_t n_idx = blockIdx.x * 256 + warp_id % 4 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + 64 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 9 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 36],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + 255) / 256, (m + 127) / 128);

    // warmup
    ampere_sgemm_128x256x8_kernel<<<grid, 256>>>(
        d_A, d_B, d_C, m, n, k,
        n * sizeof(float) * 8);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        ampere_sgemm_128x256x8_kernel<<<grid, 256>>>(
            d_A, d_B, d_C, m, n, k,
            n * sizeof(float) * 8);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}








































