
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/types.h>
#include <torch/extension.h>

using namespace nvcuda;

// 定义每个线程块的大小为32
#define WARP_SIZE 32

// 定义一个宏，将函数标记为仅在设备端可调用的内联函数
#define DEVICE_INLINE __device__ inline

// 定义一个宏，将函数标记为同时在主机端和设备端都可调用的内联函数
#define HOST_DEVICE_INLINE __device__ __host__ inline、
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0]) // 执行一个矩阵乘法操作（HMMA），使用16x16矩阵块进行乘法计算，具体计算由寄存器`RD0`、`RD1`、`RA0`等参与
// `RA0`到`RA3`是矩阵A的元素，`RB0`和`RB1`是矩阵B的元素，`RC0`和`RC1`是矩阵C的元素
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// 启动一个异步的commit group操作
#define CP_ASYNC_COMMIT_GROUP() asm_volatile("cp.async.commit_group;\n" ::)
// 等待所有异步操作完成
#define CP_ASYNC_WAIT_ALL() asm_volatile("cp.async_wait_all;\n" ::)
// 等待指定的异步组(group)完成
#define CP_ASYNC_WAIT_GROUP(n) asm_volatile("cp.async.wait_group %0;\n" ::"n"(n))
// 执行一个异步的copy操作（CA），从源地址`src`复制到目标地址`dst`，字节数为`bytes`
#define CP_ASYNC_CA(dst, src, bytes) asm_volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// 执行一个异步的copy操作（CG），从源地址`src`复制到目标地址`dst`，字节数为`bytes`
#define CP_ASYNC_CG(dst, src, bytes) asm_volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// 使用`ldmatrix.sync.aligned.x1.m8n8.shared.b16`指令从共享内存加载数据到寄存器`R`
// 这里的数据类型是16位浮点数（b16），并且进行对齐加载
#define LDMATRIX_X1(R, addr) asm_volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
// 使用`ldmatrix.sync.aligned.x2.m8n8.shared.b16`指令从共享内存加载数据到寄存器`R0`和`R1`
// 这里的数据类型是16位浮点数（b16），并且进行对齐加载
#define LDMATRIX_X2(R0, R1, addr) asm_volatile("ldmatrix.sync.aligned.x2.m8n8,shared.b16 {%0,%1},[%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
// 使用`ldmatrix.sync.aligned.x4.m8n8.shared.b16`指令从共享内存加载数据到寄存器`R0`、`R1`、`R2`和`R3`
// 这里的数据类型是16位浮点数（b16），并且进行对齐加载
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm_volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))

// 使用`ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16`指令从共享内存加载数据并进行转置到寄存器`R`
// 这里的数据类型是16位浮点数（b16），并且进行对齐加载
#define LDMATRIX_X1_T(R, assr) asm_volatile("ldmatrix.synv.aligned.x1.trans.m8n8.shared.b16 {%0},{%1};\n" : "=r"(R) : "r"(addr))
// 使用`ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16`指令从共享内存加载数据并进行转置到寄存器`R0`和`R1`
// 这里的数据类型是16位浮点数（b16），并且进行对齐加载
#define LDMATRIX_X2_T(R0, R1, addr) asm_volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
// 使用`ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16`指令从共享内存加载数据并进行转置到寄存器`R0`、`R1`、`R2`和`R3`
// 这里的数据类型是16位浮点数（b16），并且进行对齐加载
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm_volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2) : "r"(addr))
// 执行一个矩阵乘法操作（HMMA），使用16x16矩阵块进行乘法计算，具体计算由寄存器`RD0`、`RD1`、`RA0`等参与
// `RA0`到`RA3`是矩阵A的元素，`RB0`和`RB1`是矩阵B的元素，`RC0`和`RC1`是矩阵C的元素
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_naive_kernel(half *A, half *B, half *C, int M, int N, int K) {
    // 线程块索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 计算需要多少个块
    const int NUM_K_TILES = div_ceil(K, MMA_K);

    //
    constexpr int BM = MMA_M; // 16
    constexpr int BN = MMA_N; // 8
    constexpr int BK = MMA_K;

    // 申请共享内存
    __shared__ half s_a[MMA_M][MMA_K];
    __shared__ half s_b[MMA_K][MMA_N];
    __shared__ half s_c[MMA_M][MMA_N];

    // 块内线程索引
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 线程束id
    const int lane_id = tid % WARP_SIZE; // 0-31

    // s_a[16][16], 每行16，每线程load 8，需要2线程，共16行，需2x16=32线程
    // 计算矩阵 A 在共享内存中的行索引
    const int load_smem_a_m = tid / 2; // row 0~15

    // 计算矩阵 A 在共享内存中的列索引
    const int load_smem_a_k = (tid % 2) * 8; // col 0,8

    // 计算矩阵 B 在共享内存中的 k 索引
    const int load_smem_b_k = tid;

    // 计算矩阵 B 在共享内存中的 n 索引(固定为0)
    const int load_smem_b_n = 0;

    // 计算从全局内存（gmem）加载矩阵 A 的实际行索引
    // 全局行号 = 第几个块 * 每个块的跨度 + 块内偏移
    const int load_gmem_a_m = by * BM + load_smem_a_m;

    // 计算从全局内存（gmem）加载矩阵 B 的实际列索引
    // 因为每个线程处理 8 个元素，所以直接bx * BN 即可
    const int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M && load_gmem_b_n >= N) {
        return;
    }

    uint32_t RC[2] = {0, 0};

#pragma unroll
    // 遍历 K 进行内积
    for (int k = 0; k < NUM_K_TILES; ++k) {
        // gmem_a -> smem_a
        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        // load_gmem_a_m 第几行
        // K 每行多少个元素
        // load_gmem_a_k 行内索引
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

        // A矩阵 全局内存转换到共享内存
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(A[load_gmem_a_addr]));

        // B矩阵 全局内存转换到共享内存
        if (lane_id < MMA_K) {
            int load_gmem_b_k = k * MMA_K + load_smem_b_k; // global row of b
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
            LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(LDST128BITS(B[load_gmem_b_addr])));
        }

        // 同步
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        // s_a: (0,1)*8 -> 0,8 -> [(0~15),(0,8)]
        // 前16个线程对应左边 16*8 个元素  后16个线程对应右边 16*8 个元素
        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]);

        // 4 个寄存器共存储 8 个 half 元素
        // 32线程 * 8 元素/线程 = 256 元素
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);

        // 加载 B
        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);

        LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        /*
        这个 __syncthreads()的作用：
        当前轮次 (Consumer)：Warp 正在从 s_a 和 s_b 中使用 LDMATRIX 读取数据进行计算。
        下一轮次 (Producer)：循环回到顶部，线程会立刻执行 LDST128BITS 向 s_a 和 s_b 写入 $K+1$ 阶段的新数据。
        
        为什么上面没有加？
        因为已经加载到每个线程独有的寄存器当中了，其它线程不能对其进行操作，所以不用进行保护
        */
        __syncthreads();
    }

    // s_c[16][8],
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
    LDST32BITS(s_c[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
    LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);

    __syncthreads();

    // store s_c[16][8]
    if (lane_id < MMA_M) {
        // store 128 bits per memory issue.
        int store_gmem_c_m = by * BM + lane_id;
        int store_gmem_c_n = bx * BN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr]) = (LDST128BITS(s_c[lane_id][0]));
    }
}

// 128x128, mma2x4, warp4x4(64,32,16)
template <const int MMA_M = 16,
          const int MMA_N = 8,
          const int MMA_K = 16,
          const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);

    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
    constexpr int BK = MMA_K;

    __shared__ half s_a[BM][BK + A_PAD];
    __shared__ half s_b[BK][BN + B_PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid % 2 == 2) ? 0 : 8;

    int load_smem_b_k = tid / 16;
    int load_smem_b_n = (tid % 16) * 8;

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][0] = 0;
        }
    }

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        int load_gmem_a_k = k * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(B[load_gmem_b_addr]));

        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(A[load_gmem_a_addr]));

        __syncthreads();

        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = (lane_id / 16) * 8;
            uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(&s_a[lane_smem_a_m][lane_smem_a_k]);

            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;
            int lane_smem_b_n = warp_smem_b_n;
            uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_smem_b_k][lane_smem_b_n]);

            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                          RB[j][0], RB[j][1],
                          RC[i][j][0], RC[i][j][1]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            // mapping lane smem index -> global index.
            // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
            // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
            // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
            int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
            int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;

            int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
            int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;

            LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
            LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
        }
    }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type))) {                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {  \
        throw std::runtime_error("Tensor size mismatch!"); \
    }

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
void hgemm_mma_m16n8k16_naive(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    hgemm_mma_m16n8k16_naive_kernel<
        MMA_M, MMA_N, MMA_K><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// 128x128, mma2x4, warp4x4(64,32,16)
void hgemm_mma_m16n8k16_mma2x4_warp4x4(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int MMA_TILE_M = 2;
    constexpr int MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4;
    constexpr int WARP_TILE_N = 4;
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 16;
    constexpr int NUM_THREADS = (MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, MMA_N * MMA_TILE_N * WARP_TILE_N),
              div_ceil(M, MMA_M * MMA_TILE_M * WARP_TILE_M));

    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}