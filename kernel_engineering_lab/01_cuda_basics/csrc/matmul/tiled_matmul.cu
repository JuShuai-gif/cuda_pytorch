#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// ============================================================================
// Tiled GEMM (通用矩阵乘法) — 工业级 CUDA kernel
// ============================================================================
// 工业背景：
//   cuBLAS 和 CUTLASS 的核心就是 tiled GEMM。每个 AI 推理/训练框架的底层
//   都是 matmul kernel。Tiled matmul 使用 shared memory 做数据复用，
//   减少 global memory 带宽需求，是 GPU 编程的门面担当。
//
// 算法概述：
//   C[M,N] = A[M,K] @ B[K,N]
//   - 每个 thread block 负责计算 C 的一个 BM×BN tile
//   - 沿 K 维度以 BK 为步长循环加载 A 和 B 的 tile 到 shared memory
//   - 累加器使用 float 保证精度，最终写入时转为 fp16
//   - 每个线程计算 TM×TN 个输出元素（thread coarsening，减少线程开销）
//
// 参数说明：
//   BM — 输出 tile 的行数（thread block 处理的 M 方向范围）
//   BN — 输出 tile 的列数（thread block 处理的 N 方向范围）
//   BK — K 维度 tile 大小（shared memory 缓存的 K 方向范围）
//   TM — 每个线程计算的输出行数（thread coarsening）
//   TN — 每个线程计算的输出列数（thread coarsening）
//   BM/TM × BN/TN = 每个 thread block 的线程数
//
// 默认配置：BM=128, BN=128, BK=32, TM=8, TN=8
//   线程数 = (128/8) × (128/8) = 16 × 16 = 256
//   Shared memory = (BM×BK + BK×BN) × sizeof(half)
//                 = (128×32 + 32×128) × 2 = 8KB + 8KB = 16KB
//   每个线程计算 64 个输出元素，256 线程 × 64 = 16384 = BM × BN ✓
// ============================================================================

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err_ = (err);                                            \
        if (err_ != cudaSuccess) {                                           \
            throw std::runtime_error(                                        \
                std::string("CUDA error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + " - " +                          \
                cudaGetErrorString(err_));                                   \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// 核心 tiled GEMM kernel
// ---------------------------------------------------------------------------
// 启动配置：
//   grid  = (ceil(N/BN), ceil(M/BM))  — x 对应 N 方向，y 对应 M 方向
//   block = (BM/TM) * (BN/TN) = 256 个线程
//
// 线程分工：
//   16×16 的虚拟线程网格，每个线程负责输出 TM×TN=8×8 子块
//   thread_row = tid / 16  — 0..15（输出块内的行索引，步长为 TM）
//   thread_col = tid % 16  — 0..15（输出块内的列索引，步长为 TN）
//
// Shared memory 布局：
//   As[BM * BK]  — A tile 的行主序存储
//   Bs[BK * BN]  — B tile 的行主序存储
//
// 数据加载策略：
//   使用 1D 线程索引 strided 加载 —— 相邻线程访问连续的全局内存地址，
//   保证 coalesced global memory access（最大带宽利用率）。
//   每个线程加载 (BM*BK)/256 = 16 个 A 元素和 (BK*BN)/256 = 16 个 B 元素。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void tiled_matmul_kernel(
    const __half* __restrict__ A,  // [M, K] 行主序
    const __half* __restrict__ B,  // [K, N] 行主序
    __half* __restrict__ C,        // [M, N] 行主序
    int M, int N, int K)
{
    // block 索引 —— 确认当前 block 负责 C 的哪个 tile
    const int bx = blockIdx.x;  // N 方向的 tile 索引：0 .. ceil(N/BN)-1
    const int by = blockIdx.y;  // M 方向的 tile 索引：0 .. ceil(M/BM)-1

    // 当前 block 负责的输出 tile 起始位置
    const int c_row_start = by * BM;
    const int c_col_start = bx * BN;

    // 线程索引
    const int tid = threadIdx.x;

    // 每个输出 tile 内的线程网格维度
    // 线程网格：(BM/TM) × (BN/TN) = 16 × 16 = 256
    constexpr int THREAD_COLS = BN / TN;  // 16
    const int thread_row = tid / THREAD_COLS;  // 0 .. 15（当前线程负责的子块行）
    const int thread_col = tid % THREAD_COLS;  // 0 .. 15（当前线程负责的子块列）

    // 当前线程负责的输出子块的起始位置
    const int out_row_start = c_row_start + thread_row * TM;
    const int out_col_start = c_col_start + thread_col * TN;

    // -----------------------------------------------------------------------
    // Shared memory 声明 —— 缓存 A 和 B 的 tile，实现数据复用
    // 尺寸：As[128×32]=4096, Bs[32×128]=4096 half 元素 = 各 8KB
    // 总 shared memory = 16KB，远小于 A100 的 164KB 限制
    // -----------------------------------------------------------------------
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    // -----------------------------------------------------------------------
    // 累加器 —— 使用 float 精度累加以减少舍入误差
    // 每个线程累加 TM×TN = 64 个输出元素
    // -----------------------------------------------------------------------
    float accum[TM][TN] = {{0.0f}};

    // -----------------------------------------------------------------------
    // 主循环 —— 沿 K 维度以 BK 为步长遍历
    // k_block 是当前 K tile 的起始位置（全局 K 索引）
    // -----------------------------------------------------------------------
    for (int k_block = 0; k_block < K; k_block += BK) {

        // ====================================================================
        // Step 1：协同加载 A tile [BM×BK] 到 shared memory As
        // ====================================================================
        // 加载策略：每个线程以 stride = blockDim.x 加载多个元素
        // 线程 0→元素 0, 1→元素 1, ..., 31→元素 31（同一行连续地址 → coalesced）
        // 线程 32→元素 32（下一行第一个），形成 32-thread 的 block-row 对齐
        // 由于 BK=32，正好是一个 warp 覆盖 A 矩阵的一整行，coalescing 完美
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            const int ai = i / BK;       // As 中的局部行索引：0 .. BM-1
            const int ak = i % BK;       // As 中的局部列索引（K 偏移）：0 .. BK-1
            const int global_row = c_row_start + ai;
            const int global_col = k_block + ak;
            if (global_row < M && global_col < K) {
                As[i] = A[global_row * K + global_col];
            } else {
                As[i] = __float2half(0.0f);  // 边界填充零
            }
        }

        // ====================================================================
        // Step 2：协同加载 B tile [BK×BN] 到 shared memory Bs
        // ====================================================================
        // 加载策略同 A，线程 0..127 可覆盖 B tile 的一整行（BN=128）
        // 连续线程访问连续地址，保证 coalesced global memory access
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            const int bk = i / BN;       // Bs 中的局部行索引（K 偏移）：0 .. BK-1
            const int bj = i % BN;       // Bs 中的局部列索引：0 .. BN-1
            const int global_row = k_block + bk;
            const int global_col = c_col_start + bj;
            if (global_row < K && global_col < N) {
                Bs[i] = B[global_row * N + global_col];
            } else {
                Bs[i] = __float2half(0.0f);  // 边界填充零
            }
        }

        // 保证所有线程完成 shared memory 写入后再进入计算阶段
        __syncthreads();

        // ====================================================================
        // Step 3：计算当前 K tile 对输出子块的累积贡献
        // ====================================================================
        // 对 K tile 内的每个 k 偏移进行内积累加：
        //   accum[ii][jj] += As[thread_row*TM+ii][kk] * Bs[kk][thread_col*TN+jj]
        //
        // 访问模式分析（以默认配置为例，BM=BN=128, BK=32, TM=TN=8）：
        //   - As 访问：同一 warp 内所有线程访问相同的 As 元素（broadcast，无 bank conflict）
        //   - Bs 访问：连续线程访问步长为 8 的 Bs 元素（stride=16 bytes=4 banks，8 路广播）
        for (int kk = 0; kk < BK; kk++) {
            // 预加载当前 kk 对应的 A 值和 B 值
            // A：该线程负责的 TM 行在 kk 处的值
            __half a_vals[TM];
            #pragma unroll
            for (int ii = 0; ii < TM; ii++) {
                a_vals[ii] = As[(thread_row * TM + ii) * BK + kk];
            }

            // B：该线程负责的 TN 列在 kk 处的值
            __half b_vals[TN];
            #pragma unroll
            for (int jj = 0; jj < TN; jj++) {
                b_vals[jj] = Bs[kk * BN + (thread_col * TN + jj)];
            }

            // 外积累积：每个线程做 TM×TN 次乘加（FMA）
            #pragma unroll
            for (int ii = 0; ii < TM; ii++) {
                float a_f = __half2float(a_vals[ii]);
                #pragma unroll
                for (int jj = 0; jj < TN; jj++) {
                    accum[ii][jj] += a_f * __half2float(b_vals[jj]);
                    // 注：CUDA 编译器会将此模式自动转为 FMA 指令
                    //     （fused multiply-add），每个 clock cycle 执行一次
                }
            }
        }

        // 保证所有线程完成计算后再加载下一个 K tile 的 shared memory
        // 避免覆盖还未被读取的 shared memory
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Step 4：将 float 累加结果转换为 fp16 并写入全局内存 C
    // -----------------------------------------------------------------------
    // 边界检查：跳过 M / N 方向超出矩阵范围的输出元素
    for (int ii = 0; ii < TM; ii++) {
        const int row = out_row_start + ii;
        if (row >= M) continue;
        #pragma unroll
        for (int jj = 0; jj < TN; jj++) {
            const int col = out_col_start + jj;
            if (col >= N) continue;
            C[row * N + col] = __float2half_rn(accum[ii][jj]);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch GEMM kernel —— 支持批量矩阵乘法
// ---------------------------------------------------------------------------
// grid  = (ceil(N/BN), ceil(M/BM), B)  — 添加 batch 维度到 blockIdx.z
// A/B/C 均为 [B, M, K], [B, K, N], [B, M, N]
// 每个 block 通过 blockIdx.z 确定处理哪个 batch
//
// 模板参数含义与 tiled_matmul_kernel 完全一致。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void batched_matmul_kernel(
    const __half* __restrict__ A,  // [B, M, K]
    const __half* __restrict__ B,  // [B, K, N]
    __half* __restrict__ C,        // [B, M, N]
    int BATCH, int M, int N, int K)
{
    const int batch_idx = blockIdx.z;  // 当前 batch
    const int bx = blockIdx.x;         // N 方向 tile 索引
    const int by = blockIdx.y;         // M 方向 tile 索引

    const int c_row_start = by * BM;
    const int c_col_start = bx * BN;

    const int tid = threadIdx.x;
    constexpr int THREAD_COLS = BN / TN;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int out_row_start = c_row_start + thread_row * TM;
    const int out_col_start = c_col_start + thread_col * TN;

    // 计算当前 batch 的 A/B/C 偏移量
    // A[b] 的起始地址 = A + b × M × K
    const __half* A_batch = A + (size_t)batch_idx * M * K;
    const __half* B_batch = B + (size_t)batch_idx * K * N;
    __half* C_batch = C + (size_t)batch_idx * M * N;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    float accum[TM][TN] = {{0.0f}};

    for (int k_block = 0; k_block < K; k_block += BK) {
        // 加载 A tile
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            const int ai = i / BK;
            const int ak = i % BK;
            const int global_row = c_row_start + ai;
            const int global_col = k_block + ak;
            if (global_row < M && global_col < K) {
                As[i] = A_batch[global_row * K + global_col];
            } else {
                As[i] = __float2half(0.0f);
            }
        }

        // 加载 B tile
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            const int bk = i / BN;
            const int bj = i % BN;
            const int global_row = k_block + bk;
            const int global_col = c_col_start + bj;
            if (global_row < K && global_col < N) {
                Bs[i] = B_batch[global_row * N + global_col];
            } else {
                Bs[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // 内积累积
        for (int kk = 0; kk < BK; kk++) {
            __half a_vals[TM];
            #pragma unroll
            for (int ii = 0; ii < TM; ii++) {
                a_vals[ii] = As[(thread_row * TM + ii) * BK + kk];
            }

            __half b_vals[TN];
            #pragma unroll
            for (int jj = 0; jj < TN; jj++) {
                b_vals[jj] = Bs[kk * BN + (thread_col * TN + jj)];
            }

            #pragma unroll
            for (int ii = 0; ii < TM; ii++) {
                float a_f = __half2float(a_vals[ii]);
                #pragma unroll
                for (int jj = 0; jj < TN; jj++) {
                    accum[ii][jj] += a_f * __half2float(b_vals[jj]);
                }
            }
        }

        __syncthreads();
    }

    // 写回 C
    for (int ii = 0; ii < TM; ii++) {
        const int row = out_row_start + ii;
        if (row >= M) continue;
        #pragma unroll
        for (int jj = 0; jj < TN; jj++) {
            const int col = out_col_start + jj;
            if (col >= N) continue;
            C_batch[row * N + col] = __float2half_rn(accum[ii][jj]);
        }
    }
}

// ============================================================================
// Kernel Wrapper 函数 —— 从 PyTorch 调用
// ============================================================================

// ---------------------------------------------------------------------------
// 辅助函数：根据 M/N/K 大小选择最优的 tile 配置
// ---------------------------------------------------------------------------
// 选择策略：
//   - 大矩阵（min(M,N) >= 128）：BM=128, BN=128, BK=32（高并行度，256 线程）
//   - 中等矩阵（min(M,N) >= 64）：BM=64, BN=64, BK=32（中等并行度，64 线程）
//   - 小矩阵：BM=32, BN=32, BK=32（低延迟，32 线程）
// 这种自适应策略适用于 LLM 推理中不同阶段的矩阵大小差异。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
void launch_tiled_matmul_configured(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream)
{
    constexpr int THREAD_COUNT = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREAD_COUNT);

    tiled_matmul_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_batched_matmul_configured(
    const __half* A, const __half* B, __half* C,
    int BATCH, int M, int N, int K,
    cudaStream_t stream)
{
    constexpr int THREAD_COUNT = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    dim3 block(THREAD_COUNT);

    batched_matmul_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, BATCH, M, N, K);
}

// ---------------------------------------------------------------------------
// dispatch_tiled_matmul —— 根据矩阵大小选择最优配置并 launch
// ---------------------------------------------------------------------------
static void dispatch_tiled_matmul(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    int min_mn = min(M, N);

    if (min_mn >= 128) {
        // 大矩阵：128×128 tile，256 线程，最高并行度
        launch_tiled_matmul_configured<128, 128, 32, 8, 8>(
            A, B, C, M, N, K, stream);
    } else if (min_mn >= 64) {
        // 中等矩阵：64×64 tile，64 线程
        launch_tiled_matmul_configured<64, 64, 32, 8, 8>(
            A, B, C, M, N, K, stream);
    } else {
        // 小矩阵：32×32 tile，32 线程（或更低的线程开销）
        launch_tiled_matmul_configured<32, 32, 32, 8, 8>(
            A, B, C, M, N, K, stream);
    }
}

// ---------------------------------------------------------------------------
// dispatch_batched_matmul —— batch 版本的自适应 dispatch
// ---------------------------------------------------------------------------
static void dispatch_batched_matmul(
    const __half* A, const __half* B, __half* C,
    int BATCH, int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    int min_mn = min(M, N);

    if (min_mn >= 128) {
        launch_batched_matmul_configured<128, 128, 32, 8, 8>(
            A, B, C, BATCH, M, N, K, stream);
    } else if (min_mn >= 64) {
        launch_batched_matmul_configured<64, 64, 32, 8, 8>(
            A, B, C, BATCH, M, N, K, stream);
    } else {
        launch_batched_matmul_configured<32, 32, 32, 8, 8>(
            A, B, C, BATCH, M, N, K, stream);
    }
}

// ============================================================================
// 对外接口 —— 由 bindings.cpp 或 Python 直接调用
// ============================================================================

void run_tiled_matmul(
    torch::Tensor A,   // [M, K] fp16
    torch::Tensor B,   // [K, N] fp16
    torch::Tensor C)   // [M, N] fp16（原地输出）
{
    // 参数校验
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be fp16");
    TORCH_CHECK(C.scalar_type() == torch::kHalf, "C must be fp16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D: [M, K]");
    TORCH_CHECK(B.dim() == 2, "B must be 2D: [K, N]");
    TORCH_CHECK(A.size(1) == B.size(0),
        "A.shape[1] (K=%ld) must match B.shape[0] (K=%ld)",
        A.size(1), B.size(0));
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(1),
        "C shape [%ld,%ld] must match [M=%ld,N=%ld]",
        C.size(0), C.size(1), A.size(0), B.size(1));

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);

    const __half* A_ptr = reinterpret_cast<const __half*>(A.data_ptr<torch::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(B.data_ptr<torch::Half>());
    __half* C_ptr = reinterpret_cast<__half*>(C.data_ptr<torch::Half>());

    dispatch_tiled_matmul(A_ptr, B_ptr, C_ptr, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void run_batched_matmul(
    torch::Tensor A,   // [B, M, K] fp16
    torch::Tensor B,   // [B, K, N] fp16
    torch::Tensor C)   // [B, M, N] fp16（原地输出）
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be fp16");
    TORCH_CHECK(C.scalar_type() == torch::kHalf, "C must be fp16");
    TORCH_CHECK(A.dim() == 3, "A must be 3D: [B, M, K]");
    TORCH_CHECK(B.dim() == 3, "B must be 3D: [B, K, N]");
    TORCH_CHECK(A.size(2) == B.size(1),
        "A.shape[2] (K=%ld) must match B.shape[1] (K=%ld)",
        A.size(2), B.size(1));
    TORCH_CHECK(A.size(0) == B.size(0),
        "A and B must have same batch size");
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == A.size(1) && C.size(2) == B.size(2),
        "C shape [%ld,%ld,%ld] must match [B=%ld,M=%ld,N=%ld]",
        C.size(0), C.size(1), C.size(2), A.size(0), A.size(1), B.size(2));

    const int BATCH = (int)A.size(0);
    const int M = (int)A.size(1);
    const int K = (int)A.size(2);
    const int N = (int)B.size(2);

    const __half* A_ptr = reinterpret_cast<const __half*>(A.data_ptr<torch::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(B.data_ptr<torch::Half>());
    __half* C_ptr = reinterpret_cast<__half*>(C.data_ptr<torch::Half>());

    dispatch_batched_matmul(A_ptr, B_ptr, C_ptr, BATCH, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
