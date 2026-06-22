// ============================================================================
// 02_cutlass_kernel_instantiation.cpp - 模拟 CUTLASS Kernel
//                                        显式实例化
// ============================================================================
//
// 目的:
//   演示 CUTLASS 如何跨多个编译单元管理 kernel 模板实例化。
//   这是一个关键的构建系统集成模式，将接口（头文件）
//   与实现（显式实例化的 .cu 文件）分离。
//
// CUTLASS KERNEL 实例化模型:
//
//   1. gemm.h          - 公共 API: kernel 声明、extern 模板
//   2. gemm_impl.h      - 内部: kernel 实现细节
//   3. sm80_gemm.cu     - SM80 GPU 的显式实例化
//   4. sm90_gemm.cu     - SM90 GPU 的显式实例化
//   5. 用户的 main.cu   - 使用 extern 模板，链接到 (3) 和 (4)
//
// 本文件在单个文件中模拟整个模型。
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <cstdint>

// ============================================================================
// 第 1 节: 类型定义（模拟 CUDA 类型）
// ============================================================================

using half_t   = uint16_t;      // 半精度浮点
using float_t  = float;         // 单精度
using int32_t  = int;
using index_t  = int;

// 模拟 CUDA 内置变量
inline int   threadIdx_x()     { return 0; }
inline int   blockIdx_x()      { return 0; }
inline int   blockDim_x()      { return 32; }
inline int   gridDim_x()       { return 1; }
inline void  syncthreads()     { /* 模拟中无操作 */ }
inline void  syncwarp()        { /* 模拟中无操作 */ }

// ============================================================================
// 第 2 节: 架构特定配置
// ============================================================================

/// \brief 架构标签系统 — 不同 GPU 代次的标识符。
/// 这些是编译期标记，用于选择最优的 tile 大小和
/// 硬件特性。
struct Sm80Tag {  // A100 (Ampere)
    static constexpr int kComputeCapability  = 80;
    static constexpr int kSharedMemKB        = 163;  // 最大共享内存
    static constexpr int kMaxThreadsPerBlock = 1024;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasAsyncCopy      = true;  // cp.async
    static constexpr const char* name() { return "SM80 (A100)"; }
};

struct Sm90Tag {  // H100 (Hopper)
    static constexpr int kComputeCapability  = 90;
    static constexpr int kSharedMemKB        = 227;  // 最大共享内存
    static constexpr int kMaxThreadsPerBlock = 1024;
    static constexpr bool kHasTensorCore     = true;
    static constexpr bool kHasAsyncCopy      = true;
    static constexpr bool kHasTMA            = true;   // Tensor 内存加速器
    static constexpr bool kHasWGMMA          = true;   // Warp Group MMA
    static constexpr const char* name() { return "SM90 (H100)"; }
};

// ============================================================================
// 第 3 节: Kernel 配置（Tile 大小等）
// ============================================================================
//
// 在 CUTLASS 中，kernel 配置是一个模板特化，
// 编码了 tile 维度、线程布局和流水线阶段数。
// 不同架构得到不同的配置。

/// \brief 通用 kernel 配置（回退）。
template <typename ArchTag, typename ElementA, typename ElementB,
          typename ElementC, typename AccumT = float_t>
struct KernelConfig {
    using Arch    = ArchTag;
    using ElemA   = ElementA;
    using ElemB   = ElementB;
    using ElemC   = ElementC;
    using Accum   = AccumT;

    static constexpr int kThreads    = 256;
    static constexpr int kTileM      = 128;
    static constexpr int kTileN      = 128;
    static constexpr int kTileK      = 16;
    static constexpr int kStages     = 3;
    static constexpr const char* config_name() { return "通用"; }
};

/// \brief SM80 特定 kernel 配置，使用更大的 tile。
template <typename ElementA, typename ElementB,
          typename ElementC, typename AccumT>
struct KernelConfig<Sm80Tag, ElementA, ElementB, ElementC, AccumT> {
    using Arch    = Sm80Tag;
    using ElemA   = ElementA;
    using ElemB   = ElementB;
    using ElemC   = ElementC;
    using Accum   = AccumT;

    static constexpr int kThreads    = 256;
    static constexpr int kTileM      = 256;   // A100 上更大的 tile
    static constexpr int kTileN      = 128;
    static constexpr int kTileK      = 32;    // 更深的 K tile
    static constexpr int kStages     = 4;     // 更多软件流水线阶段
    static constexpr const char* config_name() { return "SM80-优化"; }
};

/// \brief SM90 特定 kernel 配置，使用 TMA 感知的设置。
template <typename ElementA, typename ElementB,
          typename ElementC, typename AccumT>
struct KernelConfig<Sm90Tag, ElementA, ElementB, ElementC, AccumT> {
    using Arch    = Sm90Tag;
    using ElemA   = ElementA;
    using ElemB   = ElementB;
    using ElemC   = ElementC;
    using Accum   = AccumT;

    static constexpr int kThreads    = 256;
    static constexpr int kTileM      = 256;   // H100 上更大
    static constexpr int kTileN      = 256;
    static constexpr int kTileK      = 64;    // 使用 TMA 的更深的 K
    static constexpr int kStages     = 5;     // 更深的流水线
    static constexpr const char* config_name() { return "SM90-TMA-优化"; }
};

// ============================================================================
// 第 4 节: Kernel 模板（声明）
// ============================================================================
//
// 这是用户在公共头文件（gemm.h）中看到的内容。
// 它声明了 kernel 类模板，但不在此处
// 定义实现。

template <typename KernelConfigT>
class GemmKernel {
public:
    using Config = KernelConfigT;

    /// \brief 启动 GEMM kernel。
    /// 在真实 CUTLASS 中，这是从用户代码调用的入口点。
    /// 实现在提供显式实例化的 .cu 文件中。
    static void launch(
        int M, int N, int K,
        typename Config::ElemA const* A, index_t lda,
        typename Config::ElemB const* B, index_t ldb,
        typename Config::ElemC*       C, index_t ldc,
        typename Config::Accum alpha = 1,
        typename Config::Accum beta  = 0
    );

    /// \brief 描述此 kernel 配置。
    static void describe();
};

// ============================================================================
// 第 5 节: extern 模板声明（公共头文件）
// ============================================================================
//
// 这些告诉编译器："不要隐式实例化这些
// 特化。它们将由显式实例化单元提供。"

extern template class GemmKernel<
    KernelConfig<Sm80Tag, half_t, half_t, half_t>>;

extern template class GemmKernel<
    KernelConfig<Sm80Tag, half_t, half_t, float_t>>;

extern template class GemmKernel<
    KernelConfig<Sm90Tag, half_t, half_t, half_t>>;

extern template class GemmKernel<
    KernelConfig<Sm90Tag, half_t, half_t, float_t>>;

// ============================================================================
// 第 6 节: Kernel 实现（模拟 .cu 文件）
// ============================================================================
//
// 在真实 CUTLASS 中，这部分会在单独的 .cu 文件中
//（例如 tools/library/gemm_sm80_half.cu）。它由 NVCC
// 编译一次，然后与用户代码链接。
//
// 实现使用架构特定的内联函数、
// 内联 PTX、共享内存声明和启动边界。

namespace kernel_impl {

    // --- 6a. Device 函数: tile 级 GEMM 计算 ---
    // 在真实 CUDA 中，这会是一个 __device__ 函数，
    // 带有用于 Tensor Core 操作的内联 PTX。

    template <typename Config>
    void device_gemm_tile(
        typename Config::ElemA const* A_smem,  // 共享内存 tile
        typename Config::ElemB const* B_smem,
        typename Config::ElemC*       C_reg,    // 寄存器累加器
        int k_tiles
    ) {
        std::cout << "  [device] 计算 " << k_tiles
                  << " 个 K-tile，大小 " << Config::kTileM
                  << "x" << Config::kTileN
                  << " 使用配置=" << Config::config_name() << "\n";
        // 在真实 CUTLASS 中: 内联 PTX 用于 mma.sync.aligned.m16n8k16...
    }

    // --- 6b. 全局函数: 实际的 CUDA kernel ---
    // 在真实 CUDA 中: __global__ void gemm_kernel(...)

    template <typename Config>
    void gemm_kernel_global(
        int M, int N, int K,
        typename Config::ElemA const* A, index_t lda,
        typename Config::ElemB const* B, index_t ldb,
        typename Config::ElemC*       C, index_t ldc,
        typename Config::Accum alpha,
        typename Config::Accum beta
    ) {
        std::cout << "[global] GemmKernel 启动:"
                  << " M=" << M << " N=" << N << " K=" << K
                  << " | 配置=" << Config::config_name()
                  << " | 线程数=" << Config::kThreads
                  << " | tile=" << Config::kTileM << "x"
                  << Config::kTileN << "x" << Config::kTileK
                  << " | 阶段数=" << Config::kStages << "\n";

        // --- 模拟共享内存 tile 加载 ---
        // 在真实 CUTLASS 中:
        //   __shared__ ElemA smem_A[kTileM * kTileK];
        //   __shared__ ElemB smem_B[kTileN * kTileK];
        //
        // 然后协作加载 global → shared memory，
        // 紧接着 __syncthreads()，然后 device_gemm_tile()。

        // 计算网格步长循环
        int total_tiles = (M / Config::kTileM) * (N / Config::kTileN);

        // 对此线程块处理的每个 tile
        for (int tile_idx = blockIdx_x(); tile_idx < total_tiles; tile_idx += gridDim_x()) {
            // 协作 global → shared memory 加载（模拟）
            // ...

            syncthreads();  // 真实 CUDA 中是 __syncthreads()

            // 计算 tile
            device_gemm_tile<Config>(
                nullptr,   // A_smem（占位符）
                nullptr,   // B_smem（占位符）
                nullptr,   // C_reg  （占位符）
                K / Config::kTileK
            );

            syncthreads();  // 真实 CUDA 中是 __syncthreads()

            // 将结果存回 global memory
            // ...
        }
    }

} // namespace kernel_impl

// ============================================================================
// 第 7 节: Kernel 成员函数定义
// ============================================================================

template <typename Config>
void GemmKernel<Config>::launch(
    int M, int N, int K,
    typename Config::ElemA const* A, index_t lda,
    typename Config::ElemB const* B, index_t ldb,
    typename Config::ElemC*       C, index_t ldc,
    typename Config::Accum alpha,
    typename Config::Accum beta
) {
    // 在真实 CUTLASS 中，这计算 grid/block 维度并
    // 使用适当的启动边界启动 kernel。
    //
    // dim3 grid(ceil_div(M, Config::kTileM), ceil_div(N, Config::kTileN));
    // dim3 block(Config::kThreads);
    //
    // gemm_kernel_global<Config><<<grid, block, smem_size, stream>>>(
    //     M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);

    std::cout << "[launch] " << Config::config_name()
              << " | 架构=" << Config::Arch::name()
              << " | 网格=("
              << (M + Config::kTileM - 1) / Config::kTileM << ", "
              << (N + Config::kTileN - 1) / Config::kTileN
              << ") | 块=" << Config::kThreads << "\n";

    // 模拟 kernel 执行
    kernel_impl::gemm_kernel_global<Config>(
        M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
}

template <typename Config>
void GemmKernel<Config>::describe() {
    std::cout << "[describe] "
              << "架构=" << Config::Arch::name()
              << " | Tile=" << Config::kTileM << "x"
              << Config::kTileN << "x" << Config::kTileK
              << " | 线程数=" << Config::kThreads
              << " | 阶段数=" << Config::kStages
              << " | 配置=" << Config::config_name() << "\n";
}

// ============================================================================
// 第 8 节: 显式模板实例化定义
// ============================================================================
//
// 在真实 CUTLASS 中，这些在单独的 .cu 文件中，
// 每个架构 + 数据类型组合一个。它们被编译一次
// 并生成与用户代码链接的目标文件。
//
// 这是构建系统生成的内容（通常通过 Python 脚本）。

// --- SM80 半精度 kernel ---
template class GemmKernel<KernelConfig<Sm80Tag, half_t, half_t, half_t>>;
template class GemmKernel<KernelConfig<Sm80Tag, half_t, half_t, float_t>>;

// --- SM90 半精度 kernel ---
template class GemmKernel<KernelConfig<Sm90Tag, half_t, half_t, half_t>>;
template class GemmKernel<KernelConfig<Sm90Tag, half_t, half_t, float_t>>;

// ============================================================================
// 第 9 节: 便捷类型别名（面向用户）
// ============================================================================
//
// 用户使用这些别名而不是冗长的模板类型。

/// SM80 (A100) 上的 FP16×FP16→FP16 GEMM
using GemmSm80Fp16Fp16Fp16 = GemmKernel<
    KernelConfig<Sm80Tag, half_t, half_t, half_t>>;

/// SM80 (A100) 上的 FP16×FP16→FP32 GEMM
using GemmSm80Fp16Fp16Fp32 = GemmKernel<
    KernelConfig<Sm80Tag, half_t, half_t, float_t>>;

/// SM90 (H100) 上的 FP16×FP16→FP16 GEMM
using GemmSm90Fp16Fp16Fp16 = GemmKernel<
    KernelConfig<Sm90Tag, half_t, half_t, half_t>>;

/// SM90 (H100) 上的 FP16×FP16→FP32 GEMM
using GemmSm90Fp16Fp16Fp32 = GemmKernel<
    KernelConfig<Sm90Tag, half_t, half_t, float_t>>;

// ============================================================================
// 第 10 节: 编译期验证
// ============================================================================

// 验证 SM80 配置使用正确的 tile 大小
using CfgSm80 = KernelConfig<Sm80Tag, half_t, half_t, half_t>;
static_assert(CfgSm80::kTileM == 256, "SM80 应使用 256 M-tile");
static_assert(CfgSm80::kStages == 4,  "SM80 应使用 4 个流水线阶段");

// 验证 SM90 配置使用 TMA 优化的大小
using CfgSm90 = KernelConfig<Sm90Tag, half_t, half_t, half_t>;
static_assert(CfgSm90::kTileN == 256, "SM90 应使用 256 N-tile");
static_assert(CfgSm90::kStages == 5,  "SM90 应使用 5 个流水线阶段");

// 验证默认（非 SM80/SM90）使用通用配置
using CfgGeneric = KernelConfig<struct UnknownArch, half_t, half_t, half_t>;
static_assert(CfgGeneric::config_name() == std::string_view("通用"));

// 验证架构标签是不同的类型
static_assert(!std::is_same_v<Sm80Tag, Sm90Tag>,
    "SM80 和 SM90 必须是不同的架构标签");

// ============================================================================
// MAIN: 模拟用户代码
// ============================================================================

int main() {
    std::cout << "=== CUTLASS Kernel 显式实例化模拟 ===\n\n";

    // --- 用户代码: 使用 extern 模板 ---
    // 编译器不会在此处生成这些特化；
    // 它使用来自第 8 节的显式实例化。

    half_t A_data[256 * 64] = {};
    half_t B_data[64 * 256] = {};
    half_t C_data[256 * 256] = {};

    std::cout << "--- SM80 FP16×FP16→FP16 ---\n";
    GemmSm80Fp16Fp16Fp16::describe();
    GemmSm80Fp16Fp16Fp16::launch(
        256, 256, 64,
        A_data, 64,
        B_data, 256,
        C_data, 256
    );

    std::cout << "\n--- SM80 FP16×FP16→FP32 ---\n";
    float_t C_fp32[256 * 256] = {};
    GemmSm80Fp16Fp16Fp32::describe();
    GemmSm80Fp16Fp16Fp32::launch(
        256, 256, 64,
        A_data, 64,
        B_data, 256,
        reinterpret_cast<GemmSm80Fp16Fp16Fp32::Config::ElemC*>(C_fp32),
        256
    );

    std::cout << "\n--- SM90 FP16×FP16→FP16 ---\n";
    GemmSm90Fp16Fp16Fp16::describe();
    GemmSm90Fp16Fp16Fp16::launch(
        512, 512, 128,
        A_data, 128,
        B_data, 512,
        C_data, 512
    );

    std::cout << "\n--- SM90 FP16×FP16→FP32 ---\n";
    float_t C_fp32_b[512 * 512] = {};
    GemmSm90Fp16Fp16Fp32::describe();
    GemmSm90Fp16Fp16Fp32::launch(
        512, 512, 128,
        A_data, 128,
        B_data, 512,
        reinterpret_cast<GemmSm90Fp16Fp16Fp32::Config::ElemC*>(C_fp32_b),
        512
    );

    std::cout << "\nKernel 显式实例化模拟完成。\n";
    std::cout << "构建模型: N 个带 extern 模板的 TU + 1 个带显式实例化的 TU\n";
    return 0;
}
