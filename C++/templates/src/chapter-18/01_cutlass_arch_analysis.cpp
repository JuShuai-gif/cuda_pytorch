// ============================================================================
// 01_cutlass_arch_analysis.cpp - CUTLASS 架构分析:
//                                 完整的 Mini CUTLASS 骨架
// ============================================================================
//
// 目的:
//   CUTLASS 模板设计模式的完整结构分析，
//   实现一个 mini-CUTLASS 骨架，演示
//   第 9-17 章涵盖的每个主要架构概念。
//
// CUTLASS 架构概览:
//
//   第 0 层: 基础类型
//     - 架构标签（ArchTag）
//     - 元素类型（float16、float32、int8 等）
//     - 内存布局描述符（RowMajor、ColumnMajor）
//
//   第 1 层: 线程级操作
//     - ThreadOp: 标量乘加、加载、存储
//     - Epilogue 操作: ReLU、GELU、bias 等
//
//   第 2 层: Warp 级操作
//     - WarpOp: 32 线程协作计算一个 tile
//     - MMA 指令: 通过 ArchTag 选择
//
//   第 3 层: Block 级操作
//     - BlockOp: 共享内存 tiling、异步拷贝
//     - Pipeline: 软件流水线阶段
//
//   第 4 层: Kernel/Device 级
//     - Kernel: 网格启动、block 调度
//     - Gemm: 公共 API、配置、分发
//
//   第 5 层: 构建系统集成
//     - 显式实例化: header + .cu 分离
//     - extern template: 抑制隐式实例化
//
// 概念模型:
//
//   用户代码
//     |
//     v
//   Gemm（公共 API） --配置--> GemmConfig
//     |                          |
//     v                          v
//   Kernel（device）  <--架构-->  ArchTag（Sm70/80/90）
//     |                            |
//     v                            v
//   BlockOp ----------------> TileConfig
//     |                            |
//     v                            v
//   WarpOp  <--mma--> MmaSelector <--架构-->
//     |
//     v
//   ThreadOp --元素--> ElementType
//     |
//     v
//   Epilogue --激活函数--> ActivationOp
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <cstdint>
#include <string>
#include <cmath>

// ============================================================================
// 第 0 层: 基础类型
// ============================================================================

// --- 0a. 元素类型（模拟 cutlass/numeric_types.h） ---
namespace cutlass_mini {

using float16_t = uint16_t;
using float32_t = float;
using float64_t = double;
using int8_t    = int8_t;
using int32_t   = int32_t;
using index_t   = int;

// 元素类型特征（编译期查询）
template <typename T>
struct NumericTraits;

template <> struct NumericTraits<float16_t> {
    static constexpr int kBits = 16;
    static constexpr bool kIsFloating = true;
    static constexpr bool kIsComplex = false;
    static constexpr const char* kName = "fp16";
};

template <> struct NumericTraits<float32_t> {
    static constexpr int kBits = 32;
    static constexpr bool kIsFloating = true;
    static constexpr bool kIsComplex = false;
    static constexpr const char* kName = "fp32";
};

template <> struct NumericTraits<float64_t> {
    static constexpr int kBits = 64;
    static constexpr bool kIsFloating = true;
    static constexpr bool kIsComplex = false;
    static constexpr const char* kName = "fp64";
};

template <> struct NumericTraits<int8_t> {
    static constexpr int kBits = 8;
    static constexpr bool kIsFloating = false;
    static constexpr bool kIsComplex = false;
    static constexpr const char* kName = "int8";
};

template <> struct NumericTraits<int32_t> {
    static constexpr int kBits = 32;
    static constexpr bool kIsFloating = false;
    static constexpr bool kIsComplex = false;
    static constexpr const char* kName = "int32";
};

// --- 0b. 内存布局描述符（模拟 cutlass/layout/matrix.h） ---
struct RowMajor {
    static constexpr index_t kRank = 2;
    static constexpr index_t offset(index_t row, index_t col, index_t ldm) {
        return row * ldm + col;
    }
    static constexpr const char* kName = "行优先";
};

struct ColumnMajor {
    static constexpr index_t kRank = 2;
    static constexpr index_t offset(index_t row, index_t col, index_t ldm) {
        return col * ldm + row;
    }
    static constexpr const char* kName = "列优先";
};

// --- 0c. 架构标签（模拟 cutlass/arch/arch.h） ---
namespace arch {

struct Sm70 {
    static constexpr int kComputeCapability = 70;
    static constexpr int kSharedMemKB = 96;
    static constexpr int kMaxThreadsPerBlock = 1024;
    static constexpr bool kHasTensorCore = true;
    static constexpr bool kHasAsyncCopy = false;
    static constexpr bool kHasTMA = false;
    static constexpr const char* kName = "Volta (SM70)";
};

struct Sm80 {
    static constexpr int kComputeCapability = 80;
    static constexpr int kSharedMemKB = 163;
    static constexpr int kMaxThreadsPerBlock = 1024;
    static constexpr bool kHasTensorCore = true;
    static constexpr bool kHasAsyncCopy = true;
    static constexpr bool kHasTMA = false;
    static constexpr const char* kName = "Ampere (SM80)";
};

struct Sm90 {
    static constexpr int kComputeCapability = 90;
    static constexpr int kSharedMemKB = 227;
    static constexpr int kMaxThreadsPerBlock = 1024;
    static constexpr bool kHasTensorCore = true;
    static constexpr bool kHasAsyncCopy = true;
    static constexpr bool kHasTMA = true;
    static constexpr const char* kName = "Hopper (SM90)";
};

} // namespace arch

} // namespace cutlass_mini

// ============================================================================
// 第 1 层: 线程级操作
// ============================================================================
//
// 在真实 CUTLASS 中的对应:
//   cutlass/thread/mma.h
//   cutlass/epilogue/thread/linear_combination.h

namespace cutlass_mini {

/// \brief 线程级乘加操作符。
/// 计算: D = alpha * A * B + beta * C 用于标量元素。
template <typename ElementA_,
          typename ElementB_,
          typename ElementC_,
          typename Policy_ = void>
class ThreadMma {
public:
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    /// \brief 乘加: C += A * B
    /// 在真实 CUTLASS 中，这使用硬件内联函数（mma、fma）。
    static ElementC mma(ElementA a, ElementB b, ElementC accum) {
        return accum + static_cast<ElementC>(a) * static_cast<ElementC>(b);
    }

    /// \brief 从全局内存加载元素（模拟）。
    static ElementA load(ElementA const* ptr, index_t offset) {
        return ptr[offset];
    }

    /// \brief 将元素存储到全局内存（模拟）。
    static void store(ElementC* ptr, index_t offset, ElementC val) {
        ptr[offset] = val;
    }
};

/// \brief Epilogue: 带激活函数的线性组合。
/// 计算: output = Activation(alpha * accum + beta * source)
template <typename ElementOutput_,
          typename ElementAccumulator_,
          template <typename> class ActivationOp>
class LinearCombinationEpilogue {
public:
    using ElementOutput      = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;

    /// \brief 应用 epilogue 变换。
    ElementOutput operator()(
        ElementAccumulator accum,
        ElementOutput      source,
        ElementAccumulator alpha = 1,
        ElementAccumulator beta  = 0
    ) const {
        ElementAccumulator val = alpha * accum + beta * static_cast<ElementAccumulator>(source);
        return ActivationOp<ElementOutput>::apply(static_cast<ElementOutput>(val));
    }

    static constexpr const char* kName = "线性组合";
};

/// \brief Identity 激活函数（直通）。
template <typename T>
struct IdentityActivation {
    static T apply(T val) { return val; }
    static constexpr const char* kName = "Identity";
};

/// \brief ReLU 激活函数。
template <typename T>
struct ReluActivation {
    static T apply(T val) { return (val > T{0}) ? val : T{0}; }
    static constexpr const char* kName = "ReLU";
};

} // namespace cutlass_mini

// ============================================================================
// 第 2 层: Warp 级操作
// ============================================================================
//
// 在真实 CUTLASS 中的对应:
//   cutlass/gemm/warp/mma_tensor_op.h

namespace cutlass_mini {

/// \brief Warp 级 tile 配置。
template <int kWarpM_, int kWarpN_, int kWarpK_>
struct WarpTileConfig {
    static constexpr int kWarpM = kWarpM_;
    static constexpr int kWarpN = kWarpN_;
    static constexpr int kWarpK = kWarpK_;
    static constexpr int kWarpSize = 32;
    static constexpr int kTileElements = kWarpM * kWarpN;
};

/// \brief Warp 级 MMA 操作符。
/// 协调 32 线程计算一个 C += A * B 的 tile。
template <typename ThreadMma_, typename WarpConfig_>
class WarpMma {
public:
    using ThreadMma  = ThreadMma_;
    using WarpConfig = WarpConfig_;
    using ElementA   = typename ThreadMma::ElementA;
    using ElementB   = typename ThreadMma::ElementB;
    using ElementC   = typename ThreadMma::ElementC;

    static constexpr int kM = WarpConfig::kWarpM;
    static constexpr int kN = WarpConfig::kWarpN;
    static constexpr int kK = WarpConfig::kWarpK;

    /// \brief 计算一次 warp 级 tile 的迭代。
    void operator()(
        ElementC*      accum_frag,
        ElementA const* frag_A,
        ElementB const* frag_B,
        index_t         lane_id
    ) {
        for (int k = 0; k < kK; ++k) {
            for (int m = 0; m < kM; ++m) {
                for (int n = 0; n < kN; ++n) {
                    if ((m * kN + n) % WarpConfig::kWarpSize == lane_id) {
                        ElementA a = frag_A[m * kK + k];
                        ElementB b = frag_B[k * kN + n];
                        accum_frag[m * kN + n] = ThreadMma::mma(a, b, accum_frag[m * kN + n]);
                    }
                }
            }
        }
    }
};

} // namespace cutlass_mini

// ============================================================================
// 第 3 层: Block 级操作
// ============================================================================
//
// 在真实 CUTLASS 中的对应:
//   cutlass/gemm/threadblock/mma_pipelined.h

namespace cutlass_mini {

/// \brief Block 级 tile 配置。
template <int kThreadblockM_, int kThreadblockN_, int kThreadblockK_,
          int kWarpM_, int kWarpN_, int kWarpK_,
          int kStages_>
struct BlockTileConfig {
    static constexpr int kThreadblockM = kThreadblockM_;
    static constexpr int kThreadblockN = kThreadblockN_;
    static constexpr int kThreadblockK = kThreadblockK_;
    static constexpr int kWarpM = kWarpM_;
    static constexpr int kWarpN = kWarpN_;
    static constexpr int kWarpK = kWarpK_;
    static constexpr int kStages = kStages_;
    static constexpr int kWarpCount =
        (kThreadblockM / kWarpM) * (kThreadblockN / kWarpN);
};

/// \brief Block 级 GEMM 操作符。
/// 管理共享内存 tiling 和 warp 协调。
template <typename WarpMma_, typename BlockConfig_>
class BlockMma {
public:
    using WarpMma     = WarpMma_;
    using BlockConfig = BlockConfig_;
    using ThreadMma   = typename WarpMma::ThreadMma;
    using ElementA    = typename WarpMma::ElementA;
    using ElementB    = typename WarpMma::ElementB;
    using ElementC    = typename WarpMma::ElementC;

    static constexpr int kM = BlockConfig::kThreadblockM;
    static constexpr int kN = BlockConfig::kThreadblockN;
    static constexpr int kK = BlockConfig::kThreadblockK;

    /// \brief 运行 block 级 GEMM 计算。
    void operator()(
        ElementC*      c_frag,
        ElementA const* a_smem,
        ElementB const* b_smem
    ) {
        std::cout << "  [BlockMma] Tile " << kM << "x" << kN << "x" << kK
                  << " | 阶段数=" << BlockConfig::kStages
                  << " | Warp数=" << BlockConfig::kWarpCount << "\n";

        WarpMma warp_mma;
        // 模拟: 每个 warp 计算其子 tile
        for (int w = 0; w < BlockConfig::kWarpCount; ++w) {
            warp_mma(c_frag, a_smem, b_smem, w % 32);
        }
    }
};

} // namespace cutlass_mini

// ============================================================================
// 第 4 层: Kernel/Device 级
// ============================================================================
//
// 在真实 CUTLASS 中的对应:
//   cutlass/gemm/kernel/gemm.h
//   cutlass/gemm/device/gemm.h

namespace cutlass_mini {
namespace kernel {

/// \brief GEMM kernel: 用户实例化的顶层模板。
/// 这是 CUTLASS 的 "kernel" — 一个类模板，而非 CUDA __global__
/// 函数。它管理网格步长循环、共享内存初始化
/// 和 block 级计算。
///
/// 模板参数分解:
///   BlockMma_     - Block 级操作符（Block+Warp+Thread 操作的组合）
///   Epilogue_     - Epilogue 操作符（最终逐元素变换）
///   ArchTag_      - 架构标识符（用于编译期 tile 选择）
template <typename BlockMma_, typename Epilogue_, typename ArchTag_>
class GemmKernel {
public:
    using BlockMma = BlockMma_;
    using Epilogue = Epilogue_;
    using ArchTag  = ArchTag_;

    using ElementA = typename BlockMma::ElementA;
    using ElementB = typename BlockMma::ElementB;
    using ElementC = typename Epilogue::ElementOutput;
    using AccumT   = typename Epilogue::ElementAccumulator;

    /// \brief 启动参数（网格/block 维度）。
    struct LaunchParams {
        int grid_m;
        int grid_n;
        int block_dim;
    };

    /// \brief 从问题维度计算启动参数。
    static LaunchParams compute_launch_params(int M, int N) {
        return {
            (M + BlockMma::kM - 1) / BlockMma::kM,
            (N + BlockMma::kN - 1) / BlockMma::kN,
            256  // 典型 block 大小
        };
    }

    /// \brief 运行 kernel（模拟单线程执行）。
    static void run(
        int M, int N, int K,
        ElementA const* A, index_t lda,
        ElementB const* B, index_t ldb,
        ElementC*       C, index_t ldc,
        AccumT alpha = 1, AccumT beta = 0
    ) {
        LaunchParams lp = compute_launch_params(M, N);

        std::cout << "[GemmKernel] 架构=" << ArchTag::kName
                  << " | 网格=" << lp.grid_m << "x" << lp.grid_n
                  << " | 线程块=" << lp.block_dim
                  << " | 问题规模=" << M << "x" << N << "x" << K << "\n";

        Epilogue epilogue;
        BlockMma block_mma;

        // 模拟网格步长循环
        for (int grid_m = 0; grid_m < lp.grid_m; ++grid_m) {
            for (int grid_n = 0; grid_n < lp.grid_n; ++grid_n) {
                // 在真实 CUTLASS 中: 协作加载 A/B tile 到共享内存
                // __syncthreads()
                // block_mma(c_frag, a_smem, b_smem)
                // __syncthreads()
                // epilogue: 将结果写入 C

                // 对寄存器片段使用累加器类型（而非输出类型）
                AccumT c_frag[BlockMma::kM * BlockMma::kN] = {};
                block_mma(c_frag, nullptr, nullptr);

                int c_row = grid_m * BlockMma::kM;
                int c_col = grid_n * BlockMma::kN;
                for (int m = 0; m < BlockMma::kM && (c_row + m) < M; ++m) {
                    for (int n = 0; n < BlockMma::kN && (c_col + n) < N; ++n) {
                        C[(c_row + m) * ldc + (c_col + n)] = epilogue(
                            c_frag[m * BlockMma::kN + n], 0, alpha, beta);
                    }
                }
            }
        }
    }
};

} // namespace kernel
} // namespace cutlass_mini

// ============================================================================
// 第 5 层: 公共 API 与配置组装
// ============================================================================
//
// 在真实 CUTLASS 中的对应:
//   cutlass/gemm/device/gemm.h（Gemm 模板）
//   tools/library/scripts/gemm_operations.py（生成配置）

namespace cutlass_mini {
namespace gemm {

/// \brief GEMM 配置: 组装所有模板参数。
/// 这是用户（或代码生成器）填充以选择特定
/// GEMM 实现的单一配置结构体。
///
/// 在真实 CUTLASS 中，这些配置由 Python 脚本生成，
/// 它们枚举所有有效的（Arch、Element、Tile）组合。
template <typename ArchTag_,
          typename ElementA_,
          typename ElementB_,
          typename ElementC_,
          typename Accumulator_ = ElementC_>
struct GemmConfig {
    using ArchTag     = ArchTag_;
    using ElementA    = ElementA_;
    using ElementB    = ElementB_;
    using ElementC    = ElementC_;
    using Accumulator = Accumulator_;
    // 注意: ActivationOp_ 是 TTP — 它不能作为成员别名存储
    // 在 C++ 中（成员别名模板不被允许）。相反，TTP
    // 直接传递给 kernel 组装器（见 AssembleGemmKernel）。

    // Tile 大小在编译期基于 ArchTag 选择
    // 通过偏特化（见第 16 章）
    static constexpr int kThreadblockM = [] {
        if constexpr (ArchTag::kComputeCapability >= 90) return 256;
        else if constexpr (ArchTag::kComputeCapability >= 80) return 256;
        else return 128;
    }();

    static constexpr int kThreadblockN = [] {
        if constexpr (ArchTag::kComputeCapability >= 90) return 256;
        else if constexpr (ArchTag::kComputeCapability >= 80) return 128;
        else return 128;
    }();

    static constexpr int kThreadblockK = [] {
        if constexpr (ArchTag::kComputeCapability >= 90) return 64;
        else if constexpr (ArchTag::kComputeCapability >= 80) return 32;
        else return 16;
    }();

    static constexpr int kStages = [] {
        if constexpr (ArchTag::kComputeCapability >= 90) return 5;
        else if constexpr (ArchTag::kComputeCapability >= 80) return 4;
        else return 2;
    }();
};

/// \brief 从 GemmConfig 组装完整的 kernel 类型。
/// 这是 "类型级组装器"，从配置片段构建完整的 GEMM kernel。
template <typename Config,
          template <typename> class ActivationOp = IdentityActivation>
using AssembleGemmKernel = kernel::GemmKernel<
    // 第 3 层: Block 级操作符
    BlockMma<
        // 第 2 层: Warp 级操作符
        WarpMma<
            // 第 1 层: 线程级操作符
            ThreadMma<
                typename Config::ElementA,
                typename Config::ElementB,
                typename Config::Accumulator
            >,
            // Warp tile 配置
            WarpTileConfig<64, 64, 32>
        >,
        // Block tile 配置
        BlockTileConfig<
            Config::kThreadblockM,
            Config::kThreadblockN,
            Config::kThreadblockK,
            64, 64, 32,
            Config::kStages
        >
    >,
    // 第 1 层（补充）: Epilogue 操作符
    LinearCombinationEpilogue<
        typename Config::ElementC,
        typename Config::Accumulator,
        ActivationOp
    >,
    // 第 0 层: 架构标签
    typename Config::ArchTag
>;

/// \brief 公共 GEMM API。
/// 这是用户与之交互的接口。它管理完整的生命周期:
/// 配置 → kernel 组装 → 启动。
///
/// ActivationOp TTP 在这里直接传递（CUTLASS 风格
/// 将 epilogue 与核心 GEMM 配置分离）。
template <typename Config,
          template <typename> class ActivationOp = IdentityActivation>
class Gemm {
public:
    using Kernel = AssembleGemmKernel<Config, ActivationOp>;
    using ElementA = typename Config::ElementA;
    using ElementB = typename Config::ElementB;
    using ElementC = typename Config::ElementC;

    /// \brief 执行 GEMM 操作。
    void operator()(
        int M, int N, int K,
        ElementA const* A, index_t lda,
        ElementB const* B, index_t ldb,
        ElementC*       C, index_t ldc,
        typename Config::Accumulator alpha = 1,
        typename Config::Accumulator beta  = 0
    ) {
        Kernel::run(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    }
};

} // namespace gemm
} // namespace cutlass_mini

// ============================================================================
// 节: 预定义配置（生成/策划的目录）
// ============================================================================
//
// 在真实 CUTLASS 中，这些由 Python 脚本从
// tools/library/scripts/gemm_operations.py 生成

using namespace cutlass_mini;
using namespace cutlass_mini::arch;

// SM80: FP16×FP16→FP16 带 ReLU
using GemmFp16Sm80Relu = gemm::Gemm<
    gemm::GemmConfig<Sm80, float16_t, float16_t, float16_t, float32_t>,
    ReluActivation
>;

// SM80: FP32×FP32→FP32（纯线性） — 默认 Identity
using GemmFp32Sm80 = gemm::Gemm<
    gemm::GemmConfig<Sm80, float32_t, float32_t, float32_t>
>;

// SM90: FP16×FP16→FP32（混合精度） — 默认 Identity
using GemmFp16Sm90Fp32Accum = gemm::Gemm<
    gemm::GemmConfig<Sm90, float16_t, float16_t, float32_t, float32_t>
>;

// ============================================================================
// 节: 编译期验证
// ============================================================================

// 验证元素特征
static_assert(NumericTraits<float16_t>::kBits == 16);
static_assert(NumericTraits<float32_t>::kIsFloating);

// 验证内存布局
static_assert(RowMajor::offset(2, 3, 10) == 23);  // 2*10 + 3

// 验证架构标签是不同的类型
static_assert(!std::is_same_v<Sm70, Sm80>);
static_assert(!std::is_same_v<Sm80, Sm90>);

// 验证 tile 大小随架构而增大
using Cfg70 = gemm::GemmConfig<Sm70, float32_t, float32_t, float32_t>;
using Cfg80 = gemm::GemmConfig<Sm80, float32_t, float32_t, float32_t>;
using Cfg90 = gemm::GemmConfig<Sm90, float32_t, float32_t, float32_t>;

static_assert(Cfg80::kThreadblockM >= Cfg70::kThreadblockM);
static_assert(Cfg90::kStages >= Cfg80::kStages);
static_assert(Cfg90::kThreadblockK >= Cfg80::kThreadblockK);

// 验证 kernel 组装产生有效类型（默认 activation = Identity）
using Kernel80 = gemm::AssembleGemmKernel<Cfg80>;
static_assert(sizeof(Kernel80) > 0, "Kernel 类型必须完整");

// ============================================================================
// MAIN: 端到端演示
// ============================================================================

int main() {
    std::cout << "=== Mini CUTLASS 架构分析 ===\n";
    std::cout << "=== （整合第 9-17 章概念） ===\n\n";

    // --- SM80 FP16 GEMM 带 ReLU ---
    std::cout << "--- SM80 FP16 GEMM 带 ReLU ---\n";
    float16_t A_fp16[256 * 64] = {};
    float16_t B_fp16[64 * 256] = {};
    float16_t C_fp16[256 * 256] = {};

    // 填充测试数据
    for (int i = 0; i < 256 * 64; ++i) A_fp16[i] = static_cast<float16_t>(i % 10);
    for (int i = 0; i < 64 * 256; ++i) B_fp16[i] = static_cast<float16_t>(i % 10);

    GemmFp16Sm80Relu gemm_fp16;
    gemm_fp16(256, 256, 64, A_fp16, 64, B_fp16, 256, C_fp16, 256);

    // --- SM80 FP32 GEMM ---
    std::cout << "\n--- SM80 FP32 GEMM ---\n";
    float32_t A_fp32[128 * 32] = {};
    float32_t B_fp32[32 * 128] = {};
    float32_t C_fp32[128 * 128] = {};

    for (int i = 0; i < 128; ++i)
        for (int j = 0; j < 32; ++j)
            A_fp32[i * 32 + j] = static_cast<float32_t>(i + j);

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 128; ++j)
            B_fp32[i * 128 + j] = static_cast<float32_t>(i + j);

    GemmFp32Sm80 gemm_fp32;
    gemm_fp32(128, 128, 32, A_fp32, 32, B_fp32, 128, C_fp32, 128);

    // --- SM90 FP16→FP32 混合精度 ---
    std::cout << "\n--- SM90 FP16→FP32 混合精度 ---\n";
    float32_t C_fp32_b[256 * 256] = {};
    GemmFp16Sm90Fp32Accum gemm_mixed;
    gemm_mixed(256, 256, 64, A_fp16, 64, B_fp16, 256,
               reinterpret_cast<GemmFp16Sm90Fp32Accum::ElementC*>(C_fp32_b), 256);

    // --- 架构摘要 ---
    std::cout << "\n=== 架构摘要 ===\n";
    std::cout << "第 0 层: ArchTags（Sm70/Sm80/Sm90）\n";
    std::cout << "  ├── NumericTraits<T>: 编译期类型查询\n";
    std::cout << "  ├── RowMajor / ColumnMajor: 内存布局\n";
    std::cout << "  └── 第 16 章: ArchTag 特化\n\n";

    std::cout << "第 1 层: 线程级操作\n";
    std::cout << "  ├── ThreadMma: 标量乘加\n";
    std::cout << "  ├── LinearCombinationEpilogue: 激活函数 + 缩放\n";
    std::cout << "  ├── 第 11 章: 线程级操作符设计\n";
    std::cout << "  └── 第 12 章: 用于激活函数的模板模板参数\n\n";

    std::cout << "第 2 层: Warp 级操作\n";
    std::cout << "  ├── WarpMma: 协作 tile 计算\n";
    std::cout << "  └── 第 11 章: Warp/Block 组合\n\n";

    std::cout << "第 3 层: Block 级操作\n";
    std::cout << "  ├── BlockMma: 共享内存 + 软件流水线\n";
    std::cout << "  └── 第 9 章: 通过模板参数进行配置\n\n";

    std::cout << "第 4 层: Kernel/Device 级\n";
    std::cout << "  ├── GemmKernel: 网格步长循环、启动参数\n";
    std::cout << "  ├── 第 14 章: Kernel 显式实例化\n";
    std::cout << "  └── 第 15 章: SFINAE 分发\n\n";

    std::cout << "第 5 层: 公共 API\n";
    std::cout << "  ├── GemmConfig: 类型级配置\n";
    std::cout << "  ├── AssembleGemmKernel: 类型级 kernel 组装\n";
    std::cout << "  ├── Gemm<T>: 面向用户的 API\n";
    std::cout << "  ├── 第 9 章:  extern template\n";
    std::cout << "  ├── 第 13 章: 命名空间 + ADL\n";
    std::cout << "  └── 第 17 章: 基于 Concept 的约束\n\n";

    std::cout << "Mini CUTLASS 架构分析完成。\n";
    return 0;
}
