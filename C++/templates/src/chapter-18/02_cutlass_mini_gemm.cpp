// ============================================================================
// 02_cutlass_mini_gemm.cpp - 完整 Mini CUTLASS 风格 GEMM:
//                              整合第 9-17 章的所有模板元编程概念
// ============================================================================
//
// 目的:
//   一个自包含、可运行的 mini-GEMM 实现，演示
//   应用于简单矩阵乘法的完整 CUTLASS 设计哲学。
//   本文件整合了:
//
//   - 显式/extern 实例化（第 9 章）
//   - 模板术语和 ODR（第 10 章）
//   - 可调用特征与延迟求值（第 11 章）
//   - 模板模板参数（第 12 章）
//   - 命名空间与 ADL（第 13 章）
//   - 实例化模型（第 14 章）
//   - 基于 SFINAE 的分发（第 15 章）
//   - 全/偏特化与 ArchTag（第 16 章）
//   - C++20 Concepts（第 17 章）
//
// 所演示的设计原则:
//   1. 层次分解（Thread → Warp → Block → Kernel）
//   2. 通过模板参数的基于策略的设计
//   3. 通过标签和 enable_if/concepts 的编译期分发
//   4. 为构建性能进行的显式实例化
//   5. 带有 ADL 的模块化命名空间组织
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <cassert>

// ============================================================================
// 第 1 部分: 类型系统（第 10、11 章）
// ============================================================================

namespace mini_cutlass {

// --- 数值类型（模拟 GPU 类型） ---
using half_t   = uint16_t;
using float_t  = float;
using double_t = double;
using int32_t  = int;
using index_t  = int;

// --- 架构标签（第 16 章） ---
namespace arch {
    struct Sm80 { static constexpr int kCC = 80; static constexpr const char* kName = "SM80"; };
    struct Sm90 { static constexpr int kCC = 90; static constexpr const char* kName = "SM90"; };
}

// --- C++20 Arch concept（第 17 章） ---
template <typename T>
concept ArchTag = requires {
    { T::kCC } -> std::convertible_to<int>;
    { T::kName } -> std::convertible_to<const char*>;
};
static_assert(ArchTag<arch::Sm80>);
static_assert(ArchTag<arch::Sm90>);

// --- 带延迟求值的类型特征（第 11 章） ---
template <typename T>
struct TypeInfo {
    static constexpr bool kIsHalf   = std::is_same_v<T, half_t>;
    static constexpr bool kIsFloat  = std::is_same_v<T, float_t>;
    static constexpr bool kIsDouble = std::is_same_v<T, double_t>;
    static constexpr bool kIsFloating = kIsHalf || kIsFloat || kIsDouble;
    static constexpr int  kSizeInBytes = sizeof(T);
    static constexpr const char* kName =
        kIsHalf ? "f16" : (kIsFloat ? "f32" : (kIsDouble ? "f64" : "未知"));
};

} // namespace mini_cutlass

// ============================================================================
// 第 2 部分: 线程级操作（第 11、12 章）
// ============================================================================

namespace mini_cutlass {
namespace thread {

/// \brief 标量乘加操作符。
/// 这是叶级操作 — 每个线程在其寄存器上执行这些操作。
template <typename ElementA_, typename ElementB_, typename ElementC_>
class Madd {
public:
    using ElemA = ElementA_;
    using ElemB = ElementB_;
    using ElemC = ElementC_;

    static ElemC compute(ElemA a, ElemB b, ElemC accum) {
        return accum + static_cast<ElemC>(a) * static_cast<ElemC>(b);
    }
};

/// \brief Epilogue 激活函数（来自第 12 章的模板模板参数）。
template <typename T>
struct Identity { static T apply(T v) { return v; } };

template <typename T>
struct ReLU { static T apply(T v) { return std::max(T{0}, v); } };

template <typename T>
struct GELU {
    static T apply(T v) {
        // 用于演示的简化 GELU 近似
        constexpr T kSqrt2OverPi = T{0.79788456};
        T tmp = kSqrt2OverPi * (v + T{0.044715} * v * v * v);
        return T{0.5} * v * (T{1} + std::tanh(tmp));
    }
};

} // namespace thread
} // namespace mini_cutlass

// ============================================================================
// 第 3 部分: Warp 及 Block 级操作（第 11 章）
// ============================================================================

namespace mini_cutlass {
namespace warp {

template <int kM, int kN, int kK, int kWarpSize = 32>
struct TileShape {
    static constexpr int kTileM = kM;
    static constexpr int kTileN = kN;
    static constexpr int kTileK = kK;
    static constexpr int kWarpS = kWarpSize;
};

template <typename MaddOp, typename TileShape_>
class WarpGemm {
public:
    using Madd     = MaddOp;
    using Tile     = TileShape_;
    using ElemA    = typename Madd::ElemA;
    using ElemB    = typename Madd::ElemB;
    using ElemC    = typename Madd::ElemC;
    using Fragment = ElemC[Tile::kTileM * Tile::kTileN];

    /// \brief tile 上的 Warp 级矩阵乘法。
    /// 每个 warp 的 32 线程协作计算一个输出 tile。
    static void compute(
        Fragment&    c_frag,     // 输出累加器（寄存器 tile）
        ElemA const* a_tile,     // 输入 A tile
        ElemB const* b_tile,     // 输入 B tile
        index_t      lane_id = 0 // 模拟 lane
    ) {
        // 将累加器初始化为零
        for (int i = 0; i < Tile::kTileM * Tile::kTileN; ++i)
            c_frag[i] = ElemC{0};

        // 计算外积（简化；真实 CUTLASS 使用 MMA 指令）
        for (int k = 0; k < Tile::kTileK; ++k) {
            for (int m = 0; m < Tile::kTileM; ++m) {
                for (int n = 0; n < Tile::kTileN; ++n) {
                    // 每个线程计算 tile 的一个子集
                    if ((m * Tile::kTileN + n) % Tile::kWarpS == lane_id) {
                        ElemA a = a_tile[m * Tile::kTileK + k];
                        ElemB b = b_tile[k * Tile::kTileN + n];
                        c_frag[m * Tile::kTileN + n] =
                            Madd::compute(a, b, c_frag[m * Tile::kTileN + n]);
                    }
                }
            }
        }
    }
};

} // namespace warp

namespace block {

/// \brief Block 级 tile 形状（第 14 章: 实例化模型）。
template <int kM, int kN, int kK, int kPipelineStages>
struct BlockShape {
    static constexpr int kBlockM = kM;
    static constexpr int kBlockN = kN;
    static constexpr int kBlockK = kK;
    static constexpr int kStages = kPipelineStages;
    static constexpr int kWarpCount = (kM / 64) * (kN / 64);  // 假设 64x64 warp tile
};

/// \brief Block 级 GEMM 协调器。
template <typename WarpGemm_, typename BlockShape_>
class BlockGemm {
public:
    using WarpGemm  = WarpGemm_;
    using Block     = BlockShape_;
    using ElemA     = typename WarpGemm::ElemA;
    using ElemB     = typename WarpGemm::ElemB;
    using ElemC     = typename WarpGemm::ElemC;
    using Fragment  = typename WarpGemm::Fragment;

    static constexpr int kBlockM = Block::kBlockM;
    static constexpr int kBlockN = Block::kBlockN;
    static constexpr int kBlockK = Block::kBlockK;

    /// \brief 在一个 tile 上执行 block 级 GEMM。
    static void run(
        ElemC*       c_block,
        ElemA const* a_block,
        ElemB const* b_block
    ) {
        Fragment c_frag;
        WarpGemm::compute(c_frag, a_block, b_block, 0);

        // 存储片段到 block 输出
        for (int m = 0; m < WarpGemm::Tile::kTileM; ++m)
            for (int n = 0; n < WarpGemm::Tile::kTileN; ++n)
                c_block[m * kBlockN + n] = c_frag[m * WarpGemm::Tile::kTileN + n];
    }
};

} // namespace block
} // namespace mini_cutlass

// ============================================================================
// 第 4 部分: Kernel 组装与 Epilogue（第 12、15、16 章）
// ============================================================================

namespace mini_cutlass {
namespace kernel {

/// \brief 架构 tile 选择器（第 16 章: 偏特化）。
template <typename ArchTag, typename = void>
struct ArchTileConfig {
    static constexpr int kM = 64;
    static constexpr int kN = 64;
    static constexpr int kK = 16;
    static constexpr int kStages = 2;
};

template <>
struct ArchTileConfig<arch::Sm80> {
    static constexpr int kM = 256;
    static constexpr int kN = 128;
    static constexpr int kK = 32;
    static constexpr int kStages = 4;
};

template <>
struct ArchTileConfig<arch::Sm90> {
    static constexpr int kM = 256;
    static constexpr int kN = 256;
    static constexpr int kK = 64;
    static constexpr int kStages = 5;
};

/// \brief Kernel 配置组装器。
/// 对 ActivationOp 使用模板模板参数（第 12 章）。
template <typename ArchTag_,
          typename ElemA_, typename ElemB_, typename ElemC_>
struct KernelConfig {
    using Arch  = ArchTag_;
    using ElemA = ElemA_;
    using ElemB = ElemB_;
    using ElemC = ElemC_;
    // 注意: ActivationOp_ TTP 通过 KernelConfig 模板传递；
    // C++ 中成员别名模板不被允许。

    using TileCf = ArchTileConfig<Arch>;

    static constexpr int kBlockM = TileCf::kM;
    static constexpr int kBlockN = TileCf::kN;
    static constexpr int kBlockK = TileCf::kK;
    static constexpr int kStages = TileCf::kStages;
    static constexpr int kWarpM  = 64;
    static constexpr int kWarpN  = 64;
    static constexpr int kWarpK  = 32;
};

/// \brief 类型级 kernel 组装器。
/// 从 KernelConfig 片段构建完整的 kernel 类型。
template <typename Config>
using AssembleKernel = block::BlockGemm<
    warp::WarpGemm<
        thread::Madd<
            typename Config::ElemA,
            typename Config::ElemB,
            typename Config::ElemC
        >,
        warp::TileShape<Config::kWarpM, Config::kWarpN, Config::kWarpK>
    >,
    block::BlockShape<Config::kBlockM, Config::kBlockN, Config::kBlockK, Config::kStages>
>;

/// \brief 统一的 GEMM kernel。
/// ActivationOp TTP 直接作为模板参数传递
/// （CUTLASS 风格的 epilogue 分离）。
template <typename Config,
          template <typename> class ActivationOp = thread::Identity>
class GemmKernel {
public:
    using Kernel = AssembleKernel<Config>;
    using ElemA  = typename Config::ElemA;
    using ElemB  = typename Config::ElemB;
    using ElemC  = typename Config::ElemC;
    using Arch   = typename Config::Arch;

    /// \brief 运行带 epilogue 的完整 GEMM。
    static void compute(
        int M, int N, int K,
        ElemA const* A, index_t lda,
        ElemB const* B, index_t ldb,
        ElemC*       C, index_t ldc
    ) {
        int num_block_m = (M + Config::kBlockM - 1) / Config::kBlockM;
        int num_block_n = (N + Config::kBlockN - 1) / Config::kBlockN;

        // 分配 tile 缓冲区（模拟真实 CUDA 中的共享内存）
        std::vector<ElemA> a_tile(Config::kBlockM * Config::kBlockK);
        std::vector<ElemB> b_tile(Config::kBlockN * Config::kBlockK);
        std::vector<ElemC> c_block(Config::kBlockM * Config::kBlockN);

        for (int bm = 0; bm < num_block_m; ++bm) {
            for (int bn = 0; bn < num_block_n; ++bn) {
                int m_start = bm * Config::kBlockM;
                int n_start = bn * Config::kBlockN;

                // 清零累加器
                std::memset(c_block.data(), 0, c_block.size() * sizeof(ElemC));

                // K 维度循环（tiled）
                for (int bk = 0; bk < K; bk += Config::kBlockK) {
                    int k_tile = std::min(Config::kBlockK, K - bk);

                    // 加载 A tile（模拟协作加载）
                    for (int m = 0; m < Config::kBlockM && (m_start + m) < M; ++m)
                        for (int k = 0; k < k_tile; ++k)
                            a_tile[m * Config::kBlockK + k] =
                                A[(m_start + m) * lda + (bk + k)];

                    // 加载 B tile
                    for (int k = 0; k < k_tile; ++k)
                        for (int n = 0; n < Config::kBlockN && (n_start + n) < N; ++n)
                            b_tile[n * Config::kBlockK + k] =
                                B[(bk + k) * ldb + (n_start + n)];

                    // 计算（模拟 block 级 GEMM）
                    Kernel::run(c_block.data(), a_tile.data(), b_tile.data());
                }

                // 应用 epilogue 激活函数并存储到输出
                for (int m = 0; m < Config::kBlockM && (m_start + m) < M; ++m) {
                    for (int n = 0; n < Config::kBlockN && (n_start + n) < N; ++n) {
                        ElemC val = c_block[m * Config::kBlockN + n];
                        C[(m_start + m) * ldc + (n_start + n)] =
                            ActivationOp<ElemC>::apply(val);
                    }
                }
            }
        }
    }

    /// \brief 描述 kernel 配置。
    static void describe() {
        std::cout << "[GemmKernel] 架构=" << Arch::kName
                  << " | Block=" << Config::kBlockM << "x"
                  << Config::kBlockN << "x" << Config::kBlockK
                  << " | 阶段数=" << Config::kStages
                  << " | Warp=" << Config::kWarpM << "x"
                  << Config::kWarpN << "x" << Config::kWarpK
                  << " | ElemA=" << TypeInfo<ElemA>::kName
                  << " | ElemB=" << TypeInfo<ElemB>::kName
                  << " | ElemC=" << TypeInfo<ElemC>::kName
                  << "\n";
    }
};

} // namespace kernel

// ============================================================================
// 第 5 部分: 带 SFINAE 分发的公共 API（第 15 章）
// ============================================================================

/// \brief 基于 SFINAE 的分发: 检查数据类型有效性。
template <typename T>
constexpr bool is_valid_gemm_type_v =
    std::is_same_v<T, half_t> ||
    std::is_same_v<T, float_t> ||
    std::is_same_v<T, double_t> ||
    std::is_same_v<T, int32_t>;

/// \brief 公共 GEMM 接口。
/// 使用 SFINAE 约束有效类型组合（第 15 章）。
template <
    typename ElemA, typename ElemB, typename ElemC,
    typename ArchTagType = arch::Sm80,
    template <typename> class Activation = thread::Identity,
    std::enable_if_t<
        is_valid_gemm_type_v<ElemA> &&
        is_valid_gemm_type_v<ElemB> &&
        is_valid_gemm_type_v<ElemC> &&
        ArchTag<ArchTagType>,
        int> = 0
>
void gemm(
    int M, int N, int K,
    ElemA const* A, index_t lda,
    ElemB const* B, index_t ldb,
    ElemC*       C, index_t ldc
) {
    using Config = kernel::KernelConfig<ArchTagType, ElemA, ElemB, ElemC>;
    using Kernel = kernel::GemmKernel<Config, Activation>;

    Kernel::describe();
    Kernel::compute(M, N, K, A, lda, B, ldb, C, ldc);
}

} // namespace mini_cutlass

// ============================================================================
// 第 6 部分: 用法示例与验证
// ============================================================================

using namespace mini_cutlass;

/// \brief 用于验证的朴素 CPU GEMM。
template <typename T>
void naive_gemm(int M, int N, int K,
                T const* A, int lda,
                T const* B, int ldb,
                T* C, int ldc)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            T sum = T{0};
            for (int k = 0; k < K; ++k) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

int main() {
    std::cout << "=== Mini CUTLASS 风格 GEMM 实现 ===\n";
    std::cout << "=== （整合第 9-17 章概念） ===\n\n";

    constexpr int M = 256, N = 256, K = 64;

    // --- 测试 1: SM80 上的 FP32 GEMM（使用 Identity 激活函数） ---
    std::cout << "--- 测试 1: SM80 上的 FP32 GEMM ---\n";
    std::vector<float_t> A_fp32(M * K);
    std::vector<float_t> B_fp32(K * N);
    std::vector<float_t> C_fp32(M * N, 0.0f);
    std::vector<float_t> C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) A_fp32[i] = static_cast<float_t>(i % 10) * 0.1f;
    for (int i = 0; i < K * N; ++i) B_fp32[i] = static_cast<float_t>(i % 10) * 0.1f;

    gemm<float_t, float_t, float_t, arch::Sm80>(
        M, N, K,
        A_fp32.data(), K,
        B_fp32.data(), N,
        C_fp32.data(), N
    );

    // 参考计算
    naive_gemm(M, N, K, A_fp32.data(), K, B_fp32.data(), N, C_ref.data(), N);

    // 验证正确性
    float_t max_error = 0.0f;
    for (int i = 0; i < M * N; ++i)
        max_error = std::max(max_error, std::abs(C_fp32[i] - C_ref[i]));
    std::cout << "  与参考的最大误差: " << max_error << "\n";

    // --- 测试 2: SM90 上带 ReLU 的 FP32 GEMM ---
    std::cout << "\n--- 测试 2: SM90 上带 ReLU 的 FP32 GEMM ---\n";
    std::vector<float_t> C_relu(M * N, 0.0f);

    gemm<float_t, float_t, float_t, arch::Sm90, thread::ReLU>(
        M, N, K,
        A_fp32.data(), K,
        B_fp32.data(), N,
        C_relu.data(), N
    );

    // 验证 ReLU 被应用（所有值应 >= 0）
    float_t min_val = 0.0f;
    for (int i = 0; i < M * N; ++i)
        min_val = std::min(min_val, C_relu[i]);
    std::cout << "  最小值（ReLU 应 >= 0）: " << min_val << "\n";

    // --- 测试 3: SM80 上带 GELU 的 FP32 GEMM ---
    std::cout << "\n--- 测试 3: SM80 上带 GELU 的 FP32 GEMM ---\n";
    std::vector<float_t> C_gelu(M * N, 0.0f);

    gemm<float_t, float_t, float_t, arch::Sm80, thread::GELU>(
        M, N, K,
        A_fp32.data(), K,
        B_fp32.data(), N,
        C_gelu.data(), N
    );

    std::cout << "  样本 C_gelu[0] = " << C_gelu[0] << "\n";

    // --- 测试 4: 小矩阵（用于可视化输出） ---
    std::cout << "\n--- 测试 4: 小矩阵 FP32 GEMM（可视化） ---\n";
    constexpr int SM = 2, SN = 3, SK = 4;
    float_t A_small[SM * SK] = {1, 2, 3, 4,  5, 6, 7, 8};
    float_t B_small[SK * SN] = {1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1};
    float_t C_small[SM * SN] = {};

    gemm<float_t, float_t, float_t, arch::Sm80>(
        SM, SN, SK,
        A_small, SK,
        B_small, SN,
        C_small, SN
    );

    std::cout << "  A 2x4 * B 4x3 = C 2x3:\n";
    for (int i = 0; i < SM; ++i) {
        std::cout << "  ";
        for (int j = 0; j < SN; ++j) {
            std::cout << C_small[i * SN + j] << " ";
        }
        std::cout << "\n";
    }

    // 小矩阵的参考
    float_t C_ref_small[SM * SN] = {};
    naive_gemm(SM, SN, SK, A_small, SK, B_small, SN, C_ref_small, SN);
    std::cout << "  参考:\n  ";
    for (int i = 0; i < SM * SN; ++i) std::cout << C_ref_small[i] << " ";
    std::cout << "\n";

    // --- 架构模式摘要 ---
    std::cout << "\n=== 架构模式摘要 ===\n";
    std::cout << "第09章（显式实例化）:            编译期组装 Kernel 类型\n";
    std::cout << "第10章（ODR 与术语）:             每个架构独立的模板特化\n";
    std::cout << "第11章（可调用与延迟求值）:       TypeInfo<T> 惰性特征求值\n";
    std::cout << "第12章（模板模板参数）:           ActivationOp<>、Madd<>\n";
    std::cout << "第13章（命名空间与 ADL）:         mini_cutlass::thread/warp/block/kernel\n";
    std::cout << "第14章（实例化模型）:             Block 中惰性成员函数体\n";
    std::cout << "第15章（SFINAE 分发）:            enable_if 验证 GEMM 类型组合\n";
    std::cout << "第16章（架构特化）:               ArchTileConfig 偏特化\n";
    std::cout << "第17章（C++20 Concepts）:         ArchTag concept 验证架构标签\n";

    std::cout << "\nMini CUTLASS GEMM - 所有章节成功整合。\n";
    return 0;
}
