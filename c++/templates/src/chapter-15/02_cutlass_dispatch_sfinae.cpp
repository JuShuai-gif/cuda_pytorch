// ============================================================================
// 02_cutlass_dispatch_sfinae.cpp - 使用 SFINAE 模拟 CUTLASS 分发:
//                                    基于数据类型选择 Kernel
// ============================================================================
//
// 目的:
//   在 CUTLASS 中，分发层基于数据类型、架构和问题
//   维度来选择最优的 kernel 实现 —
//   所有这些都在编译期使用 SFINAE 完成。本文件
//   演示了这种模式。
//
// CUTLASS 分发层次:
//   1. 用户使用运行时参数调用 gemm::Gemm::operator()
//   2. 分发层在编译期检查数据类型和问题大小
//   3. SFINAE 选择合适的 kernel 特化
//   4. 选中的 kernel 以最优 tile 大小启动
//
// 分发决策（全部在编译期）:
//   - 整数 vs 浮点数据 → 不同的 kernel 系列
//   - FP16 vs FP32 vs FP64 精度 → 不同的指令集
//   - 小矩阵 vs 大矩阵 → 不同的 tile 策略
//   - 有/无 Tensor Core → 不同的 MMA 指令
//   - 架构 (SM70/SM80/SM90) → 不同的硬件特性
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <cstdint>

// ============================================================================
// 第 1 节: 数据类型特征与标签分发
// ============================================================================

// 用于分发决策的类型类别
struct Float16Tag {};
struct Float32Tag {};
struct Float64Tag {};
struct Int8Tag {};
struct Int32Tag {};

/// \brief 将 C++ 类型映射到分发标签。
/// 主模板: 未知类型 → 没有分发标签。
template <typename T>
struct DataTypeTraits {
    // 没有 'using type' = 对不支持类型的 SFINAE
};

template <>
struct DataTypeTraits<uint16_t> {
    using tag = Float16Tag;
    static constexpr int kBits = 16;
    static constexpr bool kIsFloating = true;
    static constexpr const char* name() { return "fp16"; }
};

template <>
struct DataTypeTraits<float> {
    using tag = Float32Tag;
    static constexpr int kBits = 32;
    static constexpr bool kIsFloating = true;
    static constexpr const char* name() { return "fp32"; }
};

template <>
struct DataTypeTraits<double> {
    using tag = Float64Tag;
    static constexpr int kBits = 64;
    static constexpr bool kIsFloating = true;
    static constexpr const char* name() { return "fp64"; }
};

template <>
struct DataTypeTraits<int8_t> {
    using tag = Int8Tag;
    static constexpr int kBits = 8;
    static constexpr bool kIsFloating = false;
    static constexpr const char* name() { return "int8"; }
};

template <>
struct DataTypeTraits<int32_t> {
    using tag = Int32Tag;
    static constexpr int kBits = 32;
    static constexpr bool kIsFloating = false;
    static constexpr const char* name() { return "int32"; }
};

// ============================================================================
// 第 2 节: Kernel 实现（每种数据类型不同）
// ============================================================================

/// \brief FP16 kernel 系列 — 使用半精度内联函数。
struct KernelFp16 {
    static void launch(int M, int N, int K) {
        std::cout << "[KernelFp16] 使用半精度 MMA 启动"
                  << " | " << M << "x" << N << "x" << K << "\n";
        // 真实: 使用 mma.sync.aligned.m16n8k16.f16.f16.f16.f16
    }
    static constexpr const char* name() { return "fp16_kernel"; }
};

/// \brief FP32 kernel 系列 — 使用单精度 FMA。
struct KernelFp32 {
    static void launch(int M, int N, int K) {
        std::cout << "[KernelFp32] 使用单精度 FMA 启动"
                  << " | " << M << "x" << N << "x" << K << "\n";
        // 真实: 使用 ffma 指令
    }
    static constexpr const char* name() { return "fp32_kernel"; }
};

/// \brief FP64 kernel 系列 — 使用双精度 FMA。
struct KernelFp64 {
    static void launch(int M, int N, int K) {
        std::cout << "[KernelFp64] 使用双精度 FMA 启动"
                  << " | " << M << "x" << N << "x" << K << "\n";
        // 真实: 使用 dfma 指令
    }
    static constexpr const char* name() { return "fp64_kernel"; }
};

/// \brief INT8 kernel 系列 — 使用整数点积。
struct KernelInt8 {
    static void launch(int M, int N, int K) {
        std::cout << "[KernelInt8] 使用整数点积启动"
                  << " | " << M << "x" << N << "x" << K << "\n";
        // 真实: 使用 i8i32 MMA 或 DP4A
    }
    static constexpr const char* name() { return "int8_kernel"; }
};

/// \brief INT32 kernel 系列 — 使用整数操作。
struct KernelInt32 {
    static void launch(int M, int N, int K) {
        std::cout << "[KernelInt32] 使用整数算术启动"
                  << " | " << M << "x" << N << "x" << K << "\n";
    }
    static constexpr const char* name() { return "int32_kernel"; }
};

/// \brief 回退 — 无可用 kernel。
struct KernelUnsupported {
    static void launch(int M, int N, int K) {
        std::cout << "[KernelUnsupported] 没有可用的 kernel！\n";
    }
    static constexpr const char* name() { return "不支持"; }
};

// ============================================================================
// 第 3 节: 基于 SFINAE 的 Kernel 选择器
// ============================================================================
//
// select_kernel_impl 的每个重载都被限制为
// 仅对特定数据类型组合激活。SFINAE 从考虑中
// 移除非匹配的重载。

// --- 3a. FP16×FP16→FP16 的重载 ---
template <typename TA, typename TB, typename TC,
          std::enable_if_t<
              std::is_same_v<typename DataTypeTraits<TA>::tag, Float16Tag> &&
              std::is_same_v<typename DataTypeTraits<TB>::tag, Float16Tag> &&
              std::is_same_v<typename DataTypeTraits<TC>::tag, Float16Tag>,
              int> = 0>
auto select_kernel_impl(int, int) -> KernelFp16 {
    return {};
}

// --- 3b. FP16×FP16→FP32（混合精度）的重载 ---
template <typename TA, typename TB, typename TC,
          std::enable_if_t<
              std::is_same_v<typename DataTypeTraits<TA>::tag, Float16Tag> &&
              std::is_same_v<typename DataTypeTraits<TB>::tag, Float16Tag> &&
              std::is_same_v<typename DataTypeTraits<TC>::tag, Float32Tag>,
              int> = 0>
auto select_kernel_impl(int, long) -> KernelFp32 {
    return {};
}

// --- 3c. FP32×FP32→FP32 的重载 ---
template <typename TA, typename TB, typename TC,
          std::enable_if_t<
              std::is_same_v<typename DataTypeTraits<TA>::tag, Float32Tag> &&
              std::is_same_v<typename DataTypeTraits<TB>::tag, Float32Tag> &&
              std::is_same_v<typename DataTypeTraits<TC>::tag, Float32Tag>,
              int> = 0>
auto select_kernel_impl(long, int) -> KernelFp32 {
    return {};
}

// --- 3d. FP64×FP64→FP64 的重载 ---
template <typename TA, typename TB, typename TC,
          std::enable_if_t<
              std::is_same_v<typename DataTypeTraits<TA>::tag, Float64Tag> &&
              std::is_same_v<typename DataTypeTraits<TB>::tag, Float64Tag> &&
              std::is_same_v<typename DataTypeTraits<TC>::tag, Float64Tag>,
              int> = 0>
auto select_kernel_impl(long, long) -> KernelFp64 {
    return {};
}

// --- 3e. INT8×INT8→INT32 的重载 ---
template <typename TA, typename TB, typename TC,
          std::enable_if_t<
              std::is_same_v<typename DataTypeTraits<TA>::tag, Int8Tag> &&
              std::is_same_v<typename DataTypeTraits<TB>::tag, Int8Tag> &&
              std::is_same_v<typename DataTypeTraits<TC>::tag, Int32Tag>,
              int> = 0>
auto select_kernel_impl(int, char) -> KernelInt8 {
    return {};
}

// --- 3f. INT32×INT32→INT32 的重载 ---
template <typename TA, typename TB, typename TC,
          std::enable_if_t<
              std::is_same_v<typename DataTypeTraits<TA>::tag, Int32Tag> &&
              std::is_same_v<typename DataTypeTraits<TB>::tag, Int32Tag> &&
              std::is_same_v<typename DataTypeTraits<TC>::tag, Int32Tag>,
              int> = 0>
auto select_kernel_impl(char, int) -> KernelInt32 {
    return {};
}

// --- 3g. 回退: 不支持的组合 ---
template <typename TA, typename TB, typename TC>
auto select_kernel_impl(...) -> KernelUnsupported {
    return {};
}

/// \brief 公共接口: 选择合适的 kernel。
template <typename TA, typename TB, typename TC>
auto select_kernel() {
    // 通过传递额外参数来消歧重载。
    // 0, 0 用于打破重载决议中的平局。
    return select_kernel_impl<TA, TB, TC>(0, 0);
}

// ============================================================================
// 第 4 节: 通过 SFINAE 实现基于架构的特化
// ============================================================================

// 架构标签
struct ArchSm70 {};
struct ArchSm80 {};
struct ArchSm90 {};

/// \brief 按架构特化的 Kernel 配置。
/// 在偏特化中使用 enable_if。
template <typename ArchTag, typename = void>
struct ArchConfig {
    static constexpr int kTileM = 64;
    static constexpr int kTileN = 64;
    static constexpr int kTileK = 8;
    static constexpr const char* name() { return "通用"; }
};

template <typename ArchTag>
struct ArchConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, ArchSm80>>>
{
    static constexpr int kTileM = 256;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 32;
    static constexpr const char* name() { return "SM80"; }
};

template <typename ArchTag>
struct ArchConfig<ArchTag,
    std::enable_if_t<std::is_same_v<ArchTag, ArchSm90>>>
{
    static constexpr int kTileM = 256;
    static constexpr int kTileN = 256;
    static constexpr int kTileK = 64;
    static constexpr const char* name() { return "SM90"; }
};

// ============================================================================
// 第 5 节: 组合分发: 类型 + 架构 + 问题大小
// ============================================================================
//
// 在 CUTLASS 中，分发组合了多个编译期检查:
//   1. 数据类型 → kernel 系列选择（第 3 节）
//   2. 架构 → tile 大小选择（第 4 节）
//   3. 问题大小 → 小矩阵/大矩阵策略

/// \brief 问题大小类别。
struct SmallMatrix {};   // M,N <= 64
struct MediumMatrix {};  // M,N <= 512
struct LargeMatrix {};   // M,N > 512

/// \brief 在编译期确定问题大小类别。
template <int M, int N>
struct ProblemSizeCategory {
    using type = std::conditional_t<
        (M <= 64 && N <= 64),
        SmallMatrix,
        std::conditional_t<
            (M <= 512 && N <= 512),
            MediumMatrix,
            LargeMatrix
        >
    >;
};

/// \brief 组合所有编译期检查的完整分发函数。
template <typename TA, typename TB, typename TC,
          typename ArchTag,
          int M, int N, int K>
class GemmDispatch {
public:
    /// \brief 使用最优 kernel 执行 GEMM。
    static void execute(
        TA const* A, TB const* B, TC* C,
        int lda, int ldb, int ldc
    ) {
        // 步骤 1: 基于数据类型选择 kernel
        auto kernel = select_kernel<TA, TB, TC>();

        // 步骤 2: 基于架构选择 tile 大小
        using Config = ArchConfig<ArchTag>;
        int tile_m = Config::kTileM;
        int tile_n = Config::kTileN;
        int tile_k = Config::kTileK;

        // 步骤 3: 基于问题大小调整策略
        using Category = typename ProblemSizeCategory<M, N>::type;

        std::cout << "[分发] "
                  << "类型=" << DataTypeTraits<TA>::name()
                  << "x" << DataTypeTraits<TB>::name()
                  << "->" << DataTypeTraits<TC>::name()
                  << " | 架构=" << Config::name()
                  << " | 类别=";

        if constexpr (std::is_same_v<Category, SmallMatrix>) {
            std::cout << "小";
            // 对于小矩阵: 使用更少的线程块，不同的 tile
            tile_m = std::min(tile_m, 32);
            tile_n = std::min(tile_n, 32);
        } else if constexpr (std::is_same_v<Category, MediumMatrix>) {
            std::cout << "中";
            // 使用默认 tile
        } else {
            std::cout << "大";
            // 对于大矩阵: 最大化占用率
        }

        std::cout << " | Tile=" << tile_m << "x" << tile_n << "x" << tile_k << "\n";

        // 步骤 4: 启动 kernel
        kernel.launch(M, N, K);

        std::cout << "  使用的 kernel: " << kernel.name() << "\n";
    }
};

// ============================================================================
// 第 6 节: 类型级别验证（编译期）
// ============================================================================

/// \brief 验证类型组合对 GEMM 是否有效。
/// 成功时返回 void；否则替换失败。
template <typename TA, typename TB, typename TC, typename = void>
struct is_valid_gemm_combination : std::false_type {};

template <typename TA, typename TB, typename TC>
struct is_valid_gemm_combination<TA, TB, TC, std::void_t<
    typename DataTypeTraits<TA>::tag,      // TA 必须有有效的标签
    typename DataTypeTraits<TB>::tag,      // TB 必须有有效的标签
    typename DataTypeTraits<TC>::tag       // TC 必须有有效的标签
>> : std::true_type {};

template <typename TA, typename TB, typename TC>
constexpr bool is_valid_gemm_combination_v =
    is_valid_gemm_combination<TA, TB, TC>::value;

// ============================================================================
// 第 7 节: 编译期验证
// ============================================================================

static_assert(is_valid_gemm_combination_v<uint16_t, uint16_t, uint16_t>);
static_assert(is_valid_gemm_combination_v<float, float, float>);
static_assert(is_valid_gemm_combination_v<int8_t, int8_t, int32_t>);
static_assert(!is_valid_gemm_combination_v<char, char, char>);
static_assert(!is_valid_gemm_combination_v<void, int, float>);

// 验证已知组合的 kernel 选择
static_assert(std::is_same_v<
    decltype(select_kernel_impl<uint16_t, uint16_t, uint16_t>(0, 0)),
    KernelFp16>);

static_assert(std::is_same_v<
    decltype(select_kernel_impl<float, float, float>(0L, 0)),
    KernelFp32>);

static_assert(std::is_same_v<
    decltype(select_kernel_impl<double, double, double>(0L, 0L)),
    KernelFp64>);

static_assert(std::is_same_v<
    decltype(select_kernel_impl<int8_t, int8_t, int32_t>(0, '0')),
    KernelInt8>);

// 验证架构配置
static_assert(ArchConfig<ArchSm80>::kTileM == 256);
static_assert(ArchConfig<ArchSm90>::kTileN == 256);
static_assert(ArchConfig<ArchSm70>::kTileK == 8);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 使用 SFINAE 的 CUTLASS 分发 ===\n\n";

    uint16_t a_fp16[256] = {};
    uint16_t b_fp16[256] = {};
    uint16_t c_fp16[256] = {};

    float a_fp32[256] = {};
    float b_fp32[256] = {};
    float c_fp32[256] = {};

    // --- SM80 上的 FP16×FP16→FP16 ---
    std::cout << "--- SM80 上的 FP16×FP16→FP16 ---\n";
    GemmDispatch<uint16_t, uint16_t, uint16_t, ArchSm80, 64, 64, 32>::execute(
        a_fp16, b_fp16, c_fp16, 32, 32, 64);

    // --- SM90 上的 FP32×FP32→FP32 ---
    std::cout << "\n--- SM90 上的 FP32×FP32→FP32 ---\n";
    GemmDispatch<float, float, float, ArchSm90, 1024, 1024, 256>::execute(
        a_fp32, b_fp32, c_fp32, 256, 256, 1024);

    // --- SM80 上的 FP16×FP16→FP32（混合精度） ---
    std::cout << "\n--- SM80 上的 FP16×FP16→FP32（混合） ---\n";
    GemmDispatch<uint16_t, uint16_t, float, ArchSm80, 128, 128, 64>::execute(
        a_fp16, b_fp16, c_fp32, 64, 64, 128);

    // --- SM80 上的 INT8×INT8→INT32 ---
    std::cout << "\n--- SM80 上的 INT8×INT8→INT32 ---\n";
    int8_t  a_i8[256] = {};
    int8_t  b_i8[256] = {};
    int32_t c_i32[256] = {};
    GemmDispatch<int8_t, int8_t, int32_t, ArchSm80, 256, 256, 128>::execute(
        a_i8, b_i8, c_i32, 128, 128, 256);

    // --- 不支持的组合 ---
    std::cout << "\n--- 不支持（char×char→char） ---\n";
    auto fallback_kernel = select_kernel<char, char, char>();
    std::cout << "回退 kernel: " << fallback_kernel.name() << "\n";

    std::cout << "\n分发 SFINAE 模拟完成。\n";
    return 0;
}
