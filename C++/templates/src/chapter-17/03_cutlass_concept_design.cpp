// ============================================================================
// 03_cutlass_concept_design.cpp - 使用 C++20 Concepts 模拟
//                                  CUTLASS ArchTag 约束设计
// ============================================================================
//
// 目的:
//   使用 C++20 Concepts 重新设计 CUTLASS 的 ArchTag 系统和操作符约束。
//   Concepts 提供了一种自然的方式来表达
//   CUTLASS 当前通过 SFINAE、static_assert 和偏特化
//   强制执行的编译期需求。
//
// 使用 CONCEPTS 的关键改进:
//   1. 架构标签约束变成可读的 concept 名称
//   2. 操作符需求在模板定义时即被检查
//   3. 用户配置不支持的组合时有更好的错误消息
//   4. 基于 concept 的 kernel 选择重载决议
//   5. TMA/MMA 特性需求被表达为 concept 细化
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <concepts>
#include <string>
#include <cstdint>

// ============================================================================
// 第 1 节: 架构标签 Concept 层次结构
// ============================================================================

// --- 架构标签（与之前相同） ---
struct Sm70Tag {
    static constexpr int kCC = 70;
    static constexpr const char* name() { return "SM70 (Volta)"; }
};
struct Sm80Tag {
    static constexpr int kCC = 80;
    static constexpr const char* name() { return "SM80 (Ampere)"; }
};
struct Sm90Tag {
    static constexpr int kCC = 90;
    static constexpr const char* name() { return "SM90 (Hopper)"; }
};
struct Sm100Tag {
    static constexpr int kCC = 100;
    static constexpr const char* name() { return "SM100 (Blackwell)"; }
};

// --- Concept: 任意有效架构标签 ---
template <typename T>
concept ArchTag = requires {
    { T::kCC } -> std::convertible_to<int>;
    { T::name() } -> std::convertible_to<const char*>;
};

static_assert(ArchTag<Sm70Tag>);
static_assert(ArchTag<Sm80Tag>);
static_assert(ArchTag<Sm90Tag>);
static_assert(!ArchTag<int>);

// --- Concept 细化: 具有 Tensor Core 的架构 ---
template <typename T>
concept HasTensorCore = ArchTag<T> && (T::kCC >= 70);

// --- Concept 细化: 具有 Async Copy 的架构 ---
template <typename T>
concept HasAsyncCopy = ArchTag<T> && (T::kCC >= 80);

// --- Concept 细化: 具有 TMA（Tensor 内存加速器）的架构 ---
template <typename T>
concept HasTMA = ArchTag<T> && (T::kCC >= 90);

// --- Concept 细化: 具有 WGMMA 的架构 ---
template <typename T>
concept HasWGMMA = ArchTag<T> && (T::kCC >= 90);

// --- Concept 细化: 具有 FP4 支持的架构 ---
template <typename T>
concept HasFP4 = ArchTag<T> && (T::kCC >= 100);

static_assert(HasTensorCore<Sm80Tag>);
static_assert(HasAsyncCopy<Sm80Tag>);
static_assert(!HasAsyncCopy<Sm70Tag>);
static_assert(HasTMA<Sm90Tag>);
static_assert(!HasTMA<Sm80Tag>);
static_assert(HasWGMMA<Sm90Tag>);
static_assert(HasFP4<Sm100Tag>);
static_assert(!HasFP4<Sm90Tag>);

// ============================================================================
// 第 2 节: 数据类型 Concepts
// ============================================================================

// --- 模拟数值类型 ---
using float16_t = unsigned short;
using float32_t = float;
using float64_t = double;
using int8_t    = signed char;
using int32_t   = int;

// --- Concept: GEMM 中有效的元素类型 ---
template <typename T>
concept GemmElement = requires {
    // 任何可平凡拷贝的类型（GPU 要求）
    requires std::is_trivially_copyable_v<T>;
};

// --- Concept: 浮点元素 ---
template <typename T>
concept FloatingPointElement = GemmElement<T> && std::is_floating_point_v<T>;

// --- Concept: 半精度元素 ---
template <typename T>
concept HalfPrecisionElement = GemmElement<T> && (sizeof(T) == 2);

// --- Concept: 整数元素 ---
template <typename T>
concept IntegerElement = GemmElement<T> && std::is_integral_v<T>;

// --- Concept: 8 位整数元素 ---
template <typename T>
concept Int8Element = IntegerElement<T> && (sizeof(T) == 1);

// --- Concept: 有效累加器类型（必须至少与元素一样宽） ---
template <typename AccT, typename ElemT>
concept ValidAccumulator =
    GemmElement<AccT> &&
    GemmElement<ElemT> &&
    (sizeof(AccT) >= sizeof(ElemT));  // 累加器 >= 元素大小

static_assert(GemmElement<float32_t>);
static_assert(GemmElement<int32_t>);
static_assert(HalfPrecisionElement<float16_t>);
static_assert(Int8Element<int8_t>);
static_assert(ValidAccumulator<float32_t, float16_t>);
static_assert(!ValidAccumulator<int8_t, float32_t>);  // int8 对 float32 太窄

// ============================================================================
// 第 3 节: 通过 Concept 约束的重载进行 Kernel 选择
// ============================================================================
//
// 与 SFINAE enable_if 链不同，我们使用 concept 约束的
// 函数重载。编译器基于 concept 包含关系
// 选择最具体的匹配。

/// \brief SM70+（Volta）带 Tensor Core 的 Kernel。
template <HasTensorCore Arch, FloatingPointElement Elem>
struct TensorCoreKernel {
    static void launch(int M, int N, int K) {
        std::cout << "[TensorCoreKernel] 架构=" << Arch::name()
                  << " | mma.sync | " << M << "x" << N << "x" << K << "\n";
    }
};

/// \brief SM80+ 带异步拷贝的 Kernel。
template <HasAsyncCopy Arch, HalfPrecisionElement Elem>
struct AsyncCopyKernel {
    static void launch(int M, int N, int K) {
        std::cout << "[AsyncCopyKernel] 架构=" << Arch::name()
                  << " | cp.async + mma.sync | " << M << "x" << N << "x" << K << "\n";
    }
};

/// \brief SM90+ 带 TMA 的 Kernel。
template <HasTMA Arch, FloatingPointElement Elem>
struct TMAKernel {
    static void launch(int M, int N, int K) {
        std::cout << "[TMAKernel] 架构=" << Arch::name()
                  << " | TMA + wgmma | " << M << "x" << N << "x" << K << "\n";
    }
};

/// \brief SM100+ 带 FP4 的 Kernel。
template <HasFP4 Arch, GemmElement Elem>
struct FP4Kernel {
    static constexpr bool supports(Elem) { return true; }
    static void launch(int M, int N, int K) {
        std::cout << "[FP4Kernel] 架构=" << Arch::name()
                  << " | FP4 Tensor Core | " << M << "x" << N << "x" << K << "\n";
    }
};

/// \brief 回退 kernel（仅用于没有 Tensor Core 的 SM70 — 假设情况）。
template <ArchTag Arch, GemmElement Elem>
    requires (!HasTensorCore<Arch>)
struct FallbackKernel {
    static void launch(int M, int N, int K) {
        std::cout << "[FallbackKernel] 架构=" << Arch::name()
                  << " | FMA | " << M << "x" << N << "x" << K << "\n";
    }
};

// ============================================================================
// 第 4 节: Concept 约束的 Kernel 分发函数
// ============================================================================
//
// 分发函数使用 if constexpr 配合 concept 检查
// 在编译期选择最优 kernel。

template <ArchTag Arch, GemmElement ElemA, GemmElement ElemB,
          ValidAccumulator<ElemA> AccT>
void dispatch_gemm(int M, int N, int K,
                   ElemA const* A, ElemB const* B, AccT* C)
{
    std::cout << "[分发] ";
    std::cout << "架构=" << Arch::name() << " | ";

    // 通过 concept 检查进行编译期 kernel 选择
    if constexpr (HasFP4<Arch>) {
        std::cout << "已选择: FP4Kernel\n";
        FP4Kernel<Arch, ElemA>::launch(M, N, K);
    }
    else if constexpr (HasTMA<Arch> && FloatingPointElement<ElemA>) {
        std::cout << "已选择: TMAKernel\n";
        TMAKernel<Arch, ElemA>::launch(M, N, K);
    }
    else if constexpr (HasAsyncCopy<Arch> && HalfPrecisionElement<ElemA>) {
        std::cout << "已选择: AsyncCopyKernel\n";
        AsyncCopyKernel<Arch, ElemA>::launch(M, N, K);
    }
    else if constexpr (HasTensorCore<Arch> && FloatingPointElement<ElemA>) {
        std::cout << "已选择: TensorCoreKernel\n";
        TensorCoreKernel<Arch, ElemA>::launch(M, N, K);
    }
    else {
        std::cout << "已选择: FallbackKernel\n";
        FallbackKernel<Arch, ElemA>::launch(M, N, K);
    }
}

// ============================================================================
// 第 5 节: 通过 Concept 特化选择 Tile 大小
// ============================================================================

/// \brief Tile 大小配置: 主模板。
template <ArchTag Arch, typename = void>
struct GemmTileConfigC {
    static constexpr int kM = 64;
    static constexpr int kN = 64;
    static constexpr int kK = 8;
    static constexpr int kStages = 2;
};

/// \brief SM80 特化。
template <ArchTag Arch>
    requires HasAsyncCopy<Arch> && (!HasTMA<Arch>)
struct GemmTileConfigC<Arch> {
    static constexpr int kM = 256;
    static constexpr int kN = 128;
    static constexpr int kK = 32;
    static constexpr int kStages = 4;
};

/// \brief SM90+ 特化。
template <ArchTag Arch>
    requires HasTMA<Arch>
struct GemmTileConfigC<Arch> {
    static constexpr int kM = 256;
    static constexpr int kN = 256;
    static constexpr int kK = 64;
    static constexpr int kStages = 5;
};

// 验证 tile 选择
static_assert(GemmTileConfigC<Sm70Tag>::kM == 64);
static_assert(GemmTileConfigC<Sm80Tag>::kM == 256);
static_assert(GemmTileConfigC<Sm90Tag>::kN == 256);
static_assert(GemmTileConfigC<Sm100Tag>::kStages == 5);
static_assert(GemmTileConfigC<Sm80Tag>::kStages == 4);

// ============================================================================
// 第 6 节: 通过 Concepts 的 Epilogue 约束
// ============================================================================

/// \brief Concept: epilogue 操作（逐元素变换）。
template <typename Op, typename Elem>
concept EpilogueOp = requires(Op op, Elem val) {
    { op(val) } -> std::same_as<Elem>;   // 变换保持元素类型
};

/// \brief Concept: 带可选缩放的 epilogue。
template <typename Op, typename Elem, typename Scale>
concept ScalingEpilogueOp = EpilogueOp<Op, Elem> && requires(Op op, Elem val, Scale alpha) {
    { op(val, alpha) } -> std::same_as<Elem>;
};

// 示例 epilogue 操作
struct ReLU {
    float operator()(float x) const { return x > 0.0f ? x : 0.0f; }
    double operator()(double x) const { return x > 0.0 ? x : 0.0; }
};

struct LinearComb {
    float operator()(float x) const { return x; }
    float operator()(float x, float alpha) const { return alpha * x; }
};

static_assert(EpilogueOp<ReLU, float>);
static_assert(EpilogueOp<ReLU, double>);
static_assert(!EpilogueOp<ReLU, int>);  // ReLU 没有 int 重载
static_assert(ScalingEpilogueOp<LinearComb, float, float>);

/// \brief Concept 约束的带 epilogue 的 GEMM。
template <
    ArchTag Arch,
    FloatingPointElement Elem,
    ValidAccumulator<Elem> Acc,
    EpilogueOp<Elem> Epilogue
>
void gemm_with_epilogue(
    int M, int N, int K,
    Elem const* A, Elem const* B, Elem* C,
    Acc alpha, Acc beta,
    Epilogue epi
) {
    std::cout << "[GemmEpilogue] ";
    std::cout << "架构=" << Arch::name()
              << " | Tile=" << GemmTileConfigC<Arch>::kM
              << "x" << GemmTileConfigC<Arch>::kN
              << "x" << GemmTileConfigC<Arch>::kK
              << " | 阶段数=" << GemmTileConfigC<Arch>::kStages
              << "\n";

    // 模拟: 计算并应用 epilogue
    for (int i = 0; i < M * N; ++i) {
        Acc accum = 0;
        // （简化的 GEMM 计算）
        C[i] = epi(static_cast<Elem>(accum));
    }
}

// ============================================================================
// 第 7 节: 使用 Concept 静态断言的编译期验证
// ============================================================================

/// \brief 在编译期验证 GEMM 配置。
template <typename Arch, typename A, typename B, typename C>
constexpr bool validate_gemm_config() {
    // 这些静态断言如果违反会产生清晰的错误
    static_assert(ArchTag<Arch>,
        "Arch 必须是有效的架构标签（Sm70Tag、Sm80Tag 等）");
    static_assert(GemmElement<A>,
        "ElementA 必须是有效的 GEMM 元素类型");
    static_assert(GemmElement<B>,
        "ElementB 必须是有效的 GEMM 元素类型");
    static_assert(GemmElement<C>,
        "ElementC 必须是有效的 GEMM 元素类型");
    static_assert(ValidAccumulator<C, A>,
        "累加器类型必须至少与元素类型一样宽");
    return true;
}

static_assert(validate_gemm_config<Sm80Tag, float32_t, float32_t, float32_t>());
static_assert(validate_gemm_config<Sm90Tag, float16_t, float16_t, float32_t>());

// 这会产生清晰的错误:
// static_assert(validate_gemm_config<int, float, float, float>());
//                       ^ 错误: "Arch 必须是有效的架构标签"

// ============================================================================
// 第 8 节: Kernel 优先级的 Concept 包含
// ============================================================================
//
// Concepts 通过包含关系实现优先级化的重载决议。
// 更具体的 concept（例如 HasTMA）包含不那么具体的
// 一个（例如 HasAsyncCopy），所以当两者都匹配时
// HasTMA 重载被优先选择。

/// \brief 使用包含关系描述架构能力。
template <ArchTag Arch>
    requires HasTMA<Arch>
const char* arch_capability() {
    return "完整 TMA + WGMMA（Hopper+）";
}

template <ArchTag Arch>
    requires HasAsyncCopy<Arch> && (!HasTMA<Arch>)
const char* arch_capability() {
    return "异步拷贝 + Tensor Core（Ampere）";
}

template <ArchTag Arch>
    requires HasTensorCore<Arch> && (!HasAsyncCopy<Arch>)
const char* arch_capability() {
    return "Tensor Core（Volta/Turing）";
}

template <ArchTag Arch>
const char* arch_capability() {
    return "基本 FMA";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== CUTLASS 基于 Concept 的约束设计 ===\n\n";

    // 第 1 节: 架构标签 concepts
    std::cout << "--- 架构标签 Concepts ---\n";
    std::cout << std::boolalpha;
    std::cout << "Sm70 HasTensorCore: " << HasTensorCore<Sm70Tag> << "\n";
    std::cout << "Sm70 HasAsyncCopy:  " << HasAsyncCopy<Sm70Tag> << "\n";
    std::cout << "Sm80 HasTMA:        " << HasTMA<Sm80Tag> << "\n";
    std::cout << "Sm90 HasTMA:        " << HasTMA<Sm90Tag> << "\n";
    std::cout << "Sm100 HasFP4:       " << HasFP4<Sm100Tag> << "\n";

    // 第 4 节: 分发
    std::cout << "\n--- 基于 Concept 的分发 ---\n";
    float a[256] = {}, b[256] = {}, c[256] = {};

    dispatch_gemm<Sm70Tag>(64, 64, 32, a, b, c);
    dispatch_gemm<Sm80Tag>(128, 128, 64, a, b, c);
    dispatch_gemm<Sm90Tag>(256, 256, 128, a, b, c);
    dispatch_gemm<Sm100Tag>(512, 512, 256, a, b, c);

    // 第 5 节: Tile 大小选择
    std::cout << "\n--- 基于 Concept 的 Tile 大小 ---\n";
    std::cout << "SM70 Tile: " << GemmTileConfigC<Sm70Tag>::kM << "x"
              << GemmTileConfigC<Sm70Tag>::kN << "x"
              << GemmTileConfigC<Sm70Tag>::kK
              << " (" << GemmTileConfigC<Sm70Tag>::kStages << " 阶段)\n";
    std::cout << "SM80 Tile: " << GemmTileConfigC<Sm80Tag>::kM << "x"
              << GemmTileConfigC<Sm80Tag>::kN << "x"
              << GemmTileConfigC<Sm80Tag>::kK
              << " (" << GemmTileConfigC<Sm80Tag>::kStages << " 阶段)\n";
    std::cout << "SM90 Tile: " << GemmTileConfigC<Sm90Tag>::kM << "x"
              << GemmTileConfigC<Sm90Tag>::kN << "x"
              << GemmTileConfigC<Sm90Tag>::kK
              << " (" << GemmTileConfigC<Sm90Tag>::kStages << " 阶段)\n";

    // 第 6 节: Epilogue
    std::cout << "\n--- 带 Concepts 的 Epilogue ---\n";
    ReLU relu;
    float c2[64] = {};
    gemm_with_epilogue<Sm80Tag>(8, 8, 8, a, b, c2, 1.0f, 0.0f, relu);

    // 第 8 节: 包含关系
    std::cout << "\n--- Concept 包含 ---\n";
    std::cout << "Sm70 能力: " << arch_capability<Sm70Tag>() << "\n";
    std::cout << "Sm80 能力: " << arch_capability<Sm80Tag>() << "\n";
    std::cout << "Sm90 能力: " << arch_capability<Sm90Tag>() << "\n";
    std::cout << "Sm100 能力: " << arch_capability<Sm100Tag>() << "\n";

    std::cout << "\n基于 Concept 的 CUTLASS 约束设计完成。\n";
    return 0;
}
