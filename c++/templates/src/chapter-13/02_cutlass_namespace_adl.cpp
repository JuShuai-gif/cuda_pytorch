// ============================================================================
// 02_cutlass_namespace_adl.cpp - 模拟 CUTLASS 命名空间与 ADL 设计
// ============================================================================
//
// 目的:
//   CUTLASS 使用精心设计的命名空间层次结构，
//   结合 ADL 来实现可扩展、可组合的操作符。本文件
//   模拟了这种设计模式并解释其工作原理。
//
// CUTLASS 命名空间结构（简化版）:
//   cutlass::                          - 顶层
//   cutlass::gemm::                    - GEMM 操作
//   cutlass::epilogue::thread::        - 线程级 epilogue 操作
//   cutlass::epilogue::warp::          - Warp 级 epilogue 操作
//   cutlass::arch::                    - 架构标签/特征
//   cutlass::reference::               - 参考实现
//   cutlass::platform::                - 平台特定工具
//
// 这种层次结构允许 ADL 根据所涉及的类型找到
// 正确的操作符，而不会污染全局命名空间，
// 也不需要到处写完全限定名。
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <cmath>

// ============================================================================
// 第 1 节: 模拟 CUTLASS 命名空间层次结构
// ============================================================================

namespace cutlass {

// --- 平台工具 ---
namespace platform {

    /// \brief 模拟 CUDA 硬件特征。
    /// 在真实 CUTLASS 中，这些会探测实际的 GPU 能力。
    template <int kComputeCapability>
    struct HardwareTraits {
        static constexpr int compute_capability = kComputeCapability;
        static constexpr int warp_size          = 32;
        static constexpr int max_shared_memory  = (kComputeCapability >= 80) ? 163 * 1024 : 96 * 1024;
        static constexpr int max_registers      = 255;

        static void describe() {
            std::cout << "SM" << kComputeCapability
                      << " | Warp大小=" << warp_size
                      << " | 共享内存=" << max_shared_memory / 1024 << "KB"
                      << " | 最大寄存器数=" << max_registers << "\n";
        }
    };

} // namespace platform

// --- 架构标签 ---
namespace arch {

    // 架构标签类型 — 用作编译期标识符的空结构体
    struct Sm70 { static constexpr int id = 70; };
    struct Sm75 { static constexpr int id = 75; };
    struct Sm80 { static constexpr int id = 80; };
    struct Sm90 { static constexpr int id = 90; };

    // ADL 友好的特征查询
    template <typename ArchTag>
    int compute_capability(ArchTag) {
        return ArchTag::id;
    }

    // ADL 友好的: 将架构标签转换为特征（通过 ADL 在 ArchTag 上找到）
    template <typename ArchTag>
    auto make_traits(ArchTag tag) {
        if constexpr (std::is_same_v<ArchTag, Sm80>)
            return platform::HardwareTraits<80>{};
        else if constexpr (std::is_same_v<ArchTag, Sm90>)
            return platform::HardwareTraits<90>{};
        else
            return platform::HardwareTraits<75>{};
    }

} // namespace arch

// --- 数值类型 ---
namespace numeric {

    using float16_t = unsigned short;  // 模拟
    using float32_t = float;
    using bfloat16_t = unsigned int;   // 使用不同类型避免重定义

    // cutlass::numeric 中 ADL 友好的类型特征
    template <typename T>
    struct is_floating_point : std::is_floating_point<T> {};

    template <>
    struct is_floating_point<float16_t> : std::true_type {};

    template <>
    struct is_floating_point<bfloat16_t> : std::true_type {};

    // ADL 找到的函数: 数值类型之间的转换
    template <typename To, typename From>
    To numeric_conversion(From val) {
        return static_cast<To>(val);
    }

} // namespace numeric

// --- GEMM 命名空间 ---
namespace gemm {

    /// \brief Gemm 操作符配置。
    /// 每种张量类型生活在命名空间中；通过这些类型的 ADL
    /// 找到正确的 GEMM 实现。
    template <typename ArchTag,
              typename ElementA,
              typename ElementB,
              typename ElementC,
              typename AccumulatorT>
    struct GemmConfig {
        using Arch       = ArchTag;
        using ElemA      = ElementA;
        using ElemB      = ElementB;
        using ElemC      = ElementC;
        using Accum      = AccumulatorT;

        static void describe() {
            std::cout << "GemmConfig<Arch=" << Arch::id
                      << ", A=" << typeid(ElemA).name()
                      << ", B=" << typeid(ElemB).name()
                      << ", C=" << typeid(ElemC).name()
                      << ", 累加器=" << typeid(Accum).name()
                      << ">\n";
        }
    };

    // ADL 找到的函数: 为配置选择最佳 GEMM kernel
    // 不同命名空间中的不同重载可以通过 ADL 找到
    template <typename Config>
    void select_kernel(Config const& cfg) {
        // ADL 找到 arch::compute_capability 和 platform::HardwareTraits
        auto tag = typename Config::Arch{};
        int cc = arch::compute_capability(tag);  // ADL 在 cutlass::arch 中找到
        auto traits = arch::make_traits(tag);     // ADL 在 cutlass::arch 中找到
        traits.describe();
        cfg.describe();
    }

} // namespace gemm

// --- Epilogue 命名空间 ---
namespace epilogue {

    /// \brief Epilogue 操作命名空间 — 允许用户定义
    /// ADL 能找到的自定义操作。
    template <typename T>
    struct LinearCombination {
        T alpha, beta;

        T operator()(T accum, T source = T{0}) const {
            return alpha * accum + beta * source;
        }
    };

    // ADL 找到的操作符: 描述一个 epilogue 操作
    template <typename T>
    void describe(LinearCombination<T> const& epi) {
        std::cout << "线性组合(alpha=" << epi.alpha
                  << ", beta=" << epi.beta << ")\n";
    }

} // namespace epilogue

// --- 参考实现 ---
namespace reference {

    /// \brief 用于验证的参考 GEMM。
    /// 位于 cutlass::reference 命名空间中；ADL 使其与
    /// 优化实现保持分离。
    template <typename T>
    void gemm_reference(
        T const* A, T const* B, T* C,
        int M, int N, int K)
    {
        std::cout << "[reference::gemm] "
                  << M << "x" << N << "x" << K << " 使用 fp32\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = T{0};
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

} // namespace reference

} // namespace cutlass

// ============================================================================
// 第 2 节: 用户可扩展的 ADL 设计
// ============================================================================
//
// CUTLASS 允许用户在自己的命名空间中定义自定义类型和操作。
// ADL 确保根据参数类型找到正确的重载。

/// \brief 用户命名空间中用户自定义的激活函数。
/// 当用作 CUTLASS epilogue 的模板参数时，
/// ADL 在用户命名空间中找到正确的 describe()、
/// apply() 等。
namespace user_ops {

    struct MyCustomActivation {};

    // ADL 找到的函数: 自定义 apply
    template <typename T>
    T apply(MyCustomActivation, T val) {
        // 自定义: 斜率为 0.1 的 leaky ReLU
        return (val > T{0}) ? val : T{0.1} * val;
    }

    // ADL 找到的函数: 描述操作
    void describe(MyCustomActivation) {
        std::cout << "MyCustomActivation (LeakyReLU 斜率=0.1)\n";
    }

} // namespace user_ops

// ============================================================================
// 第 3 节: 基于 ADL 的分发（CUTLASS 模式）
// ============================================================================
//
// CUTLASS 使用 ADL 基于类型来分发操作。
// 编译器通过在所有参数类型的命名空间中查找
// 来找到正确的重载。

/// \brief 模拟使用 ADL 进行分发的 CUTLASS 操作符。
/// get_arch_traits() 函数通过 ADL 基于
/// 传递给它的 ArchTag 类型被找到。
template <typename ArchTag>
void configure_kernel_for_arch(ArchTag tag) {
    std::cout << "为以下架构配置 kernel: ";

    // ADL 在 cutlass::arch 中找到 compute_capability()
    int cc = compute_capability(tag);
    std::cout << "计算能力=" << cc / 10 << "." << cc % 10 << "\n";

    // ADL 在 cutlass::arch 中找到 make_traits()
    auto traits = make_traits(tag);
    traits.describe();
}

/// \brief 使用 ADL 找到的操作运行自定义 epilogue。
template <typename EpilogueOp, typename T>
T run_epilogue(EpilogueOp op, T value) {
    // ADL 在 EpilogueOp 的命名空间中找到 describe()
    describe(op);
    return op(value);
}

/// \brief 使用 ADL 找到的 apply() 运行自定义激活函数。
template <typename Activation, typename T>
T run_activation(Activation act, T value) {
    // ADL 在 Activation 的命名空间中找到 describe()
    describe(act);

    // ADL 在 Activation 的命名空间中找到 apply()
    return apply(act, value);
}

// ============================================================================
// 第 4 节: 编译期验证
// ============================================================================

using namespace cutlass;

// 验证架构标签
static_assert(arch::Sm80::id == 80);
static_assert(arch::Sm90::id == 90);

// 验证数值类型特征
static_assert(numeric::is_floating_point<numeric::float16_t>::value);
static_assert(numeric::is_floating_point<numeric::float32_t>::value);
static_assert(!numeric::is_floating_point<int>::value);

// 验证硬件特征
static_assert(platform::HardwareTraits<80>::warp_size == 32);
static_assert(platform::HardwareTraits<90>::max_shared_memory == 163 * 1024);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== CUTLASS 命名空间与 ADL 设计模拟 ===\n\n";

    // 第 1 节: 命名空间层次结构实战
    std::cout << "--- 架构特征（通过 ADL） ---\n";
    platform::HardwareTraits<80>::describe();
    platform::HardwareTraits<90>::describe();

    // ADL: compute_capability 在 cutlass::arch 中被找到，
    // 因为参数类型 (arch::Sm80) 位于 cutlass::arch 中
    std::cout << "\n通过 ADL 计算 capability(Sm80) = "
              << compute_capability(arch::Sm80{}) << "\n";

    std::cout << "\n--- GEMM 配置 ---\n";
    using Config = gemm::GemmConfig<
        arch::Sm80,
        numeric::float16_t,
        numeric::float16_t,
        numeric::float16_t,
        numeric::float32_t
    >;
    gemm::select_kernel(Config{});

    std::cout << "\n--- Epilogue 操作 ---\n";
    epilogue::LinearCombination<float> epi{1.0f, 0.0f};
    run_epilogue(epi, 5.0f);
    std::cout << "  结果: " << epi(5.0f) << "\n";

    epilogue::LinearCombination<float> epi2{2.0f, 1.0f};
    run_epilogue(epi2, 5.0f);
    std::cout << "  结果: " << epi2(5.0f, 3.0f) << "\n";

    // 第 2 节: 用户可扩展的 ADL
    std::cout << "\n--- 用户定义激活函数（通过 ADL） ---\n";
    user_ops::MyCustomActivation custom_act;
    float val1 = run_activation(custom_act, -3.0f);
    float val2 = run_activation(custom_act, 5.0f);
    std::cout << "  apply(-3.0) = " << val1 << "\n";
    std::cout << "  apply( 5.0) = " << val2 << "\n";

    // 第 3 节: 基于 ADL 的分发
    std::cout << "\n--- 基于 ADL 的 Kernel 配置 ---\n";
    configure_kernel_for_arch(arch::Sm80{});
    configure_kernel_for_arch(arch::Sm90{});

    // 第 1 节: 参考 GEMM
    std::cout << "\n--- 参考 GEMM ---\n";
    float A[6] = {1,2,3,4,5,6};
    float B[6] = {7,8,9,10,11,12};
    float C[4] = {};
    reference::gemm_reference(A, B, C, 2, 2, 3);
    std::cout << "  C = [" << C[0] << ", " << C[1]
              << ", " << C[2] << ", " << C[3] << "]\n";

    std::cout << "\n命名空间与 ADL 设计演示完成。\n";
    return 0;
}
