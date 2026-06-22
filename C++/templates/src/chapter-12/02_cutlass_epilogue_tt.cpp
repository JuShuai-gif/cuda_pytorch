// ============================================================================
// 02_cutlass_epilogue_tt.cpp - 使用模板模板参数模拟 CUTLASS Epilogue
// ============================================================================
//
// 目的：
//   CUTLASS epilogues 是 GEMM 计算的最后阶段。
//   在主矩阵乘累加之后，epilogue 应用逐元素操作
//   （例如 bias、激活、缩放）。
//   CUTLASS 使用模板模板参数在编译期将这些操作组合成
//   一个 DAG（有向无环图）。
//
// CUTLASS EPILOGUE 设计：
//   - 每个 epilogue 操作是一个类模板（template<typename> class）
//   - 操作通过模板模板参数进行组合
//   - epilogue 类型本身作为模板参数传递给 kernel
//   - 这允许数千种组合而无组合爆炸
//
// 真实 CUTLASS 示例 (cutlass/epilogue/thread/linear_combination.h)：
//   template <
//     typename ElementOutput_,
//     int Count,
//     typename ElementAccumulator_,
//     typename ElementCompute_,
//     template<typename> class ActivationOp = cutlass::epilogue::thread::Identity,
//     ...
//   >
//   class LinearCombination { ... };
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <cmath>
#include <vector>

// ============================================================================
// 第 1 部分：Epilogue 操作原语
// ============================================================================
//
// 每个 epilogue 操作是一个接受元素类型的类模板。
// 操作可以通过将它们作为模板模板参数嵌套来组合。

// --- 1a. 恒等（直通） ---
template <typename T>
struct Identity {
    static constexpr T apply(T val) { return val; }
    static constexpr const char* name() { return "Identity"; }
};

// --- 1b. ReLU 激活 ---
template <typename T>
struct ReLU {
    static constexpr T apply(T val) { return std::max(T{0}, val); }
    static constexpr const char* name() { return "ReLU"; }
};

// --- 1c. Sigmoid 激活（简化的快速 sigmoid） ---
template <typename T>
struct Sigmoid {
    static T apply(T val) {
        return T{1} / (T{1} + std::exp(-val));
    }
    static constexpr const char* name() { return "Sigmoid"; }
};

// --- 1d. GELU 激活（Gaussian Error Linear Unit，简化版） ---
template <typename T>
struct GELU {
    static T apply(T val) {
        // 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr T sqrt_2_over_pi = T{0.7978845608028654};
        T x3 = val * val * val;
        T inner = sqrt_2_over_pi * (val + T{0.044715} * x3);
        return T{0.5} * val * (T{1} + std::tanh(inner));
    }
    static constexpr const char* name() { return "GELU"; }
};

// --- 1e. Bias 加法（有状态操作） ---
// 注意：BiasAdd 是有状态的（存储 bias 值），这对于 CUTLASS epilogue 操作
// 来说是非典型的。大多数 CUTLASS epilogue 操作是无状态的函子。
// 这里包含是为了教学对比。
template <typename T>
struct BiasAdd {
    T bias;
    BiasAdd(T b) : bias(b) {}
    T apply(T val) const { return val + bias; }
};

// ============================================================================
// 第 2 部分：Epilogue 组合（通过 TTPs 构建 DAG）
// ============================================================================
//
// CUTLASS 通过嵌套模板模板参数来组合 epilogue 操作。
// 组合在编译期求值。

/// \brief 组合两个逐元素操作：Op1 ∘ Op2
/// 在 Op1 之后应用 Op2：result = Op2(Op1(val))
template <typename T,
          template <typename> class Op1 = Identity,
          template <typename> class Op2 = Identity,
          template <typename> class Op3 = Identity>
struct EpilogueTree {
    /// \brief 应用组合的操作树。
    /// 不能是 constexpr，因为某些操作使用非 constexpr 数学函数
    /// （例如 Sigmoid 使用 std::exp，GELU 使用 std::tanh）。
    static T apply(T val) {
        return Op3<T>::apply(Op2<T>::apply(Op1<T>::apply(val)));
    }

    /// \brief 描述操作链。
    static void describe() {
        std::cout << "EpilogueTree：" << Op1<T>::name()
                  << " → " << Op2<T>::name()
                  << " → " << Op3<T>::name() << "\n";
    }
};

// ============================================================================
// 第 3 部分：线性组合 Epilogue（CUTLASS 风格）
// ============================================================================
//
// LinearCombination epilogue 是 CUTLASS 中最常见的。
// 它计算：D = alpha * accum + beta * source + bias，
// 后跟一个可选的激活函数。

/// \brief 模拟的 LinearCombination epilogue。
/// 模板参数映射 CUTLASS 的设计：
///   - ElementD：输出元素类型
///   - ElementC：累加器类型
///   - ActivationOp：激活函数的模板模板参数
template <typename ElementD_,
          typename ElementC_ = ElementD_,
          template <typename> class ActivationOp = Identity>
class LinearCombinationEpilogue {
public:
    using ElementOutput     = ElementD_;
    using ElementAccumulator = ElementC_;

    /// \brief 计算 epilogue：D = alpha * C + beta * source（+ 激活）
    /// \param accum      来自 GEMM 的累加器值（C）
    /// \param source     源/输入矩阵值（用于残差、bias 等）
    /// \param alpha      GEMM 输出的缩放因子
    /// \param beta       源的缩放因子
    ElementOutput operator()(
        ElementAccumulator accum,
        ElementOutput      source,
        ElementAccumulator alpha = ElementAccumulator{1},
        ElementAccumulator beta  = ElementAccumulator{0}
    ) const {
        ElementAccumulator intermediate =
            alpha * accum + beta * static_cast<ElementAccumulator>(source);
        ElementOutput result = static_cast<ElementOutput>(intermediate);
        return ActivationOp<ElementOutput>::apply(result);
    }

    static void describe() {
        std::cout << "LinearCombination["
                  << "Accum=" << typeid(ElementAccumulator).name()
                  << ", Out=" << typeid(ElementOutput).name()
                  << ", Act=" << ActivationOp<ElementOutput>::name()
                  << "]\n";
    }
};

// ============================================================================
// 第 4 部分：预定义 Epilogue 类型别名（CUTLASS 风格）
// ============================================================================
//
// CUTLASS 提供了一系列常见 epilogue 配置的便捷别名。
// 用户从目录中选择一个，而不是手动构建类型。

// FP32 输出，带 ReLU 的线性组合
using EpilogueFp32Relu = LinearCombinationEpilogue<float, float, ReLU>;

// FP16 输出，带 GELU 的线性组合
using EpilogueFp16Gelu = LinearCombinationEpilogue<short, float, GELU>;

// FP32 输出，带 sigmoid 的线性组合
using EpilogueFp32Sigmoid = LinearCombinationEpilogue<float, float, Sigmoid>;

// FP32 输出，纯线性组合（无激活）
using EpilogueFp32Linear = LinearCombinationEpilogue<float, float, Identity>;

// ============================================================================
// 第 5 部分：高级 —— 带缩放和裁剪的 Epilogue
// ============================================================================

/// \brief 裁剪操作：将值限制在 [min, max]
template <typename T>
struct Clamp {
    static constexpr T apply(T val) {
        constexpr T kMin = T{0};
        constexpr T kMax = T{6};  // ReLU6 的典型值
        return std::min(std::max(val, kMin), kMax);
    }
    static constexpr const char* name() { return "Clamp(0,6)"; }
};

// 组合：LinearCombination → ReLU → Clamp
// 这就是 CUTLASS 用户创建自定义 epilogue 链的方式
template <typename T>
using ReLU6 = EpilogueTree<T, ReLU, Clamp>;

// ============================================================================
// 第 6 部分：完整的 GEMM + Epilogue 集成
// ============================================================================

/// \brief 集成了 epilogue 的模拟 GEMM kernel。
/// epilogue 作为模板参数传递，允许相同的 kernel 代码
/// 与任何 epilogue 配置一起工作。
template <typename ElementA,
          typename ElementB,
          typename ElementAccum,
          typename Epilogue>
class GemmWithEpilogue {
public:
    using EpilogueType = Epilogue;

    /// \brief 执行 GEMM：C = Epilogue(A * B)
    void compute(
        ElementA const* A,
        ElementB const* B,
        typename Epilogue::ElementOutput* C,
        int M, int N, int K,
        typename Epilogue::ElementAccumulator alpha = 1,
        typename Epilogue::ElementAccumulator beta  = 0
    ) {
        Epilogue epilogue;
        std::cout << "GemmWithEpilogue：" << M << "x" << N << "x" << K << "\n";
        Epilogue::describe();

        // 模拟：对每个元素计算点积 + epilogue
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                // 计算 A 的第 i 行与 B 的第 j 列的点积
                typename Epilogue::ElementAccumulator accum = 0;
                for (int k = 0; k < K; ++k) {
                    accum += static_cast<typename Epilogue::ElementAccumulator>(
                        A[i * K + k]) *
                        static_cast<typename Epilogue::ElementAccumulator>(
                        B[k * N + j]);
                }
                // 应用 epilogue
                C[i * N + j] = epilogue(accum, 0, alpha, beta);
            }
        }
    }
};

// ============================================================================
// 第 7 部分：编译期验证
// ============================================================================

// 验证 constexpr 友好的操作（ReLU、Identity、Clamp 是 constexpr 的）
static_assert(ReLU<float>::apply(-1.0f) == 0.0f);
static_assert(ReLU<float>::apply( 1.0f) == 1.0f);
static_assert(Identity<float>::apply(42.0f) == 42.0f);
// 注意：7.0 会被裁剪到 6.0，但 constexpr 上下文可能有所不同

// 验证 epilogue 类型正确组合
using TestEpilogue = LinearCombinationEpilogue<float, float, ReLU>;
static_assert(std::is_same_v<TestEpilogue::ElementOutput, float>);
static_assert(std::is_same_v<TestEpilogue::ElementAccumulator, float>);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== CUTLASS Epilogue 模板模板参数设计 ===\n\n";

    // 第 2 部分：EpilogueTree 组合
    std::cout << "--- EpilogueTree 测试 ---\n";
    std::cout << "ReLU ∘ Identity(" << -5.0f << ") = "
              << EpilogueTree<float, ReLU>::apply(-5.0f) << "\n";
    std::cout << "ReLU ∘ Identity(" << 5.0f << ") = "
              << EpilogueTree<float, ReLU>::apply(5.0f) << "\n";

    std::cout << "\nGELU(" << 1.0f << ") = "
              << GELU<float>::apply(1.0f) << "\n";
    std::cout << "GELU(" << -1.0f << ") = "
              << GELU<float>::apply(-1.0f) << "\n";

    // 第 3 部分：LinearCombination epilogue
    std::cout << "\n--- LinearCombination 测试 ---\n";
    EpilogueFp32Relu epi_relu;
    float result = epi_relu(2.5f, 0.0f, 1.0f, 0.0f);
    std::cout << "LC(2.5) w/ReLU = " << result << "\n";

    result = epi_relu(-2.5f, 0.0f, 1.0f, 0.0f);
    std::cout << "LC(-2.5) w/ReLU = " << result << "\n";

    // 使用 beta 测试（残差加）
    EpilogueFp32Linear epi_linear;
    result = epi_linear(3.0f, 1.0f, 1.0f, 0.5f);
    std::cout << "LC(3.0, src=1.0, alpha=1, beta=0.5) = " << result << "\n";

    // 第 5 部分：ReLU6
    std::cout << "\n--- ReLU6（ReLU + Clamp） ---\n";
    ReLU6<float>::describe();
    std::cout << "ReLU6(-2.0) = " << ReLU6<float>::apply(-2.0f) << "\n";
    std::cout << "ReLU6( 3.0) = " << ReLU6<float>::apply(3.0f) << "\n";
    std::cout << "ReLU6( 7.0) = " << ReLU6<float>::apply(7.0f) << "\n";

    // 第 6 部分：完整的 GEMM + epilogue 集成
    std::cout << "\n--- GEMM + Epilogue 集成 ---\n";
    std::vector<float> A = {1, 2, 3, 4, 5, 6};   // 2x3
    std::vector<float> B = {7, 8, 9, 10, 11, 12}; // 3x2
    std::vector<float> C(4);                        // 2x2

    using GemmRelu = GemmWithEpilogue<float, float, float, EpilogueFp32Relu>;
    GemmRelu gemm;
    gemm.compute(A.data(), B.data(), C.data(), 2, 2, 3);

    std::cout << "GEMM+ReLU 后的 C 矩阵：\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << C[i * 2 + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nEpilogue TTP 设计演示完成。\n";
    return 0;
}
