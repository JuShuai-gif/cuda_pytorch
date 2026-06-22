// ============================================================================
// 02_cutlass_instantiation_model.cpp - 模拟 CUTLASS 显式实例化策略
// ============================================================================
//
// 目的：
//   CUTLASS 将模板声明（头文件）与模板定义（实现头文件）分离，
//   并在专门的编译单元中提供显式实例化。本文件在单个文件中模拟
//   该模型，以方便教学理解。
//
// CUTLASS 模式：
//   - cutlass/gemm/gemm.h          ← 公共 API，仅声明
//   - cutlass/gemm/gemm_impl.h     ← 实现细节（由 .cu 包含）
//   - tools/library/.../gemm_sm80.cu ← 显式实例化
//
//   用户代码：    #include <cutlass/gemm/gemm.h>
//                 使用 extern templates（这里不生成代码）
//
//   构建系统：    gemm_sm80.cu → 显式模板实例化
//                 与用户代码链接
//
// ============================================================================

#include <iostream>
#include <type_traits>

// ============================================================================
// 第 0 部分：模拟的文件结构
// ============================================================================
//
// 本文件模拟三个概念性文件：
//   [1] 公共 API 头文件      （等价于 gemm.h）
//   [2] 实现细节              （等价于 gemm_impl.h）
//   [3] 显式实例化            （等价于 gemm_sm80.cu）
//
// 下面通过命名空间块和注释来界定它们。

// ============================================================================
// 第 1 部分：公共 API 头文件（等价于 gemm.h）
// ============================================================================
// 用户包含此文件。它声明了模板但不定义它们。
// extern template 声明抑制了隐式实例化。

namespace cutlass_sim {
namespace gemm {

// --- ArchTag：架构标识符（模拟） ---
struct ArchTagSm80 {};
struct ArchTagSm90 {};

// --- 元素类型 ---
using float16_t = short;   // 模拟半精度
using float32_t = float;   // 单精度
using int32_t   = int;

// --- Gemm 配置（类似于 cutlass::gemm::GemmConfig） ---
template <typename ArchTag,
          typename ElementA,
          typename ElementB,
          typename ElementC,
          typename Accumulator = float32_t>
struct GemmConfig {
    using Arch       = ArchTag;
    using Element_A  = ElementA;
    using Element_B  = ElementB;
    using Element_C  = ElementC;
    using Accum      = Accumulator;

    // 分块维度（简化版；真实的 CUTLASS 有更多参数）
    static constexpr int kTileM = 128;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 32;
};

// --- 前向声明的 Gemm kernel ---
// 实际实现在 impl 头文件中。
template <typename Config>
class Gemm {
public:
    // 构造函数接受问题维度（行数、列数、内维度）
    Gemm(int m, int n, int k, void const* A, void const* B, void* C);

    // 运行 GEMM 计算
    void run();

    // 查询配置
    static constexpr const char* arch_name();
    static constexpr const char* element_name();

private:
    int   m_, n_, k_;
    void const *A_, *B_;
    void *      C_;
};

// --- extern template 声明 ---
// 这些告诉编译器："不要在这个翻译单元中生成 Gemm<...>::Gemm() 等代码。
// 链接器会在显式实例化单元中找到它们。"
//
// 在真实的 CUTLASS 中，这些是由 Python 脚本生成的，
// 脚本会枚举所有受支持的配置，并为每个配置发出 extern 声明。

extern template class Gemm<GemmConfig<ArchTagSm80, float16_t, float16_t, float16_t>>;
extern template class Gemm<GemmConfig<ArchTagSm80, float16_t, float16_t, float32_t>>;
extern template class Gemm<GemmConfig<ArchTagSm90, float16_t, float16_t, float16_t>>;

// --- 为用户提供的便捷别名 ---
using GemmFp16Fp16Fp16Sm80 = Gemm<GemmConfig<ArchTagSm80, float16_t, float16_t, float16_t>>;
using GemmFp16Fp16Fp32Sm80 = Gemm<GemmConfig<ArchTagSm80, float16_t, float16_t, float32_t>>;

} // namespace gemm
} // namespace cutlass_sim

// ============================================================================
// 第 2 部分：实现细节（等价于 gemm_impl.h）
// ============================================================================
// 用户代码不会包含此文件。它只被提供显式实例化的 .cu 文件包含。
//
// 模式：#ifdef CUTLASS_IMPL_INCLUDE → include gemm_impl.h

namespace cutlass_sim {
namespace gemm {

// --- 成员函数定义 ---

template <typename Config>
Gemm<Config>::Gemm(int m, int n, int k, void const* A, void const* B, void* C)
    : m_(m), n_(n), k_(k), A_(A), B_(B), C_(C)
{}

template <typename Config>
void Gemm<Config>::run() {
    // 在真实的 CUTLASS 中，这里会启动设备 kernel，
    // 使用针对特定 Config 调优的 launch bounds。伪代码：
    //
    //   dim3 grid(ceil_div(m_, kTileM), ceil_div(n_, kTileN));
    //   dim3 block(256);
    //   device_kernel<Config><<<grid, block>>>(m_, n_, k_, A_, B_, C_);
    //
    std::cout << "[GEMM] Arch: " << arch_name()
              << ", Elements: " << element_name()
              << ", Dims: " << m_ << "x" << n_ << "x" << k_ << "\n";
}

template <typename Config>
constexpr const char* Gemm<Config>::arch_name() {
    if constexpr (std::is_same_v<typename Config::Arch, ArchTagSm80>)
        return "SM80 (A100)";
    else if constexpr (std::is_same_v<typename Config::Arch, ArchTagSm90>)
        return "SM90 (H100)";
    else
        return "Unknown";
}

template <typename Config>
constexpr const char* Gemm<Config>::element_name() {
    // 简化版：只检查 ElementC 作为演示
    if constexpr (std::is_same_v<typename Config::Element_C, float16_t>)
        return "fp16";
    else if constexpr (std::is_same_v<typename Config::Element_C, float32_t>)
        return "fp32";
    else
        return "unknown";
}

} // namespace gemm
} // namespace cutlass_sim

// ============================================================================
// 第 3 部分：显式实例化单元（等价于 gemm_sm80.cu）
// ============================================================================
// 在真实的 CUTLASS 中，这是用 nvcc 编译的独立 .cu 文件。
// 它包含 gemm_impl.h 并提供显式模板实例化定义。

// --- 模拟：通常这里会 #include impl 头文件 ---
// #include "gemm_impl.h"

// --- 显式模板实例化定义 ---
// 这些是 gemm.h 中 extern template 声明的匹配定义。
// 链接器将用户代码中的引用解析到这些定义。

namespace cutlass_sim {
namespace gemm {

template class Gemm<GemmConfig<ArchTagSm80, float16_t, float16_t, float16_t>>;
template class Gemm<GemmConfig<ArchTagSm80, float16_t, float16_t, float32_t>>;
template class Gemm<GemmConfig<ArchTagSm90, float16_t, float16_t, float16_t>>;

} // namespace gemm
} // namespace cutlass_sim

// ============================================================================
// 第 4 部分：用户代码（模拟）
// ============================================================================
// 这是用户的视角：他们包含公共头文件，
// 使用便捷别名，并链接到显式实例化目标文件。

void simulated_user_code() {
    using namespace cutlass_sim::gemm;

    float16_t a_buf[1024] = {};
    float16_t b_buf[1024] = {};
    float16_t c_buf[1024] = {};

    // 用户创建 Gemm 对象。由于 extern template 声明，
    // 构造函数不会在此 TU 中生成。
    // 它将由链接器从显式实例化单元中解析。
    GemmFp16Fp16Fp16Sm80 gemm(64, 64, 32, a_buf, b_buf, c_buf);
    gemm.run();
}

// ============================================================================
// 第 5 部分：显式实例化模型的优势
// ============================================================================
//
// 1. 编译时间：用户 TU 不会解析/实例化实现细节。
//    CUTLASS 头文件可能有 100K+ 行；没有 extern template，
//    每个翻译单元都会重新解析和重新实例化所有这些。
//
// 2. 二进制体积：每个特化只存在一个副本，而不是 N 个副本
//    （N = 使用它的 TU 数量）。链接器去重仍会消耗时间，
//    且 COMDAT 折叠并非保证。
//
// 3. KERNEL 缓存：显式实例化文件只需编译一次并缓存。
//    这使得构建系统优化如 ccache/sccache 成为可能。
//
// 4. 选择性：只有受支持的配置被实例化。用户不能
//    意外地实例化不支持的组合。
//
// 5. 关注点分离：架构特定的调优存在于 .cu 文件中，
//    与公共 API 分离。
//
// 权衡：
// - 需要构建系统支持来生成/管理实例化文件
// - 不能将所有成员函数定义为内联（必须提取到外部）
// - 头文件中的模板元编程仅限于类型计算；
//   运行期逻辑必须可分离

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== CUTLASS 显式实例化模型演示 ===\n\n";
    simulated_user_code();
    std::cout << "\n所有显式实例化均成功解析。\n";
    return 0;
}
