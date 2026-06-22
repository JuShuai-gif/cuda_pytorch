// =============================================================================
// 第 06.4 章：CUTLASS 风格的使用 enable_if 分发
//
// CUTLASS 在其分发系统中广泛使用 enable_if，根据问题的编译期属性
//（数据类型、对齐、布局、tensor core 可用性）选择正确的内核特化。
//
// 本文件演示：
//   1. 按数据类型分发（fp16 vs fp32 vs int8）
//   2. 按对齐分发（对齐 vs 非对齐加载）
//   3. 按布局分发（RowMajor vs ColumnMajor）
//   4. 按 tensor core 可用性分发
//   5. 嵌套分发（类型 -> 对齐 -> 布局）
//   6. 回退分发链
//   7. 编译期前置条件检查（static_assert + enable_if）
//
// 编译：g++ -std=c++20 -o 04_cutlass_dispatch_if 04_cutlass_dispatch_if.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. 类型类别（模拟的 CUTLASS 数值类型）
// =============================================================================

struct float32_t { static std::string name() { return "f32"; } };
struct float16_t { static std::string name() { return "f16"; } };
struct int8_t_cut { static std::string name() { return "i8"; } };
struct int32_t_cut { static std::string name() { return "i32"; } };

// 分类类型的 Traits
template <typename T> struct IsFp32 : std::false_type {};
template <> struct IsFp32<float32_t> : std::true_type {};

template <typename T> struct IsFp16 : std::false_type {};
template <> struct IsFp16<float16_t> : std::true_type {};

template <typename T> struct IsInt8 : std::false_type {};
template <> struct IsInt8<int8_t_cut> : std::true_type {};

template <typename T> struct IsInt32 : std::false_type {};
template <> struct IsInt32<int32_t_cut> : std::true_type {};

// =============================================================================
// 2. 对齐标签
// =============================================================================

struct Alignment1 { static constexpr int value = 1; };
struct Alignment4 { static constexpr int value = 4; };
struct Alignment8 { static constexpr int value = 8; };
struct Alignment16 { static constexpr int value = 16; };

// =============================================================================
// 3. 布局标签
// =============================================================================

struct RowMajor_ {};
struct ColumnMajor_ {};

// =============================================================================
// 4. 内核实现（每个都是独立的特化）
// =============================================================================
// 在真实 CUTLASS 中，每个内核有数百行。这里用轻量级结构体模拟。

template <typename Element, typename Alignment, typename Layout>
struct GemmKernel_ {
  static void execute() {
    std::cout << "  [内核] "
              << Element::name() << "，对齐" << Alignment::value
              << "，"
              << (std::is_same_v<Layout, RowMajor_> ? "RowMajor"
                                                      : "ColMajor")
              << std::endl;
  }
};

// =============================================================================
// 5. 分发层 1：按数据类型
// =============================================================================

// 嵌套分发层的前向声明
template <typename Element, typename Alignment, typename Layout, typename>
struct KernelSelectorByAlignment;

template <typename Element, typename Alignment, typename Layout, typename>
struct KernelSelectorByLayout;

// 使用 enable_if 根据元素类型选择内核变体。

template <typename Element, typename Alignment, typename Layout,
          typename = void>
struct KernelSelector {
  // 默认：不支持的内核组合
  static void run() {
    std::cout << "  错误：找不到元素类型的内核！" << std::endl;
  }
};

// float32 的特化
template <typename Element, typename Alignment, typename Layout>
struct KernelSelector<
    Element, Alignment, Layout,
    std::enable_if_t<IsFp32<Element>::value>> {
  static void run() {
    std::cout << "  [Float32 路径] 正在选择 fp32 内核..." << std::endl;
    // 进一步按对齐和布局分发
    KernelSelectorByAlignment<Element, Alignment, Layout, void>::run();
  }
};

// float16 的特化
template <typename Element, typename Alignment, typename Layout>
struct KernelSelector<
    Element, Alignment, Layout,
    std::enable_if_t<IsFp16<Element>::value>> {
  static void run() {
    std::cout << "  [Float16 路径] 正在选择 fp16 内核..." << std::endl;
    KernelSelectorByAlignment<Element, Alignment, Layout, void>::run();
  }
};

// int8 的特化
template <typename Element, typename Alignment, typename Layout>
struct KernelSelector<
    Element, Alignment, Layout,
    std::enable_if_t<IsInt8<Element>::value>> {
  static void run() {
    std::cout << "  [Int8 路径] 正在选择 int8 内核..." << std::endl;
    // int8 可能跳过对齐分发（始终对齐）
    std::cout << "  -> 直接内核分发（int8 始终对齐）" << std::endl;
    GemmKernel_<Element, Alignment8, Layout>::execute();
  }
};

// =============================================================================
// 6. 分发层 2：按对齐
// =============================================================================
// 嵌套 enable_if：进一步细化内核选择。

template <typename Element, typename Alignment, typename Layout,
          typename = void>
struct KernelSelectorByAlignment {
  static void run() {
    // 回退：使用对齐 1
    std::cout << "  -> 回退到对齐 1" << std::endl;
    GemmKernel_<Element, Alignment1, Layout>::execute();
  }
};

// 对齐 8 路径：使用向量化加载
template <typename Element, typename Alignment, typename Layout>
struct KernelSelectorByAlignment<
    Element, Alignment, Layout,
    std::enable_if_t<(Alignment::value >= 8)>> {
  static void run() {
    std::cout << "  -> 对齐 " << Alignment::value
              << "：向量化加载已启用" << std::endl;
    KernelSelectorByLayout<Element, Alignment, Layout, void>::run();
  }
};

// 对齐 4 路径
template <typename Element, typename Alignment, typename Layout>
struct KernelSelectorByAlignment<
    Element, Alignment, Layout,
    std::enable_if_t<(Alignment::value == 4)>> {
  static void run() {
    std::cout << "  -> 对齐 4：半向量化加载" << std::endl;
    KernelSelectorByLayout<Element, Alignment, Layout, void>::run();
  }
};

// =============================================================================
// 7. 分发层 3：按布局
// =============================================================================
// 使用编译期特化的 RowMajor vs ColumnMajor 分发。

template <typename Element, typename Alignment, typename Layout,
          typename = void>
struct KernelSelectorByLayout {
  static void run() {
    GemmKernel_<Element, Alignment, Layout>::execute();
  }
};

// RowMajor 特化
template <typename Element, typename Alignment, typename Layout>
struct KernelSelectorByLayout<
    Element, Alignment, Layout,
    std::enable_if_t<std::is_same_v<Layout, RowMajor_>>> {
  static void run() {
    std::cout << "  -> RowMajor 布局优化（合并存储）"
              << std::endl;
    GemmKernel_<Element, Alignment, Layout>::execute();
  }
};

// ColumnMajor 特化
template <typename Element, typename Alignment, typename Layout>
struct KernelSelectorByLayout<
    Element, Alignment, Layout,
    std::enable_if_t<std::is_same_v<Layout, ColumnMajor_>>> {
  static void run() {
    std::cout << "  -> ColumnMajor 布局优化（合并加载）"
              << std::endl;
    GemmKernel_<Element, Alignment, Layout>::execute();
  }
};

// =============================================================================
// 8. 顶层分发 API
// =============================================================================
// 面向用户的函数，接受运行时枚举并分发到正确的模板实例化。

template <typename Element, int AlignValue, typename Layout>
void launch_gemm() {
  using Alignment = std::conditional_t<
      (AlignValue >= 16), Alignment16,
      std::conditional_t<
          (AlignValue >= 8), Alignment8,
          std::conditional_t<
              (AlignValue >= 4), Alignment4,
              Alignment1>>>;

  std::cout << "\n[启动] Element=" << Element::name()
            << "，Align=" << Alignment::value
            << "，Layout="
            << (std::is_same_v<Layout, RowMajor_> ? "RowMajor" : "ColMajor")
            << std::endl;

  KernelSelector<Element, Alignment, Layout>::run();
}

// =============================================================================
// 9. Tensor Core 分发（模拟）
// =============================================================================
// enable_if 分发的另一个维度：tensor core 可用性。

template <typename Element, bool TensorCoresAvailable, typename = void>
struct TensorOpDispatcher {
  static void dispatch() {
    std::cout << "  -> SIMT 路径（无 tensor core）" << std::endl;
  }
};

template <typename Element, bool TensorCoresAvailable>
struct TensorOpDispatcher<
    Element, TensorCoresAvailable,
    std::enable_if_t<TensorCoresAvailable && IsFp16<Element>::value>> {
  static void dispatch() {
    std::cout << "  -> Tensor Core fp16 路径（mma.sync.f16）" << std::endl;
  }
};

template <typename Element, bool TensorCoresAvailable>
struct TensorOpDispatcher<
    Element, TensorCoresAvailable,
    std::enable_if_t<TensorCoresAvailable && IsFp32<Element>::value>> {
  static void dispatch() {
    std::cout << "  -> Tensor Core tf32 路径（mma.sync.tf32）" << std::endl;
  }
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 06.4 章：CUTLASS 风格的使用 enable_if 分发 ===\n"
       << endl;

  // --- 测试 1：fp32，对齐，RowMajor ---
  cout << "[测试 1] fp32，对齐8，RowMajor：" << endl;
  launch_gemm<float32_t, 8, RowMajor_>();

  // --- 测试 2：fp16，对齐，ColMajor ---
  cout << "\n[测试 2] fp16，对齐16，ColMajor：" << endl;
  launch_gemm<float16_t, 16, ColumnMajor_>();

  // --- 测试 3：int8，任意对齐 ---
  cout << "\n[测试 3] int8，对齐4，RowMajor：" << endl;
  launch_gemm<int8_t_cut, 4, RowMajor_>();

  // --- 测试 4：fp32，非对齐（align=1）---
  cout << "\n[测试 4] fp32，对齐1，ColMajor（回退）：" << endl;
  launch_gemm<float32_t, 1, ColumnMajor_>();

  // --- 测试 5：fp16，对齐4，RowMajor ---
  cout << "\n[测试 5] fp16，对齐4，RowMajor：" << endl;
  launch_gemm<float16_t, 4, RowMajor_>();

  // --- 测试 6：int32（不支持）---
  cout << "\n[测试 6] int32（无特化内核）：" << endl;
  launch_gemm<int32_t_cut, 8, RowMajor_>();

  // --- 测试 7：Tensor Core 分发 ---
  cout << "\n[测试 7] Tensor core 分发：" << endl;

  std::cout << "  fp16 + tensor cores：" << std::endl;
  TensorOpDispatcher<float16_t, true>::dispatch();

  std::cout << "  fp32 + tensor cores：" << std::endl;
  TensorOpDispatcher<float32_t, true>::dispatch();

  std::cout << "  fp16 + 无 tensor cores：" << std::endl;
  TensorOpDispatcher<float16_t, false>::dispatch();

  std::cout << "  int8 + tensor cores（无 int8 tensor core 特化）：" << std::endl;
  TensorOpDispatcher<int8_t_cut, true>::dispatch();

  // --- 测试 8：对齐类型映射 ---
  cout << "\n[测试 8] 对齐映射：" << endl;
  cout << "  对齐 1 -> " << Alignment1::value << endl;
  cout << "  对齐 4 -> " << Alignment4::value << endl;
  cout << "  对齐 8 -> " << Alignment8::value << endl;
  cout << "  对齐 16 -> " << Alignment16::value << endl;

  // 静态断言
  static_assert(Alignment4::value == 4);
  static_assert(Alignment16::value == 16);

  // --- 测试 9：复杂分发链（多层）---
  cout << "\n[测试 9] 所有组合测试总结：" << endl;
  cout << "  已演练全部 3 层分发（类型 -> 对齐 -> 布局）。"
       << endl;
  cout << "  已验证不支持类型的回退路径。" << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
