// =============================================================================
// 第 06.3 章：检测类型作为模板特化
//
// 一个常见的元编程任务：判断给定类型 T 是否是特定模板的实例化
//（例如，T 是 std::vector<...> 吗？T 是 std::shared_ptr<...> 吗？）。
//
// 本文件实现：
//   1. is_specialization_of<T, Template>：检测任意实例化
//   2. 从特化中提取模板实参
//   3. 检测特定容器类型（vector、list、map 等）
//   4. 检测智能指针类型（unique_ptr、shared_ptr）
//   5. 检测 std::optional、std::variant、std::tuple
//   6. CUTLASS 风格：检测类型是否是 GemmConfiguration
//   7. CUTLASS 风格：检测布局特化
//   8. 使用通配符进行偏特化匹配
//
// 编译：g++ -std=c++20 -o 03_is_same_specialization 03_is_same_specialization.cpp
// =============================================================================

#include <cassert>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

// =============================================================================
// 1. is_specialization_of：核心检测模板
// =============================================================================
// 主模板：T 不是 Template 的特化
// 偏特化：T 匹配 Template<Args...>

template <typename T, template <typename...> class Template>
struct IsSpecializationOf : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct IsSpecializationOf<Template<Args...>, Template> : std::true_type {};

template <typename T, template <typename...> class Template>
inline constexpr bool is_specialization_of_v =
    IsSpecializationOf<T, Template>::value;

// =============================================================================
// 2. 提取模板实参
// =============================================================================
// 给定特化 T = SomeTemplate<A, B, C>，提取 A、B、C。
// 主模板处理非特化情况（空包）。

template <typename T>
struct TemplateArgs;

template <template <typename...> class Template, typename... Args>
struct TemplateArgs<Template<Args...>> {
  // 模板实参数量
  static constexpr std::size_t count = sizeof...(Args);

  // 使用辅助按索引提取
  template <std::size_t I>
  using at = std::tuple_element_t<I, std::tuple<Args...>>;
};

// =============================================================================
// 3. 常见类型的便捷别名
// =============================================================================

// --- std::vector ---
template <typename T>
using is_vector = IsSpecializationOf<T, std::vector>;

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

// --- std::list ---
template <typename T>
using is_list = IsSpecializationOf<T, std::list>;

template <typename T>
inline constexpr bool is_list_v = is_list<T>::value;

// --- std::unique_ptr ---
template <typename T>
using is_unique_ptr = IsSpecializationOf<T, std::unique_ptr>;

template <typename T>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;

// --- std::shared_ptr ---
template <typename T>
using is_shared_ptr = IsSpecializationOf<T, std::shared_ptr>;

template <typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

// --- std::optional ---
template <typename T>
using is_optional = IsSpecializationOf<T, std::optional>;

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

// --- std::tuple ---
template <typename T>
inline constexpr bool is_tuple_v =
    IsSpecializationOf<T, std::tuple>::value;

// --- std::variant ---
template <typename T>
inline constexpr bool is_variant_v =
    IsSpecializationOf<T, std::variant>::value;

// --- std::map（2 个参数）---
// 注意：std::map 有默认的比较器和分配器，因此可以有 4 个参数
// 我们使用更灵活的检测：检查是否匹配 std::map<A, B, ...>
template <typename T>
struct IsMap : std::false_type {};

template <typename K, typename V, typename... Rest>
struct IsMap<std::map<K, V, Rest...>> : std::true_type {};

template <typename T>
inline constexpr bool is_map_v = IsMap<T>::value;

// =============================================================================
// 4. 智能指针 Trait 提取
// =============================================================================
// 从智能指针中提取元素类型。

template <typename T>
struct SmartPointerTraits {
  static constexpr bool is_smart_ptr = false;
};

template <typename T>
struct SmartPointerTraits<std::unique_ptr<T>> {
  static constexpr bool is_smart_ptr = true;
  using element_type = T;
};

template <typename T>
struct SmartPointerTraits<std::shared_ptr<T>> {
  static constexpr bool is_smart_ptr = true;
  using element_type = T;
};

// =============================================================================
// 5. 通用容器元素类型提取器
// =============================================================================
// 对于可能有 value_type 的任何容器类类型...

template <typename T, typename = void>
struct ContainerElement {
  using type = void;  // 非识别的容器
};

template <typename T>
struct ContainerElement<T, std::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};

template <typename T>
using container_element_t = typename ContainerElement<T>::type;

// =============================================================================
// 6. CUTLASS 风格：检测 GemmConfiguration 特化
// =============================================================================

// CUTLASS 布局标签
struct RowMajor {};
struct ColumnMajor {};

// CUTLASS 风格 GemmConfiguration（最小化）
template <typename Shape, typename Mainloop, typename Epilogue,
          typename ElementA = float, typename ElementB = float,
          typename ElementC = float>
struct GemmConfiguration {
  using shape    = Shape;
  using mainloop = Mainloop;
  using epilogue = Epilogue;
};

// Trait：类型是 GemmConfiguration 吗？
template <typename T>
struct IsGemmConfig : std::false_type {};

template <typename... Args>
struct IsGemmConfig<GemmConfiguration<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_gemm_config_v = IsGemmConfig<T>::value;

// 从 GemmConfiguration 提取 Shape（或 void 如果不是配置）
template <typename T, typename = void>
struct GetGemmShape {
  using type = void;
};

template <typename... Args>
struct GetGemmShape<GemmConfiguration<Args...>> {
  using type = typename GemmConfiguration<Args...>::shape;
};

// =============================================================================
// 7. CUTLASS 风格：检测布局特化
// =============================================================================
// 在 CUTLASS 中，某些类型是"布局标签"（RowMajor、ColumnMajor）。
// 检测类型是否是有效的布局。

template <typename T>
struct IsLayout : std::false_type {};

template <>
struct IsLayout<RowMajor> : std::true_type {};

template <>
struct IsLayout<ColumnMajor> : std::true_type {};

template <typename T>
inline constexpr bool is_layout_v = IsLayout<T>::value;

// =============================================================================
// 8. 变参模板检测：is_any_of
// =============================================================================
// 检查类型是否是任意一组模板的特化。

template <typename T, template <typename...> class... Templates>
struct IsAnyOf;

template <typename T>
struct IsAnyOf<T> : std::false_type {};

template <typename T, template <typename...> class First,
          template <typename...> class... Rest>
struct IsAnyOf<T, First, Rest...>
    : std::conditional_t<IsSpecializationOf<T, First>::value,
                         std::true_type,
                         IsAnyOf<T, Rest...>> {};

template <typename T, template <typename...> class... Templates>
inline constexpr bool is_any_of_v = IsAnyOf<T, Templates...>::value;

// 检查类型是否是标准序列容器
template <typename T>
inline constexpr bool is_std_sequence_v =
    is_any_of_v<T, std::vector, std::list, std::deque>;

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 06.3 章：检测类型作为模板特化 ===\n" << endl;

  // --- 测试 1：is_specialization_of ---
  cout << "[测试 1] is_specialization_of：" << endl;
  static_assert(is_specialization_of_v<vector<int>, std::vector>);
  static_assert(is_specialization_of_v<list<double>, std::list>);
  static_assert(is_specialization_of_v<tuple<int, char>, std::tuple>);
  static_assert(!is_specialization_of_v<int, std::vector>);
  static_assert(!is_specialization_of_v<vector<int>, std::list>);

  cout << "  vector<int> 是 vector：   " << is_vector_v<vector<int>> << endl;
  cout << "  int 是 vector：           " << is_vector_v<int> << endl;
  cout << "  list<double> 是 list：    " << is_list_v<list<double>> << endl;

  // --- 测试 2：TemplateArgs 提取 ---
  cout << "\n[测试 2] TemplateArgs 提取：" << endl;
  using VArgs = TemplateArgs<vector<int>>;
  static_assert(VArgs::count == 2);  // T, Alloc
  static_assert(is_same_v<VArgs::at<0>, int>);

  using TArgs = TemplateArgs<tuple<int, double, char>>;
  static_assert(TArgs::count == 3);
  static_assert(is_same_v<TArgs::at<0>, int>);
  static_assert(is_same_v<TArgs::at<1>, double>);
  static_assert(is_same_v<TArgs::at<2>, char>);
  cout << "  vector<int> 有 " << VArgs::count << " 个模板实参" << endl;
  cout << "  tuple<int,double,char> 有 " << TArgs::count
       << " 个模板实参" << endl;

  // --- 测试 3：智能指针检测 ---
  cout << "\n[测试 3] 智能指针检测：" << endl;
  static_assert(is_unique_ptr_v<unique_ptr<int>>);
  static_assert(is_shared_ptr_v<shared_ptr<double>>);
  static_assert(!is_unique_ptr_v<shared_ptr<int>>);
  static_assert(!is_unique_ptr_v<int*>);

  cout << "  unique_ptr<int>：   unique_ptr=" << is_unique_ptr_v<unique_ptr<int>>
       << "，shared_ptr=" << is_shared_ptr_v<unique_ptr<int>> << endl;
  cout << "  shared_ptr<double>：unique_ptr="
       << is_unique_ptr_v<shared_ptr<double>>
       << "，shared_ptr=" << is_shared_ptr_v<shared_ptr<double>> << endl;

  // 智能指针 traits
  static_assert(
      is_same_v<SmartPointerTraits<unique_ptr<int>>::element_type, int>);
  static_assert(
      is_same_v<SmartPointerTraits<shared_ptr<double>>::element_type, double>);

  // --- 测试 4：optional、variant、tuple 检测 ---
  cout << "\n[测试 4] optional/variant/tuple 检测：" << endl;
  static_assert(is_optional_v<optional<int>>);
  static_assert(is_variant_v<variant<int, double>>);
  static_assert(is_tuple_v<tuple<>>);
  static_assert(!is_optional_v<int>);

  cout << "  optional<int>：optional=" << is_optional_v<optional<int>>
       << endl;
  cout << "  variant<int,double>：variant="
       << is_variant_v<variant<int, double>> << endl;
  cout << "  空 tuple：tuple=" << is_tuple_v<tuple<>> << endl;

  // --- 测试 5：map 检测 ---
  cout << "\n[测试 5] map 检测：" << endl;
  static_assert(is_map_v<map<int, double>>);
  static_assert(!is_map_v<vector<int>>);
  cout << "  map<int,double>：" << is_map_v<map<int, double>> << endl;

  // --- 测试 6：容器元素提取 ---
  cout << "\n[测试 6] 容器元素提取：" << endl;
  static_assert(is_same_v<container_element_t<vector<int>>, int>);
  static_assert(is_same_v<container_element_t<list<string>>, string>);
  static_assert(is_same_v<container_element_t<int>, void>);

  cout << "  vector<int>::element_type = "
       << typeid(container_element_t<vector<int>>).name() << endl;
  cout << "  int::element_type = "
       << typeid(container_element_t<int>).name() << "（void）" << endl;

  // --- 测试 7：CUTLASS GemmConfiguration 检测 ---
  cout << "\n[测试 7] GemmConfiguration 检测：" << endl;
  struct DummyShape {};
  struct DummyMainloop {};
  struct DummyEpilogue {};

  using Cfg = GemmConfiguration<DummyShape, DummyMainloop, DummyEpilogue>;

  static_assert(is_gemm_config_v<Cfg>);
  static_assert(!is_gemm_config_v<int>);

  cout << "  GemmConfiguration：is_config=" << is_gemm_config_v<Cfg> << endl;
  cout << "  int：is_config=" << is_gemm_config_v<int> << endl;

  // --- 测试 8：布局检测 ---
  cout << "\n[测试 8] 布局检测：" << endl;
  static_assert(is_layout_v<RowMajor>);
  static_assert(is_layout_v<ColumnMajor>);
  static_assert(!is_layout_v<int>);

  cout << "  RowMajor：    " << is_layout_v<RowMajor> << endl;
  cout << "  ColumnMajor：" << is_layout_v<ColumnMajor> << endl;
  cout << "  int：        " << is_layout_v<int> << endl;

  // --- 测试 9：is_any_of ---
  cout << "\n[测试 9] is_any_of（多模板）：" << endl;
  cout << "  vector<int> 是序列："
       << is_std_sequence_v<vector<int>> << endl;
  cout << "  list<double> 是序列："
       << is_std_sequence_v<list<double>> << endl;
  cout << "  set<int> 是序列："
       << is_std_sequence_v<set<int>> << endl;

  static_assert(is_std_sequence_v<vector<int>>);
  static_assert(is_std_sequence_v<list<double>>);
  static_assert(!is_std_sequence_v<set<int>>);  // set 不在列表中

  cout << "\n所有测试通过！" << endl;
  return 0;
}
