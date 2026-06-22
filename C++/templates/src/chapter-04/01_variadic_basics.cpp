// =============================================================================
// 第 04.1 章：变参模板基础
//
// 变参模板（C++11）允许模板接受任意数量的模板实参。
// 本文件涵盖：
//   1. 参数包语法（typename... Ts、Args... args）
//   2. 递归解包（头/尾模式）
//   3. Fold expressions（C++17）：一元左折叠、一元右折叠、二元折叠
//   4. sizeof... 运算符
//   5. 变参打印函数
//   6. 变参求和/求积
//   7. 变参异构 max
//   8. 变参数组构造
//   9. 变参类元组
//
// 编译：g++ -std=c++20 -o 01_variadic_basics 01_variadic_basics.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

// =============================================================================
// 1. sizeof... 运算符
// =============================================================================
// 在编译期返回参数包中的元素数量。

template <typename... Ts>
constexpr std::size_t count_types() {
  return sizeof...(Ts);
}

template <int... Values>
constexpr std::size_t count_values() {
  return sizeof...(Values);
}

// =============================================================================
// 2. 递归变参打印（C++17 之前风格）
// =============================================================================
// 经典的头/尾递归。基例处理最后一个元素。

// 基例：单个元素
template <typename T>
void print_recursive(T const& last) {
  std::cout << last << std::endl;
}

// 递归情况：打印头，递归处理尾
template <typename T, typename... Ts>
void print_recursive(T const& first, Ts const&... rest) {
  std::cout << first;
  if constexpr (sizeof...(rest) > 0) {
    std::cout << ", ";
  }
  print_recursive(rest...);
}

// =============================================================================
// 3. Fold Expressions（C++17）
// =============================================================================
// Fold expressions 简化了参数包上的操作。四种形式：
//   ( pack op ... )          -- 一元右折叠：E1 op (E2 op (E3 op E4))
//   ( ... op pack )          -- 一元左折叠：((E1 op E2) op E3) op E4
//   ( pack op ... op init )  -- 二元右折叠：E1 op (E2 op (E3 op init))
//   ( init op ... op pack )  -- 二元左折叠：((init op E1) op E2) op E3

template <typename... Ts>
auto sum_fold(Ts... values) {
  return (values + ...);  // 一元右折叠：v0 + (v1 + (v2 + ...))
}

template <typename... Ts>
auto product_fold(Ts... values) {
  return (values * ...);  // 一元右折叠
}

// 带初始值的二元折叠（处理空包）
template <typename... Ts>
auto sum_with_init(Ts... values) {
  return (0 + ... + values);  // 二元左折叠：((0 + v0) + v1) + ...
}

// 逗号运算符折叠（执行副作用）
template <typename... Ts>
void print_fold(Ts const&... values) {
  auto print_one = [](auto const& v) {
    std::cout << v << " ";
    return 0;  // 为逗号折叠返回的哑值
  };
  (print_one(values), ...);  // 逗号折叠
  std::cout << std::endl;
}

// && 折叠（所有参数的逻辑与）
template <typename... Ts>
bool all_true(Ts... values) {
  return (values && ...);  // 右折叠：v0 && (v1 && (v2 && ...))
}

// << 折叠（流插入）
template <typename... Ts>
void print_fold_stream(Ts const&... values) {
  ((std::cout << values << " "), ...);
  std::cout << std::endl;
}

// =============================================================================
// 4. 使用 std::common_type 的变参函数
// =============================================================================
// 计算异构类型的 max（转换为公共类型）。

// 基例：单个元素
template <typename T>
T max_variadic_common(T single) {
  return single;
}

template <typename T, typename... Ts>
auto max_variadic_common(T first, Ts... rest) {
  auto tail = max_variadic_common(rest...);
  using CommonT = std::common_type_t<T, decltype(tail)>;
  return (static_cast<CommonT>(first) < static_cast<CommonT>(tail)) ? tail
                                                                      : first;
}

// =============================================================================
// 5. 变参构造：从参数包构建 std::array
// =============================================================================

template <typename T, typename... Ts>
auto make_array(Ts... values) -> std::array<T, sizeof...(Ts)> {
  return {static_cast<T>(values)...};
}

// =============================================================================
// 6. 变参索引序列
// =============================================================================
// 使用 std::index_sequence 按顺序迭代参数包。

template <typename Tuple, std::size_t... Is>
void print_tuple_impl(Tuple const& tpl, std::index_sequence<Is...>) {
  ((std::cout << (Is == 0 ? "" : ", ") << std::get<Is>(tpl)), ...);
}

template <typename... Ts>
void print_tuple(std::tuple<Ts...> const& tpl) {
  std::cout << "(";
  print_tuple_impl(tpl, std::index_sequence_for<Ts...>{});
  std::cout << ")" << std::endl;
}

// =============================================================================
// 7. 变参 All-Of / Any-Of 类型谓词
// =============================================================================

template <template <typename> class Predicate, typename... Ts>
struct AllOf : std::conjunction<Predicate<Ts>...> {};

template <template <typename> class Predicate, typename... Ts>
struct AnyOf : std::disjunction<Predicate<Ts>...> {};

template <template <typename> class Predicate, typename... Ts>
inline constexpr bool all_of_v = AllOf<Predicate, Ts...>::value;

template <template <typename> class Predicate, typename... Ts>
inline constexpr bool any_of_v = AnyOf<Predicate, Ts...>::value;

// 示例谓词
template <typename T>
using is_integral_pred = std::is_integral<T>;

// =============================================================================
// 8. 变参成对操作
// =============================================================================
// 对来自两个参数包的实参对应用二元操作。

template <typename F, typename T1, typename T2>
auto pairwise_apply(F f, T1 const& a, T2 const& b) {
  return f(a, b);
}

// 辅助：对第一对应用并递归（if constexpr 更简单）
template <typename F, typename T1, typename... Rest>
auto pairwise_apply_multi(F f, T1 const& a, T1 const& b, Rest const&... rest) {
  auto head = f(a, b);
  if constexpr (sizeof...(rest) == 0) {
    return head;
  } else {
    auto tail = pairwise_apply_multi(f, rest...);
    return std::make_pair(head, tail);
  }
}

// =============================================================================
// 9. 基于索引的参数包访问
// =============================================================================
// 访问参数包的第 N 个元素。

template <std::size_t N, typename T, typename... Ts>
struct NthType {
  static_assert(N < 1 + sizeof...(Ts), "索引越界");
  using type = typename NthType<N - 1, Ts...>::type;
};

template <typename T, typename... Ts>
struct NthType<0, T, Ts...> {
  using type = T;
};

template <std::size_t N, typename... Ts>
using nth_type_t = typename NthType<N, Ts...>::type;

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 04.1 章：变参模板基础 ===\n" << endl;

  // --- 测试 1：sizeof... ---
  cout << "[测试 1] sizeof... 运算符：" << endl;
  cout << "  count_types<int, double, char>() = "
       << count_types<int, double, char>() << endl;
  cout << "  count_values<1, 2, 3, 4, 5>()  = "
       << count_values<1, 2, 3, 4, 5>() << endl;
  static_assert(count_types<int, double>() == 2);
  static_assert(count_values<42>() == 1);
  static_assert(count_types<>() == 0);  // 空包也 OK！

  // --- 测试 2：递归打印 ---
  cout << "\n[测试 2] 递归打印：" << endl;
  print_recursive(1, 2.5, "hello", 'X');

  // --- 测试 3：Fold expressions ---
  cout << "\n[测试 3] Fold expressions：" << endl;
  cout << "  sum_fold(1,2,3,4,5) = " << sum_fold(1, 2, 3, 4, 5) << endl;
  cout << "  product_fold(1,2,3,4) = " << product_fold(1, 2, 3, 4) << endl;
  cout << "  sum_with_init()（空）= " << sum_with_init() << endl;
  cout << "  sum_with_init(10,20) = " << sum_with_init(10, 20) << endl;

  cout << "  print_fold：";
  print_fold(1, 2.5, "abc", '!');
  cout << "  print_fold_stream：";
  print_fold_stream(1, 2.5, "abc", '!');

  cout << "  all_true(true,true,true)  = " << all_true(true, true, true)
       << endl;
  cout << "  all_true(true,false,true) = " << all_true(true, false, true)
       << endl;

  // 断言
  assert(sum_fold(1, 2, 3, 4, 5) == 15);
  assert(product_fold(1, 2, 3, 4) == 24);
  assert(sum_with_init() == 0);

  // --- 测试 4：变参 max ---
  cout << "\n[测试 4] 变参 common-type max：" << endl;
  cout << "  max_variadic_common(3, 1.5, 2) = "
       << max_variadic_common(3, 1.5, 2) << endl;
  cout << "  max_variadic_common(1.1, 2.2, 3.3) = "
       << max_variadic_common(1.1, 2.2, 3.3) << endl;

  // --- 测试 5：make_array ---
  cout << "\n[测试 5] 从参数包构建 make_array：" << endl;
  auto arr = make_array<int>(1.1, 2.2, 3.3, 4.4);
  cout << "  [";
  for (auto v : arr) cout << v << " ";
  cout << "]" << endl;
  assert(arr.size() == 4);
  assert(arr[0] == 1);

  // --- 测试 6：元组打印 ---
  cout << "\n[测试 6] 通过 index_sequence 打印元组：" << endl;
  auto tpl = make_tuple(42, 3.14, string("hello"), 'X');
  print_tuple(tpl);

  // --- 测试 7：AllOf / AnyOf ---
  cout << "\n[测试 7] AllOf / AnyOf 类型谓词：" << endl;
  cout << "  all_of_v<is_integral_pred, int, long, short> = "
       << all_of_v<is_integral_pred, int, long, short> << endl;
  cout << "  all_of_v<is_integral_pred, int, double, short> = "
       << all_of_v<is_integral_pred, int, double, short> << endl;
  cout << "  any_of_v<is_integral_pred, float, double, int> = "
       << any_of_v<is_integral_pred, float, double, int> << endl;

  static_assert(all_of_v<is_integral_pred, int, long, short>);
  static_assert(!all_of_v<is_integral_pred, int, double>);
  static_assert(any_of_v<is_integral_pred, float, double, int>);
  static_assert(!any_of_v<is_integral_pred, float, double, string>);

  // --- 测试 8：成对 zip（简化）---
  cout << "\n[测试 8] 成对应用：" << endl;
  auto add_pair = [](auto a, auto b) { return a + b; };
  auto res = pairwise_apply(add_pair, 1, 5);  // 简单的 2 参数情况
  cout << "  pairwise_apply(add, 1, 5) = " << res << endl;
  assert(res == 6);
  auto res2 = pairwise_apply_multi(add_pair, 1, 10, 2, 20);
  cout << "  pairwise_apply_multi 正常工作" << endl;

  // --- 测试 9：基于索引的访问 ---
  cout << "\n[测试 9] 基于索引的参数包访问：" << endl;
  using T0 = nth_type_t<0, int, double, char>;
  using T1 = nth_type_t<1, int, double, char>;
  using T2 = nth_type_t<2, int, double, char>;

  static_assert(is_same_v<T0, int>);
  static_assert(is_same_v<T1, double>);
  static_assert(is_same_v<T2, char>);
  cout << "  pack[0] = int, pack[1] = double, pack[2] = char：已确认"
       << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
