// =============================================================================
// 第 01.1 章：基础函数模板
// 内容：模板实参推导、重载决议、
//       字符数组特化、隐式/显式实例化
// 编译：g++ -std=c++20 -o 01_basic_function_template 01_basic_function_template.cpp
// =============================================================================

#include <cassert>
#include <complex>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

// -----------------------------------------------------------------------------
// 1. 最简单的函数模板：两个值取 max（按值传递）
// -----------------------------------------------------------------------------
// 模板实参推导：编译器根据 a 和 b 的类型推导出 T。
// 对于 max(3, 5) -> T = int；对于 max(3.14, 2.71) -> T = double。
// 要求：T 必须支持 operator<。

template <typename T>
inline T max_by_val(T a, T b) {
  // 注意：按值返回可避免处理临时对象时的悬垂引用。
  // 但对于拷贝开销大的类型，这种方式并不理想。
  return (a < b) ? b : a;
}

// -----------------------------------------------------------------------------
// 2. 使用两个不同模板参数的 max
// -----------------------------------------------------------------------------
// 当两个实参类型不同时，单参数版 max_by_val 无法推导出公共类型 T。
// 这个重载版本可处理异构类型。
// 技巧：使用三个模板参数——T1、T2 和一个公共返回类型。
// C++11+：可使用 auto + decltype 或后置返回类型。

template <typename T1, typename T2>
inline auto max_hetero(T1 a, T2 b) -> decltype(a < b ? b : a) {
  return (a < b) ? b : a;
}

// -----------------------------------------------------------------------------
// 3. 按引用返回的 max（潜在陷阱）
// -----------------------------------------------------------------------------
// 返回 T const& 可避免拷贝，但对临时对象使用时很危险。
// 仅在两个实参都保证是左值时使用。

template <typename T>
inline T const& max_by_ref(T const& a, T const& b) {
  return (a < b) ? b : a;
}

// -----------------------------------------------------------------------------
// 4. C 风格字符串（char 数组 / const char*）的重载
// -----------------------------------------------------------------------------
// 通用的 max_by_val 会比较指针地址而非字符串内容。
// 这个重载通过 std::strcmp 提供字典序比较。
// C++11+：推荐使用 std::string_view，但这里展示经典做法。

inline char const* max_by_val(char const* a, char const* b) {
  return (std::strcmp(a, b) < 0) ? b : a;
}

// -----------------------------------------------------------------------------
// 5. 使用递归和 fold expression（C++17+）的三参数 max
// -----------------------------------------------------------------------------

// 递归版变参 max（fold expression 之前的风格）
template <typename T>
T max_variadic(T single) {
  return single;
}

template <typename T, typename... Ts>
auto max_variadic(T first, Ts... rest) {
  auto tail_max = max_variadic(rest...);
  return (first < tail_max) ? tail_max : first;
}

// C++17 fold expression，使用带初始值的二元折叠
template <typename T, typename... Ts>
auto max_fold_binary(T first, Ts... values) {
  auto pick = [](auto const& a, auto const& b) {
    return (a < b) ? b : a;
  };
  return (pick(first, values), ...);  // 二元左折叠
}

// -----------------------------------------------------------------------------
// 6. std::complex 的重载——默认无排序，所以我们比较模长
// -----------------------------------------------------------------------------

template <typename T>
inline bool operator<(std::complex<T> const& a,
                       std::complex<T> const& b) {
  // 比较模长的平方以避免 sqrt 开销
  auto mag_sq_a = a.real() * a.real() + a.imag() * a.imag();
  auto mag_sq_b = b.real() * b.real() + b.imag() * b.imag();
  return mag_sq_a < mag_sq_b;
}

// -----------------------------------------------------------------------------
// 7. 使用显式模板实参的 max（显式实例化）
// -----------------------------------------------------------------------------
// 有时因类型不匹配导致推导失败。调用者可以显式指定 T：
// max_by_val<double>(3, 4.5)。这会将 int(3) 转换为 double(3.0)。

// -----------------------------------------------------------------------------
// 8. 函数模板的默认模板实参（C++20：仅限类...
//    但我们可以用 constrained auto 作为变通方案）
//    C++20 中，concepts 可以优雅地约束推导出的类型。

template <typename T>
concept Comparable = requires(T a, T b) {
  { a < b } -> std::convertible_to<bool>;
};

template <Comparable T>
inline T max_constrained(T a, T b) {
  return (a < b) ? b : a;
}

// -----------------------------------------------------------------------------
// 9. 演示：使用后置返回类型和 decltype 的函数模板
//    不同类型值的 max -> 公共类型。
// -----------------------------------------------------------------------------

template <typename T1, typename T2>
inline auto max_common(T1 const& a, T2 const& b)
    -> std::common_type_t<T1, T2> {
  return (a < b) ? b : a;
}

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 01.1 章：基础函数模板 ===\n" << endl;

  // --- 测试 1：基本 max_by_val ---
  cout << "max_by_val(10, 20) = " << max_by_val(10, 20) << endl;
  cout << "max_by_val(3.14, 2.71) = " << max_by_val(3.14, 2.71) << endl;
  cout << "max_by_val('x', 'a') = " << max_by_val('x', 'a') << endl;

  // --- 测试 2：异构 max ---
  cout << "max_hetero(42, 3.14) = " << max_hetero(42, 3.14) << endl;
  cout << "max_hetero(2.71f, 100) = " << max_hetero(2.71f, 100) << endl;

  // --- 测试 3：max_by_ref（仅左值）---
  int a_int = 42, b_int = 7;
  int const& r = max_by_ref(a_int, b_int);
  cout << "max_by_ref(a_int, b_int) = " << r
       << " (a_int=" << a_int << ", b_int=" << b_int << ")" << endl;
  // 警告：max_by_ref(42, 7) 会绑定临时对象，未定义行为！

  // --- 测试 4：C 字符串重载 ---
  cout << "max_by_val(\"apple\", \"zebra\") = "
       << max_by_val("apple", "zebra") << endl;

  // --- 测试 5：变参 max ---
  cout << "max_variadic(1, 5, 3, 9, 2) = "
       << max_variadic(1, 5, 3, 9, 2) << endl;
  cout << "max_fold_binary(1.1, 2.2, 3.3, 0.5) = "
       << max_fold_binary(1.1, 2.2, 3.3, 0.5) << endl;

  // --- 测试 6：复数 ---
  complex<double> c1(3.0, 4.0), c2(1.0, 2.0);  // c1 模=5, c2 模=sqrt(5)~2.236
  cout << "max_by_val(c1=(3,4), c2=(1,2)) = "
       << max_by_val(c1, c2) << endl;

  // --- 测试 7：显式实例化 ---
  cout << "max_by_val<double>(3, 4.5) = "
       << max_by_val<double>(3, 4.5) << endl;

  // --- 测试 8：约束版 max ---
  cout << "max_constrained(100, 200) = "
       << max_constrained(100, 200) << endl;

  // --- 测试 9：common_type（注意有符号/无符号的陷阱！）---
  cout << "max_common(42, 3.14159) = "
       << max_common(42, 3.14159) << endl;

  // --- 断言 ---
  assert(max_by_val(10, 20) == 20);
  assert(max_by_val(3.14, 2.71) == 3.14);
  assert(max_hetero(42, 3.14) > 3.14);
  assert(max_fold_binary(1, 2, 3, 4, 5) == 5);
  assert(max_constrained(100, 200) == 200);

  cout << "\n所有断言通过！" << endl;
  return 0;
}
