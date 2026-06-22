// =============================================================================
// 第 08.3 章：模板元编程 -- 阶乘、斐波那契、类型计算
//
// 模板元编程（TMP）使用模板实例化作为编译期计算机制。
// 值被编码为类型（例如 integral_constant），
// 计算通过递归模板实例化进行。
//
// 本文实现经典的 TMP 示例：
//   1. 编译期阶乘（递归模板）
//   2. 编译期斐波那契（递归模板）
//   3. 编译期 GCD 和 LCM
//   4. 编译期幂运算
//   5. 类型列表 map/filter/fold（编译期算法）
//   6. 编译期整数序列（index_sequence）
//   7. TMP vs constexpr：权衡
//   8. 高阶元函数（元函数组合）
//
// 编译：g++ -std=c++20 -o 03_tmp_factorial 03_tmp_factorial.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. TMP 阶乘（经典递归模板）
// =============================================================================
// 原始的 TMP 示例。Factorial<N>::value 在编译期计算 N!。
// 每次实例化都是一个不同的类型，消耗编译器内存。

template <int N>
struct Factorial {
  // N! = N * (N-1)!
  static constexpr int value = N * Factorial<N - 1>::value;
};

// 基础情况：0! = 1
template <>
struct Factorial<0> {
  static constexpr int value = 1;
};

// 便捷变量模板
template <int N>
inline constexpr int factorial_v = Factorial<N>::value;

// =============================================================================
// 2. TMP 斐波那契
// =============================================================================

template <int N>
struct Fibonacci {
  // F(N) = F(N-1) + F(N-2)
  static constexpr int value =
      Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template <>
struct Fibonacci<0> {
  static constexpr int value = 0;
};

template <>
struct Fibonacci<1> {
  static constexpr int value = 1;
};

template <int N>
inline constexpr int fibonacci_v = Fibonacci<N>::value;

// =============================================================================
// 3. TMP GCD（编译期欧几里得算法）
// =============================================================================

template <int A, int B>
struct Gcd {
  static constexpr int value = Gcd<B, A % B>::value;
};

template <int A>
struct Gcd<A, 0> {
  static constexpr int value = A;
};

template <int A, int B>
inline constexpr int gcd_v = Gcd<A, B>::value;

// LCM 通过 GCD：lcm(a,b) = a * b / gcd(a,b)
template <int A, int B>
struct Lcm {
  static constexpr int value = (A * B) / Gcd<A, B>::value;
};

template <int A, int B>
inline constexpr int lcm_v = Lcm<A, B>::value;

// =============================================================================
// 4. TMP 幂运算
// =============================================================================

template <int Base, int Exp>
struct Power {
  // pow = Base * Base^(Exp-1)
  static constexpr long long value = Base * Power<Base, Exp - 1>::value;
};

template <int Base>
struct Power<Base, 0> {
  static constexpr long long value = 1;
};

template <int Base, int Exp>
inline constexpr long long power_v = Power<Base, Exp>::value;

// =============================================================================
// 5. TMP 整数序列（std::integer_sequence）
// =============================================================================
// 重新实现以展示 TMP 如何构建序列。

template <int... Is>
struct IntSequence {
  static constexpr std::size_t size = sizeof...(Is);
};

// 生成 IntSequence<0, 1, ..., N-1>
template <int N, int... Is>
struct MakeIntSequenceImpl : MakeIntSequenceImpl<N - 1, N - 1, Is...> {};

template <int... Is>
struct MakeIntSequenceImpl<0, Is...> {
  using type = IntSequence<Is...>;
};

template <int N>
using make_int_sequence = typename MakeIntSequenceImpl<N>::type;

// =============================================================================
// 6. TMP 类型级别布尔逻辑
// =============================================================================

// and_v：所有参数都为 true 时结果才为 true
template <bool... Bs>
struct And : std::true_type {};

template <bool... Bs>
struct And<false, Bs...> : std::false_type {};

template <bool... Bs>
struct And<true, Bs...> : And<Bs...> {};

template <bool... Bs>
inline constexpr bool and_v = And<Bs...>::value;

// or_v：任一参数为 true 时结果为 true
template <bool... Bs>
struct Or : std::false_type {};

template <bool... Bs>
struct Or<true, Bs...> : std::true_type {};

template <bool... Bs>
struct Or<false, Bs...> : Or<Bs...> {};

template <bool... Bs>
inline constexpr bool or_v = Or<Bs...>::value;

// not_v
template <bool B>
using Not = std::bool_constant<!B>;

template <bool B>
inline constexpr bool not_v = Not<B>::value;

// =============================================================================
// 7. TMP 类型级别条件选择
// =============================================================================

// if_t：条件类型选择
template <bool Cond, typename T, typename F>
struct If {
  using type = T;
};

template <typename T, typename F>
struct If<false, T, F> {
  using type = F;
};

template <bool Cond, typename T, typename F>
using if_t = typename If<Cond, T, F>::type;

// =============================================================================
// 8. Type Map：对参数包的每个元素应用元函数
// =============================================================================

// std::add_pointer 的元函数包装器
template <typename T>
using AddPointerMF = std::add_pointer<T>;

// TypeMap：接受一个 template template 参数和一个参数包
template <template <typename> class F, typename... Ts>
struct TypeMap {
  using type = std::tuple<typename F<Ts>::type...>;
};

template <template <typename> class F, typename... Ts>
using type_map_t = typename TypeMap<F, Ts...>::type;

// =============================================================================
// 9. TMP Accumulate（编译期对类型进行 Fold）
// =============================================================================

// SizeOfAccumulator：对所有 Ts 的大小求和
template <typename... Ts>
struct SizeOfAccumulator {
  static constexpr std::size_t value = (sizeof(Ts) + ...);  // fold（C++17）
};

// =============================================================================
// 10. TMP vs constexpr 对比
// =============================================================================
// 两者都能实现编译期计算，但 constexpr 更具可读性。

// TMP 版本（基于类型的递归）
template <int N>
struct SquareTMP {
  static constexpr int value = N * N;
};

// constexpr 版本（基于值）
constexpr int square_constexpr(int n) {
  return n * n;
}

// TMP：元函数组合
template <int N>
struct IncrAndDouble {
  static constexpr int value = SquareTMP<N + 1>::value * 2;
};

// constexpr：函数组合
constexpr int incr_and_double_ce(int n) {
  return square_constexpr(n + 1) * 2;
}

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 08.3 章：TMP -- 阶乘、斐波那契、类型计算 ===\n"
       << endl;

  // --- 测试 1：阶乘 ---
  cout << "[Test 1] TMP 阶乘：" << endl;
  cout << "  Factorial<0>::value = " << Factorial<0>::value << endl;
  cout << "  Factorial<5>::value = " << Factorial<5>::value << endl;
  cout << "  Factorial<10>::value = " << Factorial<10>::value << endl;
  cout << "  factorial_v<7> = " << factorial_v<7> << endl;

  static_assert(Factorial<0>::value == 1);
  static_assert(Factorial<1>::value == 1);
  static_assert(Factorial<5>::value == 120);
  static_assert(factorial_v<7> == 5040);

  // --- 测试 2：斐波那契 ---
  cout << "\n[Test 2] TMP 斐波那契：" << endl;
  cout << "  Fibonacci<0>::value = " << Fibonacci<0>::value << endl;
  cout << "  Fibonacci<1>::value = " << Fibonacci<1>::value << endl;
  cout << "  Fibonacci<5>::value = " << Fibonacci<5>::value << endl;
  cout << "  Fibonacci<10>::value = " << Fibonacci<10>::value << endl;

  static_assert(Fibonacci<0>::value == 0);
  static_assert(Fibonacci<1>::value == 1);
  static_assert(Fibonacci<10>::value == 55);
  static_assert(fibonacci_v<20> == 6765);

  // --- 测试 3：GCD 和 LCM ---
  cout << "\n[Test 3] TMP GCD 和 LCM：" << endl;
  cout << "  gcd<12, 18> = " << gcd_v<12, 18> << endl;
  cout << "  gcd<100, 35> = " << gcd_v<100, 35> << endl;
  cout << "  lcm<12, 18> = " << lcm_v<12, 18> << endl;
  cout << "  lcm<7, 13> = " << lcm_v<7, 13> << endl;

  static_assert(gcd_v<12, 18> == 6);
  static_assert(gcd_v<100, 35> == 5);
  static_assert(lcm_v<12, 18> == 36);
  static_assert(lcm_v<7, 13> == 91);

  // --- 测试 4：幂运算 ---
  cout << "\n[Test 4] TMP 幂运算：" << endl;
  cout << "  power<2, 10> = " << power_v<2, 10> << endl;
  cout << "  power<3, 4> = " << power_v<3, 4> << endl;
  cout << "  power<5, 0> = " << power_v<5, 0> << endl;

  static_assert(power_v<2, 10> == 1024);
  static_assert(power_v<3, 4> == 81);
  static_assert(power_v<5, 0> == 1);

  // --- 测试 5：IntSequence ---
  cout << "\n[Test 5] TMP IntSequence：" << endl;
  using Seq5 = make_int_sequence<5>;
  cout << "  make_int_sequence<5>::size = " << Seq5::size << endl;
  static_assert(Seq5::size == 5);
  static_assert(std::is_same_v<Seq5, IntSequence<0, 1, 2, 3, 4>>);

  // --- 测试 6：TMP 布尔逻辑 ---
  cout << "\n[Test 6] TMP 布尔逻辑：" << endl;
  static_assert(and_v<true, true, true>);
  static_assert(!and_v<true, false, true>);
  static_assert(or_v<false, false, true>);
  static_assert(!or_v<false, false, false>);
  static_assert(not_v<true> == false);
  static_assert(not_v<false> == true);
  cout << "  and<true,true,true> = " << and_v<true, true, true> << endl;
  cout << "  and<true,false,true> = " << and_v<true, false, true> << endl;
  cout << "  or<false,false,true> = " << or_v<false, false, true> << endl;

  // --- 测试 7：类型级别条件选择 ---
  cout << "\n[Test 7] TMP 条件类型选择：" << endl;
  static_assert(std::is_same_v<if_t<true, int, double>, int>);
  static_assert(std::is_same_v<if_t<false, int, double>, double>);
  static_assert(std::is_same_v<if_t<(sizeof(int) > sizeof(char)), long, short>,
                               long>);
  cout << "  if<true, int, double> = int：确认" << endl;
  cout << "  if<false, int, double> = double：确认" << endl;

  // --- 测试 8：TypeMap ---
  cout << "\n[Test 8] TMP TypeMap（对参数包应用元函数）：" << endl;
  using Mapped = type_map_t<AddPointerMF, int, double, char>;
  static_assert(std::is_same_v<Mapped, std::tuple<int*, double*, char*>>);
  cout << "  TypeMap<AddPointer, int, double, char> = "
       << "std::tuple<int*, double*, char*>：确认" << endl;

  // --- 测试 9：Accumulate ---
  cout << "\n[Test 9] TMP 大小累加：" << endl;
  cout << "  sizeof(int) + sizeof(double) + sizeof(char) = "
       << SizeOfAccumulator<int, double, char>::value << endl;
  static_assert(SizeOfAccumulator<int, double, char>::value ==
                sizeof(int) + sizeof(double) + sizeof(char));

  // --- 测试 10：TMP vs constexpr ---
  cout << "\n[Test 10] TMP vs constexpr 对比：" << endl;
  static_assert(SquareTMP<5>::value == 25);       // TMP
  static_assert(square_constexpr(5) == 25);        // constexpr
  static_assert(IncrAndDouble<5>::value == 72);    // TMP 组合：((5+1)^2)*2 = 72
  static_assert(incr_and_double_ce(5) == 72);      // constexpr 组合
  cout << "  TMP 和 constexpr 都产生正确结果" << endl;
  cout << "  TMP：SquareTMP<5>=" << SquareTMP<5>::value
       << ", IncrAndDouble<5>=" << IncrAndDouble<5>::value << endl;
  cout << "  CE： square_constexpr(5)=" << square_constexpr(5)
       << ", incr_and_double_ce(5)=" << incr_and_double_ce(5) << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
