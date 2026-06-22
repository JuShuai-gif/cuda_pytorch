// =============================================================================
// 第 08.1 章：constexpr 基础 -- 编译期函数
//
// constexpr（C++11）及其演进（C++14 放宽、C++17 lambda、C++20
// 动态分配）允许函数在编译期求值。
//
// 主题：
//   1. constexpr vs const：编译期求值 vs 运行期不可变性
//   2. constexpr 函数（C++14 前为纯函数，无副作用）
//   3. constexpr 变量（保证编译期求值）
//   4. C++14 放宽的 constexpr（多个 return 语句、循环）
//   5. C++17 constexpr lambda
//   6. C++20 constexpr std::vector 和 std::string
//   7. 模板上下文中的 constexpr
//   8. 使用 static_assert 进行编译期断言
//
// 编译：g++ -std=c++20 -o 01_constexpr_basics 01_constexpr_basics.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. constexpr vs const
// =============================================================================
// const：值在运行期不可变；可能也可能不是编译期的。
// constexpr：值 MUST 可在编译期计算（C++11），或者 CAN
//            在编译期求值（C++14+）。

const     int runtime_const = []() { return 42; }();  // 运行期，但不可变
constexpr int compile_const  = 42;  // MUST 是编译期

// =============================================================================
// 2. C++11 constexpr 函数（受限：单个 return 语句）
// =============================================================================

constexpr int sqaure_c11(int x) {
  return x * x;  // 单个 return：有效的 C++11 constexpr
}

constexpr int factorial_c11(int n) {
  return (n <= 1) ? 1 : n * factorial_c11(n - 1);  // 递归 OK
}

// =============================================================================
// 3. C++14 放宽的 constexpr（循环、多个语句、局部变量）
// =============================================================================

constexpr int factorial_c14(int n) {
  int result = 1;
  for (int i = 2; i <= n; ++i) {
    result *= i;
  }
  return result;
}

constexpr bool is_prime(int n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if (n % 2 == 0) return false;
  for (int i = 3; i * i <= n; i += 2) {
    if (n % i == 0) return false;
  }
  return true;
}

// =============================================================================
// 4. constexpr 变量（保证编译期）
// =============================================================================
// constexpr 变量在编译期求值。如果初始化器不能在编译期求值，
// 会产生编译错误。

constexpr int fib10 = []() constexpr {
  int a = 0, b = 1;
  for (int i = 0; i < 10; ++i) {
    int tmp = a + b;
    a = b;
    b = tmp;
  }
  return a;
}();  // C++17 constexpr lambda

// =============================================================================
// 5. constexpr 与模板参数
// =============================================================================

template <int N>
struct CompileTimeArray {
  std::array<int, N> data{};

  constexpr CompileTimeArray() {
    for (int i = 0; i < N; ++i) {
      data[i] = i * i;
    }
  }

  constexpr int operator[](int i) const { return data[i]; }
  constexpr int size() const { return N; }
};

// =============================================================================
// 6. constexpr 与自定义类型
// =============================================================================

struct Point2D_CE {
  double x, y;

  constexpr Point2D_CE(double x_, double y_) : x(x_), y(y_) {}

  constexpr double distance_sq() const {
    return x * x + y * y;
  }

  constexpr Point2D_CE operator+(Point2D_CE const& other) const {
    return Point2D_CE(x + other.x, y + other.y);
  }
};

// 验证 constexpr 构造
constexpr Point2D_CE p1(3.0, 4.0);
constexpr Point2D_CE p2(1.0, 2.0);
constexpr Point2D_CE p3 = p1 + p2;
static_assert(p1.distance_sq() == 25.0, "3^2 + 4^2 = 25");

// =============================================================================
// 7. C++17 constexpr Lambda
// =============================================================================

constexpr auto square_lambda = [](int x) constexpr { return x * x; };
static_assert(square_lambda(5) == 25);

// 捕获 constexpr 值的 Lambda（C++17 要求 [=] 或按值捕获）
constexpr int base_val = 10;
constexpr auto add_base = [](int x) constexpr { return base_val + x; };
static_assert(add_base(5) == 15);

// =============================================================================
// 8. constexpr 与 std::array（C++14+）及手动编译期数组求和
// =============================================================================
// 注意：constexpr std::vector 需要 C++20 且需要完全符合标准的编译器。
// GCC 11 尚不支持此功能，所以这里使用 std::array。

constexpr int sum_array_cxx14() {
  constexpr std::array<int, 5> arr{1, 2, 3, 4, 5};
  int sum = 0;
  for (int x : arr) sum += x;
  return sum;
}

constexpr int compile_time_arr_sum = sum_array_cxx14();
static_assert(compile_time_arr_sum == 15);

// =============================================================================
// 9. 带分支的 constexpr 幂函数
// =============================================================================

constexpr long long int_pow(long long base, int exp) {
  long long result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    base *= base;
    exp >>= 1;
  }
  return result;
}

static_assert(int_pow(2, 10) == 1024);
static_assert(int_pow(3, 4) == 81);
static_assert(int_pow(5, 0) == 1);

// =============================================================================
// 10. constexpr 字符串长度和比较（C++14+）
// =============================================================================

constexpr std::size_t strlen_ce(char const* str) {
  std::size_t len = 0;
  while (str[len] != '\0') ++len;
  return len;
}

constexpr bool streq_ce(char const* a, char const* b) {
  while (*a && *b) {
    if (*a != *b) return false;
    ++a; ++b;
  }
  return *a == *b;
}

static_assert(strlen_ce("hello") == 5);
static_assert(streq_ce("abc", "abc"));
static_assert(!streq_ce("abc", "abd"));

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 08.1 章：constexpr 基础 ===\n" << endl;

  // --- 测试 1：C++11 风格 ---
  cout << "[Test 1] C++11 constexpr：" << endl;
  cout << "  sqaure_c11(7) = " << sqaure_c11(7) << endl;
  cout << "  factorial_c11(5) = " << factorial_c11(5) << endl;
  static_assert(sqaure_c11(7) == 49);
  static_assert(factorial_c11(5) == 120);

  // --- 测试 2：C++14 风格 ---
  cout << "\n[Test 2] C++14 constexpr（带循环）：" << endl;
  cout << "  factorial_c14(6) = " << factorial_c14(6) << endl;
  cout << "  is_prime(17) = " << is_prime(17) << endl;
  cout << "  is_prime(15) = " << is_prime(15) << endl;

  static_assert(factorial_c14(6) == 720);
  static_assert(is_prime(17));
  static_assert(!is_prime(15));
  static_assert(is_prime(2));
  static_assert(!is_prime(1));

  // 枚举 20 以内的素数
  cout << "  <= 20 的素数：";
  for (int i = 1; i <= 20; ++i) {
    if (is_prime(i)) cout << i << " ";
  }
  cout << endl;

  // --- 测试 3：constexpr 变量 ---
  cout << "\n[Test 3] constexpr 变量：" << endl;
  cout << "  fib10 = " << fib10 << endl;
  static_assert(fib10 == 55);  // 第 10 个斐波那契数（从 0 开始）

  // --- 测试 4：CompileTimeArray ---
  cout << "\n[Test 4] CompileTimeArray 模板：" << endl;
  constexpr CompileTimeArray<5> cta;
  cout << "  CompileTimeArray<5>：";
  for (int i = 0; i < cta.size(); ++i) cout << cta[i] << " ";
  cout << endl;
  static_assert(cta[0] == 0);
  static_assert(cta[1] == 1);
  static_assert(cta[2] == 4);
  static_assert(cta[3] == 9);
  static_assert(cta[4] == 16);

  // --- 测试 5：constexpr 自定义类型 ---
  cout << "\n[Test 5] constexpr Point2D：" << endl;
  cout << "  p1.distance_sq() = " << p1.distance_sq() << endl;
  cout << "  p3 = p1 + p2 = (" << p3.x << ", " << p3.y << ")" << endl;

  constexpr Point2D_CE p4(0, 0);
  constexpr Point2D_CE p5(6, 8);
  static_assert(p5.distance_sq() == 100.0);

  // --- 测试 6：constexpr lambda ---
  cout << "\n[Test 6] C++17 constexpr lambda：" << endl;
  constexpr int sq25 = square_lambda(25);
  cout << "  square_lambda(25) = " << sq25 << endl;
  cout << "  add_base(15) = " << add_base(15) << endl;
  static_assert(sq25 == 625);

  // Lambda 也可在运行期使用
  auto sq_runtime = [](int x) constexpr { return x * x; };
  cout << "  运行期 lambda：sq_runtime(11) = " << sq_runtime(11) << endl;

  // --- 测试 7：constexpr 数组 ---
  cout << "\n[Test 7] constexpr 数组求和：" << endl;
  cout << "  compile_time_arr_sum = " << compile_time_arr_sum << endl;
  static_assert(compile_time_arr_sum == 15);

  // --- 测试 8：int_pow ---
  cout << "\n[Test 8] constexpr int_pow：" << endl;
  cout << "  int_pow(2, 8) = " << int_pow(2, 8) << endl;
  cout << "  int_pow(3, 5) = " << int_pow(3, 5) << endl;
  cout << "  int_pow(10, 0) = " << int_pow(10, 0) << endl;
  static_assert(int_pow(2, 8) == 256);
  static_assert(int_pow(3, 5) == 243);

  // --- 测试 9：constexpr 字符串操作 ---
  cout << "\n[Test 9] constexpr 字符串操作：" << endl;
  constexpr auto len = strlen_ce("constexpr");
  cout << "  strlen_ce(\"constexpr\") = " << len << endl;
  cout << "  streq_ce(\"hello\", \"hello\") = " << streq_ce("hello", "hello")
       << endl;
  cout << "  streq_ce(\"hello\", \"world\") = " << streq_ce("hello", "world")
       << endl;
  static_assert(len == 9);

  // --- 测试 10：在运行期位置调用 constexpr ---
  cout << "\n[Test 10] 在运行期调用 constexpr 函数：" << endl;
  int n = 10;
  cout << "  factorial_c14(n=10) = " << factorial_c14(n) << endl;
  assert(factorial_c14(n) == 3628800);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
