// =============================================================================
// 第 06.1 章：enable_if 基础 -- 条件函数重载
//
// std::enable_if（C++11）和 std::enable_if_t（C++14）基于编译期布尔条件
// 启用或禁用模板实例化。这是 SFINAE（替换失败不是错误）的基础。
//
// 主题：
//   1. 基本 enable_if：有条件地启用函数模板
//   2. 返回类型位置的 enable_if
//   3. 模板参数位置的 enable_if
//   4. 函数参数位置的 enable_if（哑参数）
//   5. 作为 enable_if 替代方案的标签分发
//   6. C++20 requires 子句作为现代替代方案
//   7. 互斥 enable_if 的多重载
//   8. 避免歧义重载
//
// 编译：g++ -std=c++20 -o 01_basic_enable_if 01_basic_enable_if.cpp
// =============================================================================

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. 从头实现 enable_if
// =============================================================================
// 在使用 std::enable_if 之前理解其机制。

template <bool Cond, typename T = void>
struct EnableIf {};

template <typename T>
struct EnableIf<true, T> {
  using type = T;
};

template <bool Cond, typename T = void>
using enable_if_t = typename EnableIf<Cond, T>::type;

// =============================================================================
// 2. 返回类型位置的 enable_if
// =============================================================================
// 最易读的形式：将返回类型包裹在 enable_if 中。

// 适用于整数类型的版本
template <typename T>
enable_if_t<std::is_integral_v<T>, T> half_value(T val) {
  return val / 2;
}

// 适用于浮点类型的版本
template <typename T>
enable_if_t<std::is_floating_point_v<T>, T> half_value(T val) {
  return val / T(2);
}

// =============================================================================
// 3. 模板参数位置的 enable_if
// =============================================================================
// 当返回类型依赖其他内容时，或函数没有返回类型（构造函数）时很有用。

template <typename T,
          typename = enable_if_t<std::is_integral_v<T>>>
struct IntegerOnly {
  T val;

  explicit IntegerOnly(T v) : val(v) {}

  T get() const { return val; }
  bool is_odd() const { return val % 2 != 0; }
};

// =============================================================================
// 4. 函数参数位置的 enable_if（哑参数）
// =============================================================================
// 避免污染模板参数列表。使用带默认值的哑指针参数。

template <typename T>
T double_value(T val,
                enable_if_t<std::is_arithmetic_v<T>, int> = 0) {
  return val * T(2);
}

// std::string 的重载（连接）
template <typename T>
T double_value(T val,
                enable_if_t<std::is_same_v<T, std::string>, char> = 0) {
  return val + val;
}

// =============================================================================
// 5. 标签分发（enable_if 的替代方案）
// =============================================================================
// 当条件基于类型类别（而非布尔值）时，标签分发更清晰。
// 使用辅助标签类型并对其重载。

struct integral_tag {};
struct floating_tag {};
struct other_tag {};

template <typename T>
constexpr auto get_tag() {
  if constexpr (std::is_integral_v<T>) return integral_tag{};
  else if constexpr (std::is_floating_point_v<T>) return floating_tag{};
  else return other_tag{};
}

template <typename T>
T process_impl(T val, integral_tag) {
  std::cout << "  [整数] " << val << " -> " << (val + 1) << std::endl;
  return val + 1;
}

template <typename T>
T process_impl(T val, floating_tag) {
  std::cout << "  [浮点] " << val << " -> " << std::sqrt(val) << std::endl;
  return static_cast<T>(std::sqrt(val));
}

template <typename T>
T process_impl(T val, other_tag) {
  std::cout << "  [其他] 无操作" << std::endl;
  return val;
}

template <typename T>
T process(T val) {
  return process_impl(val, get_tag<T>());
}

// =============================================================================
// 6. C++20 requires 子句（现代替代方案）
// =============================================================================

template <typename T>
requires std::is_integral_v<T>
T triple_value(T val) {
  return val * T(3);
}

template <typename T>
requires std::is_floating_point_v<T>
T triple_value(T val) {
  return val * T(3.0);
}

// 替代语法：后置 requires
template <typename T>
T quad_value(T val) requires std::is_arithmetic_v<T> {
  return val * T(4);
}

// =============================================================================
// 7. 多个互斥重载
// =============================================================================
// 仔细设计 enable_if 条件以避免歧义。

template <typename T>
enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, std::string>
classify(T) {
  return "整数（非 bool）";
}

template <typename T>
enable_if_t<std::is_floating_point_v<T>, std::string>
classify(T) {
  return "浮点数";
}

template <typename T>
enable_if_t<std::is_same_v<T, bool>, std::string>
classify(T) {
  return "布尔值";
}

template <typename T>
enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>,
             std::string>
classify(T) {
  return "其他类型";
}

// =============================================================================
// 8. 构造函数的 enable_if（SFINAE 排除不需要的重载）
// =============================================================================

template <typename T>
class Wrapper {
 public:
  // 默认构造函数始终可用
  Wrapper() : val_(T{}) {}

  // 从值构造
  explicit Wrapper(T val) : val_(val) {}

  // 从指针构造（显式解引用版本）
  // 我们使用不同的名称以避免构造函数重载歧义
  static Wrapper from_pointer(T ptr) requires std::is_pointer_v<T> {
    Wrapper w;
    w.val_ = ptr;
    std::cout << "  Wrapper<指针>：已存储地址 " << w.val_ << std::endl;
    return w;
  }

  T get() const { return val_; }

 private:
  T val_;
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 06.1 章：enable_if 基础 ===\n" << endl;

  // --- 测试 1：返回类型 enable_if ---
  cout << "[测试 1] 返回类型中的 enable_if：" << endl;
  cout << "  half_value(10)   = " << half_value(10) << endl;
  cout << "  half_value(3.14) = " << half_value(3.14) << endl;
  assert(half_value(10) == 5);
  assert(half_value(3.14) == 1.57);

  // --- 测试 2：模板参数 enable_if ---
  cout << "\n[测试 2] 模板参数中的 enable_if：" << endl;
  IntegerOnly<int> io(7);
  cout << "  IntegerOnly<int>(7).is_odd() = " << io.is_odd() << endl;
  assert(io.is_odd());

  // IntegerOnly<double> io_d(3.14);  // 错误：被 SFINAE 排除

  // --- 测试 3：函数参数 enable_if ---
  cout << "\n[测试 3] 函数参数中的 enable_if：" << endl;
  cout << "  double_value(5)     = " << double_value(5) << endl;
  cout << "  double_value(3.14)  = " << double_value(3.14) << endl;
  cout << "  double_value(string(\"hi\")) = "
       << double_value(string("hi")) << endl;
  assert(double_value(5) == 10);
  assert(double_value(string("hi")) == "hihi");

  // --- 测试 4：标签分发 ---
  cout << "\n[测试 4] 标签分发：" << endl;
  cout << "  process(42)：    " << process(42) << endl;
  cout << "  process(16.0)：  " << process(16.0) << endl;
  cout << "  process(\"abc\")：" << process("abc") << endl;
  assert(process(42) == 43);
  assert(process(16.0) == 4.0);

  // --- 测试 5：C++20 requires ---
  cout << "\n[测试 5] C++20 requires 子句：" << endl;
  cout << "  triple_value(10)  = " << triple_value(10) << endl;
  cout << "  triple_value(1.5) = " << triple_value(1.5) << endl;
  cout << "  quad_value(7)     = " << quad_value(7) << endl;
  assert(triple_value(10) == 30);
  assert(quad_value(7) == 28);

  // --- 测试 6：classify 重载 ---
  cout << "\n[测试 6] classify（互斥重载）：" << endl;
  cout << "  classify(42)    -> " << classify(42) << endl;
  cout << "  classify(3.14)  -> " << classify(3.14) << endl;
  cout << "  classify(true)  -> " << classify(true) << endl;
  cout << "  classify(\"x\")   -> " << classify("x") << endl;

  assert(classify(42) == "整数（非 bool）");
  assert(classify(3.14) == "浮点数");
  assert(classify(true) == "布尔值");
  assert(classify("x") == "其他类型");

  // --- 测试 7：带 enable_if 构造函数的 Wrapper ---
  cout << "\n[测试 7] enable_if 构造函数：" << endl;
  Wrapper<int> wi(42);
  cout << "  Wrapper<int>(42).get() = " << wi.get() << endl;
  assert(wi.get() == 42);

  int val = 99;
  auto wp = Wrapper<int*>::from_pointer(&val);
  cout << "  Wrapper<int*>::from_pointer().get() = " << wp.get() << endl;
  assert(*wp.get() == 99);

  // --- 测试 8：SFINAE 安全性 ---
  // 如果没有 enable_if 保护，以下将失败：
  // half_value("hello") 会尝试 string/2 -- 但 SFINAE 移除了重载
  // 编译成功，因为没有可行的重载存在。

  cout << "\n所有测试通过！" << endl;
  return 0;
}
