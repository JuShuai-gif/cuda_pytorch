// =============================================================================
// 第 08.2 章：constexpr if -- 编译期分支
//
// C++17 的 if constexpr 实现了在单个函数模板内的编译期条件编译，
// 在很多情况下消除了使用基于 SFINAE 的重载的需要。
// 编译器会完全丢弃未采用的分支（不仅仅是死代码消除 -- 它根本不会被实例化）。
//
// 主题：
//   1. 基本 if constexpr vs 普通 if
//   2. 类型依赖分支（is_integral、is_floating_point 等）
//   3. 编译期递归终止
//   4. 使用 constexpr if 进行完美转发
//   5. 使用 constexpr if 进行可变参数模板折叠
//   6. 用 if constexpr 替代 std::enable_if
//   7. if constexpr 与 requires 表达式（C++20）
//   8. 多层 constexpr if 链
//
// 编译：g++ -std=c++20 -o 02_constexpr_if 02_constexpr_if.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. if constexpr vs 普通 if
// =============================================================================
// 普通 if：两个分支对所有 T 都必须有效。
// if constexpr：只有采用的分支必须有效；另一个分支被丢弃。

template <typename T>
auto get_value_regular(T const& container) {
  // 错误：如果 T 没有 .size()，即使条件为 false，编译器
  // 也必须为所有 T 编译整个函数体。
  if (false) {
    return container.size();  // 仍然必须对所有 T 有效！
  }
  return 0;
}

template <typename T>
auto get_value_constexpr_if(T const& container) {
  // OK：对于没有 .size() 的类型，false 分支被完全丢弃
  if constexpr (false) {
    return container.size();  // 永远不会被实例化
  } else {
    return 0;
  }
}

// =============================================================================
// 2. 使用 if constexpr 进行类型依赖分支
// =============================================================================

template <typename T>
std::string type_category(T const&) {
  if constexpr (std::is_integral_v<T>) {
    return "integral";
  } else if constexpr (std::is_floating_point_v<T>) {
    return "floating-point";
  } else if constexpr (std::is_same_v<T, std::string>) {
    return "string";
  } else if constexpr (std::is_pointer_v<T>) {
    return "pointer";
  } else {
    return "other";
  }
}

// =============================================================================
// 3. 按类型不同处理值
// =============================================================================

template <typename T>
T process_by_type(T val) {
  if constexpr (std::is_integral_v<T>) {
    // 整数：移位、掩码、位运算
    return static_cast<T>(val * 2 + (val & 1));
  } else if constexpr (std::is_floating_point_v<T>) {
    // 浮点数：数学函数
    return static_cast<T>(std::sqrt(val * val + 1.0));
  } else if constexpr (std::is_same_v<T, std::string>) {
    // 字符串：追加
    return val + "_processed";
  } else {
    return val;  // 其他类型：无操作
  }
}

// =============================================================================
// 4. 编译期递归终止
// =============================================================================

// 使用 if constexpr 终止的可变参数递归打印
template <typename T, typename... Ts>
void print_all(T const& first, Ts const&... rest) {
  std::cout << first;
  if constexpr (sizeof...(rest) > 0) {
    std::cout << ", ";
    print_all(rest...);  // 仅在 rest 非空时实例化
  } else {
    std::cout << std::endl;
  }
}

// =============================================================================
// 5. 使用 constexpr if 进行完美转发
// =============================================================================
// 根据值类别选择拷贝或移动。

template <typename T>
auto forward_or_copy(T&& arg) {
  // if constexpr + 左值引用检测
  if constexpr (std::is_lvalue_reference_v<T>) {
    // arg 是左值：拷贝它
    using ValueT = std::remove_reference_t<T>;
    std::cout << "  Copying lvalue" << std::endl;
    return ValueT(arg);  // 拷贝
  } else {
    // arg 是右值：移动它
    std::cout << "  Moving rvalue" << std::endl;
    return std::move(arg);  // 移动
  }
}

// =============================================================================
// 6. 用 if constexpr 替代 enable_if 重载
// =============================================================================

// C++17 之前：使用 enable_if 的两个重载
template <typename T>
std::enable_if_t<std::is_integral_v<T>, T> half_enable_if(T val) {
  return val / 2;
}
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, T> half_enable_if(T val) {
  return val / T(2);
}

// C++17+：使用 if constexpr 的单个函数
template <typename T>
T half_constexpr_if(T val) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(val / 2);  // 整数除法
  } else if constexpr (std::is_floating_point_v<T>) {
    return val / T(2.0);  // 浮点除法
  } else {
    return val;  // 回退：无操作
  }
}

// =============================================================================
// 7. if constexpr 与 Concepts / requires（C++20）
// =============================================================================

template <typename T>
std::string describe(T const& obj) {
  if constexpr (requires { obj.size(); }) {
    return "Has .size() = " + std::to_string(obj.size());
  } else if constexpr (requires { std::to_string(obj); }) {
    return "Convertible to string: " + std::to_string(obj);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return "String: \"" + obj + "\"";
  } else {
    return "Unknown type";
  }
}

// =============================================================================
// 8. 多层编译期分发链
// =============================================================================
// 使用 if constexpr 模拟 CUTLASS 风格的 kernel 选择器。

enum class DataType { F32, F16, I8, I32 };
enum class AlignType { Align1, Align4, Align8 };

template <DataType DType, AlignType Align>
struct KernelLauncher {
  static void launch() {
    std::cout << "  Kernel<";

    // 对 DataType 进行编译期 switch
    if constexpr (DType == DataType::F32) {
      std::cout << "F32";
    } else if constexpr (DType == DataType::F16) {
      std::cout << "F16";
    } else if constexpr (DType == DataType::I8) {
      std::cout << "I8";
    } else if constexpr (DType == DataType::I32) {
      std::cout << "I32";
    } else {
      std::cout << "???";
    }

    std::cout << ", Align";

    // 对 Align 进行编译期 switch
    if constexpr (Align == AlignType::Align1) {
      std::cout << "1";
    } else if constexpr (Align == AlignType::Align4) {
      std::cout << "4";
    } else if constexpr (Align == AlignType::Align8) {
      std::cout << "8";
    }

    std::cout << "> launched!" << std::endl;
  }
};

// =============================================================================
// 9. if constexpr 用于静态断言
// =============================================================================
// 根据类型选择不同的 static_assert 条件。

template <typename T>
void validate_type() {
  if constexpr (std::is_integral_v<T>) {
    static_assert(sizeof(T) <= 8, "Integral type too large");
  } else if constexpr (std::is_floating_point_v<T>) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "Only float and double supported");
  }
}

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 08.2 章：constexpr if（编译期分支） ===\n"
       << endl;

  // --- 测试 1：constexpr if 基础 ---
  cout << "[Test 1] constexpr if 始终为 false 的分支：" << endl;
  cout << "  get_value_constexpr_if(vec) = "
       << get_value_constexpr_if(vector<int>{1, 2, 3}) << endl;

  // --- 测试 2：type_category ---
  cout << "\n[Test 2] type_category（类型依赖分支）：" << endl;
  cout << "  int:       " << type_category(42) << endl;
  cout << "  double:    " << type_category(3.14) << endl;
  cout << "  string:    " << type_category(string("hi")) << endl;
  cout << "  int*:      " << type_category((int*)nullptr) << endl;
  cout << "  char:      " << type_category('x') << endl;

  assert(type_category(42) == "integral");
  assert(type_category(3.14) == "floating-point");
  assert(type_category(string("hi")) == "string");

  // --- 测试 3：process_by_type ---
  cout << "\n[Test 3] process_by_type：" << endl;
  cout << "  process_by_type(10)     = " << process_by_type(10) << endl;
  cout << "  process_by_type(3.0)    = " << process_by_type(3.0) << endl;
  cout << "  process_by_type(string(\"data\")) = "
       << process_by_type(string("data")) << endl;

  assert(process_by_type(10) == 20);    // 10*2 + (10&1) = 20 + 0 = 20
  assert(process_by_type(string("data")) == "data_processed");

  // --- 测试 4：使用 if constexpr 递归的 print_all ---
  cout << "\n[Test 4] print_all（使用 if constexpr 的可变参数）：" << endl;
  cout << "  ";
  print_all(1, 2.5, "three", '!');

  // --- 测试 5：forward_or_copy ---
  cout << "\n[Test 5] forward_or_copy（完美转发选择）：" << endl;
  string s = "hello";
  auto c1 = forward_or_copy(s);          // 应该拷贝（s 是左值）
  cout << "  拷贝后：s='" << s << "', c1='" << c1 << "'" << endl;
  assert(s == "hello");
  assert(c1 == "hello");

  auto c2 = forward_or_copy(string("world"));  // 应该移动
  cout << "  移动后：c2='" << c2 << "'" << endl;
  assert(c2 == "world");

  // --- 测试 6：half_constexpr_if vs half_enable_if ---
  cout << "\n[Test 6] half：enable_if vs constexpr if：" << endl;
  cout << "  half_enable_if(10)    = " << half_enable_if(10) << endl;
  cout << "  half_constexpr_if(10) = " << half_constexpr_if(10) << endl;
  cout << "  half_constexpr_if(3.14) = " << half_constexpr_if(3.14) << endl;

  assert(half_constexpr_if(10) == 5);
  assert(half_constexpr_if(3.14) == 1.57);

  // --- 测试 7：if constexpr 与 requires ---
  cout << "\n[Test 7] if constexpr + requires（C++20）：" << endl;
  vector<int> v{1, 2, 3};
  cout << "  describe(vector)：" << describe(v) << endl;
  cout << "  describe(42)：    " << describe(42) << endl;
  cout << "  describe(\"abc\")： " << describe("abc") << endl;

  // --- 测试 8：多层分发 ---
  cout << "\n[Test 8] 多层 if constexpr 分发：" << endl;
  KernelLauncher<DataType::F32, AlignType::Align8>::launch();
  KernelLauncher<DataType::F16, AlignType::Align4>::launch();
  KernelLauncher<DataType::I8, AlignType::Align1>::launch();

  // --- 测试 9：validate_type（if constexpr 中的 static_assert） ---
  cout << "\n[Test 9] validate_type static_assert：" << endl;
  validate_type<int>();     // OK：sizeof(int) <= 8
  validate_type<char>();    // OK
  validate_type<double>();  // OK：sizeof(double) == 8
  // validate_type<long double>(); 如果 sizeof > 8 会 static_assert 失败
  cout << "  int、char、double 验证成功" << endl;

  // --- 测试 10：嵌套 if constexpr ---
  cout << "\n[Test 10] 嵌套 if constexpr：" << endl;
  auto nested_check = [](auto val) {
    if constexpr (std::is_arithmetic_v<decltype(val)>) {
      if constexpr (std::is_integral_v<decltype(val)>) {
        cout << "  Integral: " << val << endl;
      } else {
        cout << "  Floating: " << val << endl;
      }
    } else {
      cout << "  Non-arithmetic" << endl;
    }
  };
  nested_check(42);
  nested_check(3.14);
  nested_check("hello");

  cout << "\n所有测试通过！" << endl;
  return 0;
}
