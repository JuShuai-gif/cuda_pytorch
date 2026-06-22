// =============================================================================
// 第 03.1 章：非类型模板参数 -- 基础
//
// 非类型模板参数允许编译期常量参数化模板。
// 它们必须是结构类型（整数、枚举、指针、左值引用等）。
// C++20 允许浮点类型和具有公有基类且无 mutable 成员的类类型。
//
// 主题：
//   1. 整型非类型：Stack<T, maxSize>
//   2. 从非类型推导数组大小
//   3. auto 非类型参数（C++17）
//   4. 浮点非类型（C++20）
//   5. 字符串字面量作为非类型（C++20 通过 char 数组包装器）
//   6. 函数模板中的非类型
//
// 编译：g++ -std=c++20 -o 01_nontype_basics 01_nontype_basics.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. 定容量的 Stack（非类型整型参数）
// =============================================================================
// 使用非类型整数作为栈容量可避免动态分配。
// 大小在编译期已知，可实现更好的优化。

template <typename T, std::size_t MaxSize>
class FixedStack {
 public:
  static constexpr std::size_t max_size = MaxSize;

  FixedStack() : top_(0) {}

  bool push(T const& value) {
    if (top_ >= MaxSize) return false;  // 栈满
    data_[top_++] = value;
    return true;
  }

  bool push(T&& value) {
    if (top_ >= MaxSize) return false;
    data_[top_++] = std::move(value);
    return true;
  }

  bool pop() {
    if (top_ == 0) return false;
    --top_;
    return true;
  }

  T& top() { return data_[top_ - 1]; }
  T const& top() const { return data_[top_ - 1]; }

  [[nodiscard]] bool empty() const { return top_ == 0; }
  [[nodiscard]] bool full() const { return top_ == MaxSize; }
  [[nodiscard]] std::size_t size() const { return top_; }
  [[nodiscard]] std::size_t capacity() const { return MaxSize; }

  // 迭代器支持
  T* begin() { return data_; }
  T* end() { return data_ + top_; }
  T const* begin() const { return data_; }
  T const* end() const { return data_ + top_; }

 private:
  T           data_[MaxSize];  // 固定大小数组，无堆分配
  std::size_t top_;
};

// =============================================================================
// 2. 通过非类型推导的数组大小推导
// =============================================================================
// 模板实参推导可以从实参确定数组大小。
// 这就是 std::size 对内置数组的工作原理。

template <typename T, std::size_t N>
constexpr std::size_t array_size(T (&/*arr*/)[N]) noexcept {
  return N;
}

// 用于比较的 C++17 版本使用 std::size
// 以上是经典实现。

// =============================================================================
// 3. 带 auto 非类型的函数模板（C++17）
// =============================================================================
// 从 C++17 开始，非类型参数可以使用 'auto' 作为占位符。

template <auto Value>
constexpr auto identity_value = Value;

template <auto N>
struct WrapValue {
  static constexpr auto value = N;
};

// =============================================================================
// 4. 浮点非类型参数（C++20）
// =============================================================================
// C++20 允许浮点非类型模板参数。对编译期数学常量很有用。

template <double Scale>
struct Scaler {
  static constexpr double factor = Scale;

  static double apply(double x) { return x * factor; }
};

// =============================================================================
// 5. C++20 字符串字面量作为非类型（通过 FixedString 包装器）
// =============================================================================
// 如果包装在结构类型（所有成员公有，无 mutable 状态）中，
// 字符串字面量可用作非类型参数。

template <std::size_t N>
struct FixedString {
  char data[N]{};

  constexpr FixedString() = default;

  constexpr FixedString(char const (&str)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
      data[i] = str[i];
    }
  }

  constexpr std::size_t size() const { return N - 1; }  // 不含空终止符

  constexpr char const* c_str() const { return data; }

  // 满足结构类型要求
  friend constexpr bool operator==(FixedString const&,
                                    FixedString const&) = default;
};

// 使用 FixedString 作为非类型模板参数
template <FixedString Name>
struct Logger {
  static void log(std::string const& msg) {
    std::cout << "[" << Name.data << "] " << msg << std::endl;
  }
};

// =============================================================================
// 6. 带默认值的类模板中的非类型
// =============================================================================
// 非类型参数可以有默认值，就像类型参数一样。

template <typename T, std::size_t Alignment = alignof(T)>
struct AlignedStorage {
  alignas(Alignment) unsigned char buffer[sizeof(T)];

  T* ptr() { return reinterpret_cast<T*>(buffer); }
  T const* ptr() const { return reinterpret_cast<T const*>(buffer); }
};

// =============================================================================
// 7. 指针和引用非类型参数
// =============================================================================
// C++17 之前，非类型可以是有外部链接的指针/引用。
// C++17+ 允许任何 constexpr 指针/引用。

// 具有外部链接的全局变量（C++17 之前要求）
extern int global_value;
int global_value = 100;

template <int* Ptr>
struct PointerHolder {
  static int get() { return *Ptr; }
  static void set(int v) { *Ptr = v; }
};

template <int& Ref>
struct ReferenceHolder {
  static int get() { return Ref; }
  static void set(int v) { Ref = v; }
};

// =============================================================================
// 8. 模板模板参数 + 非类型组合
// =============================================================================
// 演示混合使用模板模板参数和非类型参数。

template <template <typename, std::size_t> class Container,
          typename T, std::size_t N>
struct ContainerWrapper {
  Container<T, N> c;
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 03.1 章：非类型模板参数 ===\n" << endl;

  // --- 测试 1：FixedStack ---
  cout << "[测试 1] 具有编译期容量的 FixedStack：" << endl;
  FixedStack<int, 5> fs;
  assert(fs.empty());
  assert(fs.capacity() == 5);

  fs.push(10);
  fs.push(20);
  fs.push(30);
  cout << "  3 次 push 后的大小：" << fs.size() << endl;
  assert(fs.size() == 3 && fs.top() == 30);

  fs.push(40);
  fs.push(50);
  assert(fs.full());
  bool overflow = fs.push(60);
  cout << "  在满栈上 push：" << (overflow ? "成功" : "拒绝")
       << endl;
  assert(!overflow && fs.full());

  // Range-for
  cout << "  元素：";
  for (auto v : fs) cout << v << " ";
  cout << endl;

  // 不同的非类型创建不同的类型
  FixedStack<int, 10> fs2;
  static_assert(!is_same_v<decltype(fs), decltype(fs2)>);
  cout << "  FixedStack<int,5> != FixedStack<int,10>：已确认" << endl;

  // --- 测试 2：array_size ---
  cout << "\n[测试 2] 数组大小推导：" << endl;
  int arr1[42];
  double arr2[7];
  cout << "  array_size(arr1) = " << array_size(arr1) << endl;
  cout << "  array_size(arr2) = " << array_size(arr2) << endl;
  static_assert(array_size(arr1) == 42);
  static_assert(array_size(arr2) == 7);

  // --- 测试 3：auto 非类型 ---
  cout << "\n[测试 3] auto 非类型参数（C++17）：" << endl;
  cout << "  identity_value<42> = " << identity_value<42> << endl;
  cout << "  identity_value<'X'> = " << identity_value<'X'> << endl;
  cout << "  identity_value<true> = " << identity_value<true> << endl;

  using WV = WrapValue<999>;
  cout << "  WrapValue<999>::value = " << WV::value << endl;
  static_assert(WV::value == 999);

  // --- 测试 4：浮点非类型（C++20）---
  cout << "\n[测试 4] 浮点非类型（C++20）：" << endl;
  Scaler<3.14> s_pi;
  Scaler<2.0>  s_double;
  cout << "  Scaler<3.14>::apply(10) = " << s_pi.apply(10) << endl;
  cout << "  Scaler<2.0>::apply(10)  = " << s_double.apply(10) << endl;
  static_assert(Scaler<3.14>::factor == 3.14);
  static_assert(Scaler<2.0>::factor == 2.0);

  // --- 测试 5：FixedString 作为非类型（C++20）---
  cout << "\n[测试 5] C++20 字符串作为非类型：" << endl;
  Logger<"GPU_Kernel">::log("正在启动计算...");
  Logger<"CUDA_Runtime">::log("正在初始化设备...");

  static_assert(FixedString("hello").size() == 5);

  // --- 测试 6：非类型默认值 ---
  cout << "\n[测试 6] 默认对齐存储：" << endl;
  AlignedStorage<int> as_int;
  AlignedStorage<double, 64> as_double_aligned64;
  cout << "  alignof(AlignedStorage<int>)     = " << alignof(decltype(as_int))
       << "（默认：alignof(int)=" << alignof(int) << "）" << endl;
  cout << "  alignof(AlignedStorage<double,64>) = "
       << alignof(decltype(as_double_aligned64)) << endl;

  // --- 测试 7：指针/引用非类型 ---
  cout << "\n[测试 7] 指针/引用非类型：" << endl;
  PointerHolder<&global_value>::set(42);
  cout << "  PointerHolder<&global_value>::get() = "
       << PointerHolder<&global_value>::get() << endl;
  assert(PointerHolder<&global_value>::get() == 42);

  ReferenceHolder<global_value>::set(77);
  cout << "  ReferenceHolder<global_value>::get() = "
       << ReferenceHolder<global_value>::get() << endl;
  assert(ReferenceHolder<global_value>::get() == 77);
  assert(global_value == 77);

  // --- 测试 8：不同实例化具有不同的静态存储 ---
  cout << "\n[测试 8] 独立实例化：" << endl;
  FixedStack<int, 3> fsi3;
  FixedStack<int, 5> fsi5;
  FixedStack<double, 5> fsd5;

  fsi3.push(1);
  fsi5.push(999);
  assert(fsi5.top() == 999);
  assert(fsi3.top() == 1);

  cout << "  每个 (T, MaxSize) 对都是一个独立的类型。" << endl;
  cout << "  fsi3.top()=" << fsi3.top() << "，fsi5.top()=" << fsi5.top() << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
