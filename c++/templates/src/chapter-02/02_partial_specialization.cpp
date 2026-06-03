// =============================================================================
// 第 02.2 章：偏特化
//
// 演示：
//   1. 主模板 vs 偏特化 vs 全特化
//   2. 指针特化（针对指针类型优化）
//   3. void* 包装器（类型擦除存储）
//   4. 通过偏特化提取 trait
//   5. 常见模式：通过特化实现 is_same、remove_const、conditional
//   6. 数组特化（定长和不定长）
//
// 注意：函数模板不能偏特化（只能全特化），
//       而类模板既可以偏特化也可以全特化。
//
// 编译：g++ -std=c++20 -o 02_partial_specialization 02_partial_specialization.cpp
// =============================================================================

#include <cassert>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 1. 主模板与指针偏特化
// =============================================================================
// 一个简单的类型包装器，在编译期存储类型注解。
// 主模板处理通用类型。

template <typename T>
struct TypeInfo {
  static constexpr char const* category = "通用";
  static constexpr bool is_pointer      = false;
  using value_type = T;

  static void print() {
    std::cout << "TypeInfo<通用>：T，sizeof=" << sizeof(T) << std::endl;
  }
};

// 指针类型的偏特化：T*
// 当 T 匹配指针模式时编译器选择此版本。
// 此特化比主模板更具体。

template <typename T>
struct TypeInfo<T*> {
  static constexpr char const* category = "指针";
  static constexpr bool is_pointer      = true;
  using value_type = T;  // 剥离指针

  static void print() {
    std::cout << "TypeInfo<指针>：T*，指向类型 sizeof=" << sizeof(T)
              << std::endl;
  }
};

// const 类型的偏特化
template <typename T>
struct TypeInfo<T const> {
  static constexpr char const* category = "const";
  static constexpr bool is_pointer      = false;
  using value_type = T;

  static void print() {
    std::cout << "TypeInfo<const>：T，sizeof=" << sizeof(T) << std::endl;
  }
};

// =============================================================================
// 2. Void 指针包装器（小型类型擦除）
// =============================================================================
// 通用前向声明的主模板，包装 void* 并跟踪原始类型
//（运行时擦除，编译期保留）。

template <typename T>
class VoidPtrWrapper {
 public:
  explicit VoidPtrWrapper(T* ptr) : ptr_(static_cast<void*>(ptr)) {}

  // 用正确的类型恢复指针
  T* get() { return static_cast<T*>(ptr_); }
  T const* get() const { return static_cast<T const*>(ptr_); }

 private:
  void* ptr_;
};

// void* 的偏特化：作为基例。
// 这避免了无限递归（T = void 会是 VoidPtrWrapper<void>
// 并存储 void** 等）

template <>
class VoidPtrWrapper<void> {
 public:
  explicit VoidPtrWrapper(void* ptr) : ptr_(ptr) {}
  void* get() { return ptr_; }
  void const* get() const { return ptr_; }

 private:
  void* ptr_;
};

// =============================================================================
// 3. 通过偏特化提取 Trait
// =============================================================================
// 从头构建的一组类型 traits，以说明 <type_traits> 所依赖的特化机制。

// --- is_same（二元 trait，多重特化）---
template <typename T, typename U>
struct IsSame : std::false_type {};

template <typename T>
struct IsSame<T, T> : std::true_type {};

template <typename T, typename U>
inline constexpr bool my_is_same_v = IsSame<T, U>::value;

// --- remove_const ---
template <typename T>
struct RemoveConst {
  using type = T;
};

template <typename T>
struct RemoveConst<T const> {
  using type = T;
};

template <typename T>
using my_remove_const_t = typename RemoveConst<T>::type;

// --- conditional：编译期选择类型 ---
template <bool Cond, typename T, typename F>
struct Conditional {
  using type = F;
};

template <typename T, typename F>
struct Conditional<true, T, F> {
  using type = T;
};

template <bool Cond, typename T, typename F>
using my_conditional_t = typename Conditional<Cond, T, F>::type;

// --- is_integral（演示每种整数类型的全特化）---
template <typename T>
struct IsIntegral : std::false_type {};

template <> struct IsIntegral<bool>           : std::true_type {};
template <> struct IsIntegral<char>           : std::true_type {};
template <> struct IsIntegral<signed char>    : std::true_type {};
template <> struct IsIntegral<unsigned char>  : std::true_type {};
template <> struct IsIntegral<short>          : std::true_type {};
template <> struct IsIntegral<unsigned short> : std::true_type {};
template <> struct IsIntegral<int>            : std::true_type {};
template <> struct IsIntegral<unsigned int>   : std::true_type {};
template <> struct IsIntegral<long>           : std::true_type {};
template <> struct IsIntegral<unsigned long>  : std::true_type {};
template <> struct IsIntegral<long long>           : std::true_type {};
template <> struct IsIntegral<unsigned long long>  : std::true_type {};

template <typename T>
inline constexpr bool my_is_integral_v = IsIntegral<T>::value;

// =============================================================================
// 4. 数组特化（定长和不定长）
// =============================================================================

// 定长数组：T[N]
template <typename T, std::size_t N>
struct TypeInfo<T[N]> {
  static constexpr char const* category = "定长数组";
  using value_type = T;
  static constexpr std::size_t extent = N;

  static void print() {
    std::cout << "TypeInfo<数组>：T[" << N << "]，元素 sizeof="
              << sizeof(T) << std::endl;
  }
};

// 不定长数组：T[]（如函数参数）
template <typename T>
struct TypeInfo<T[]> {
  static constexpr char const* category = "不定长数组";
  using value_type = T;

  static void print() {
    std::cout << "TypeInfo<T[]>：不定长，元素 sizeof=" << sizeof(T)
              << std::endl;
  }
};

// =============================================================================
// 5. 带指针偏特化的 Stack
// =============================================================================
// 演示偏特化如何为指针类型提供与值类型不同的行为。

template <typename T>
struct SmartStack {
  std::vector<T> data;

  void push(T const& value) { data.push_back(value); }
  T    pop() {
    T val = std::move(data.back());
    data.pop_back();
    return val;
  }
  std::size_t size() const { return data.size(); }
};

// 指针类型的偏特化：接管堆对象的所有权
template <typename T>
struct SmartStack<T*> {
  std::vector<T*> data;

  void push(T* ptr) { data.push_back(ptr); }

  T* pop() {
    T* ptr = data.back();
    data.pop_back();
    return ptr;
  }

  // 所有权感知的清理
  ~SmartStack() {
    for (auto* ptr : data) {
      delete ptr;  // 假设拥有所有权
    }
  }

  std::size_t size() const { return data.size(); }
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 02.2 章：偏特化 ===\n" << endl;

  // --- 测试 1：TypeInfo 特化 ---
  cout << "[测试 1] TypeInfo 分发：" << endl;
  TypeInfo<int>::print();
  TypeInfo<int*>::print();
  TypeInfo<int const>::print();
  TypeInfo<double*>::print();
  TypeInfo<char const*>::print();  // const char* 触发指针特化，而非 const 特化

  // Trait 值
  static_assert(TypeInfo<int>::is_pointer == false);
  static_assert(TypeInfo<int*>::is_pointer == true);
  static_assert(my_is_same_v<TypeInfo<int*>::value_type, int>);

  // --- 测试 2：数组特化 ---
  cout << "\n[测试 2] 数组类型信息：" << endl;
  TypeInfo<int[5]>::print();
  TypeInfo<double[]>::print();

  static_assert(TypeInfo<int[5]>::extent == 5);
  static_assert(my_is_same_v<TypeInfo<int[3]>::value_type, int>);

  // --- 测试 3：VoidPtrWrapper ---
  cout << "\n[测试 3] VoidPtrWrapper：" << endl;
  int val = 42;
  VoidPtrWrapper<int> vpw(&val);
  cout << "  包装 int*：" << *vpw.get() << endl;
  assert(*vpw.get() == 42);

  double dval = 3.14;
  VoidPtrWrapper<double> vpw_d(&dval);
  cout << "  包装 double*：" << *vpw_d.get() << endl;
  assert(*vpw_d.get() == 3.14);

  // void 特化
  int v2 = 99;
  VoidPtrWrapper<void> vpw_void(&v2);
  cout << "  包装 void*：" << *static_cast<int*>(vpw_void.get()) << endl;
  assert(*static_cast<int*>(vpw_void.get()) == 99);

  // --- 测试 4：自定义类型 traits ---
  cout << "\n[测试 4] 自定义类型 traits：" << endl;
  static_assert(IsSame<int, int>::value == true);
  static_assert(IsSame<int, double>::value == false);
  static_assert(my_is_same_v<int, int>);

  static_assert(my_is_same_v<my_remove_const_t<int const>, int>);
  static_assert(my_is_same_v<my_remove_const_t<int>, int>);
  static_assert(my_is_same_v<my_remove_const_t<int const volatile>, int volatile>);

  static_assert(my_is_same_v<my_conditional_t<true, int, double>, int>);
  static_assert(my_is_same_v<my_conditional_t<false, int, double>, double>);

  static_assert(my_is_integral_v<int>);
  static_assert(my_is_integral_v<char>);
  static_assert(my_is_integral_v<unsigned long long>);
  static_assert(!my_is_integral_v<float>);
  static_assert(!my_is_integral_v<string>);
  cout << "  所有 static_assert trait 检查通过！" << endl;

  // --- 测试 5：SmartStack 指针特化 ---
  cout << "\n[测试 5] SmartStack 指针特化：" << endl;
  // 值版本
  SmartStack<int> ss_val;
  ss_val.push(1);
  ss_val.push(2);
  cout << "  值栈大小：" << ss_val.size() << endl;
  assert(ss_val.size() == 2);
  assert(ss_val.pop() == 2);

  // 指针版本（拥有堆对象所有权）
  SmartStack<int*> ss_ptr;
  ss_ptr.push(new int(10));
  ss_ptr.push(new int(20));
  cout << "  指针栈大小：" << ss_ptr.size() << endl;
  assert(ss_ptr.size() == 2);

  int* ptr = ss_ptr.pop();
  cout << "  弹出指针的值：" << *ptr << endl;
  assert(*ptr == 20);
  delete ptr;  // 手动清理，因为我们已经弹出了它

  // 剩余元素（以及栈本身）由 ~SmartStack 销毁

  // --- 测试 6：复杂嵌套 ---
  cout << "\n[测试 6] 嵌套特化：" << endl;
  // 指向 const int 的指针 -> 指针特化
  TypeInfo<int const*>::print();
  // 指向 int 的 const 指针 -> 指针特化的 const 特化
  TypeInfo<int* const>::print();

  static_assert(my_is_same_v<TypeInfo<int const*>::value_type, int const>);
  static_assert(my_is_same_v<TypeInfo<int* const>::value_type, int*>);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
