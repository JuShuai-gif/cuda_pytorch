// =============================================================================
// 第 07.2 章：完美转发 -- 实现 forward<T>
//
// 完美转发是一种将参数的值类别（左值/右值）在通过中间函数模板传递时
// 保持不变的技术。它依赖于：
//
//   1. 万能/转发引用：推导上下文中的 T&&
//   2. 引用折叠规则
//   3. std::forward<T>（或自定义实现）
//   4. std::remove_reference（用于剥离引用）
//
// 本文从头实现一个简化的 forward<T>，并演示各种转发模式。
//
// 编译：g++ -std=c++20 -o 02_perfect_forwarding 02_perfect_forwarding.cpp
// =============================================================================

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// 1. 引用折叠规则（基础）
// =============================================================================
// C++ 在 T 是引用类型时对 T&& 应用引用折叠：
//
//   T     | T&&   | 结果
//   ------+-------+--------
//   X&    | &&    | X&   （左值引用折叠）
//   X&&   | &&    | X&&  （右值引用保持右值）
//   X     | &&    | X&&  （无引用，保持右值引用）
//
// 这就是转发引用能够工作的机制。

template <typename T>
struct ReferenceCollapseDemo {
  // 当 T = int&： T&& = int& &&  -> int&  （折叠）
  // 当 T = int&&：T&& = int&& && -> int&& （保持）
  // 当 T = int：  T&& = int&&    -> int&& （保持）

  static void print_type() {
    std::cout << "  T = " << typeid(T).name() << std::endl;
    std::cout << "  T&& = " << typeid(T&&).name() << std::endl;
  }
};

// =============================================================================
// 2. 实现 remove_reference（forward 的基础）
// =============================================================================

template <typename T> struct RemoveRef      { using type = T; };
template <typename T> struct RemoveRef<T&>  { using type = T; };
template <typename T> struct RemoveRef<T&&> { using type = T; };

template <typename T>
using remove_ref_t = typename RemoveRef<T>::type;

// =============================================================================
// 3. 从零实现 forward<T>
// =============================================================================
// forward<T> 有条件地将其参数转换为右值引用。
//
// 关键洞见：T 由转发函数推导，携带着值类别信息。
// 如果 T = X&，参数是左值；如果 T = X，参数是右值。

namespace custom {

// 左值重载：T 被推导为 X&，折叠为 X&
template <typename T>
constexpr T&& forward(remove_ref_t<T>& arg) noexcept {
  return static_cast<T&&>(arg);
}

// 右值重载：T 被推导为 X，返回 X&&
template <typename T>
constexpr T&& forward(remove_ref_t<T>&& arg) noexcept {
  static_assert(!std::is_lvalue_reference_v<T>,
                "Cannot forward an rvalue as an lvalue");
  return static_cast<T&&>(arg);
}

}  // namespace custom

// =============================================================================
// 4. 演示转发模式
// =============================================================================

// 一个简单的 sink 函数，区分左值和右值
void sink(int& x) {
  std::cout << "  sink(lvalue ref): " << x << std::endl;
  x += 10;
}

void sink(int&& x) {
  std::cout << "  sink(rvalue ref): " << x << std::endl;
}

// 转发包装器：使用 T&& + forward<T> 来保持值类别
template <typename T>
void forwarder(T&& arg) {
  std::cout << "  forwarder: " << (std::is_lvalue_reference_v<T> ? "lvalue" : "rvalue")
            << std::endl;
  sink(custom::forward<T>(arg));
}

// 字符串的转发包装器
void sink_str(std::string& s) {
  std::cout << "  sink(lvalue string): '" << s << "'" << std::endl;
}

void sink_str(std::string&& s) {
  std::cout << "  sink(rvalue string): '" << s << "' (moving)" << std::endl;
  std::string consumed(std::move(s));
}

template <typename T>
void forwarder_str(T&& s) {
  sink_str(custom::forward<T>(s));
}

// =============================================================================
// 5. 就地构造模式：完美转发给构造函数
// =============================================================================
// 完美转发的经典用法：emplace_back、make_unique 等。

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_custom(Args&&... args) {
  return std::unique_ptr<T>(new T(custom::forward<Args>(args)...));
}

struct Widget {
  std::string name;
  int         value;

  Widget(std::string const& n, int v) : name(n), value(v) {
    std::cout << "  Widget(lvalue str): " << name << std::endl;
  }
  Widget(std::string&& n, int v) : name(std::move(n)), value(v) {
    std::cout << "  Widget(rvalue str): " << name << std::endl;
  }
};

// =============================================================================
// 6. 使用可变参数模板进行转发
// =============================================================================

template <typename F, typename... Args>
decltype(auto) call_with_forward(F&& f, Args&&... args) {
  return f(custom::forward<Args>(args)...);
}

int add(int a, int b) { return a + b; }

void print_values(int a, std::string const& b, double c) {
  std::cout << "  print_values: " << a << ", '" << b << "', " << c
            << std::endl;
}

// =============================================================================
// 7. 陷阱：对同一个对象转发两次
// =============================================================================
// 不要转发同一个对象两次，除非你知道第一个消费者不会从中 move
// （或者你明确希望两者都获得拷贝/左值）。

template <typename T>
void forward_twice_bad(T&& arg) {
  auto copy = arg;                    // 拷贝（始终安全）
  sink(custom::forward<T>(arg));       // 第一次使用
  // sink(custom::forward<T>(arg));    // 第二次使用：可能已是 moved-from 状态！
  (void)copy;
}

// =============================================================================
// 8. 转发引用 vs 具体类型的重载
// =============================================================================

// 转发引用重载：几乎匹配所有情况
template <typename T>
void overload_fwd(T&& arg) {
  std::cout << "  overload_fwd(T&&): generic forwarding" << std::endl;
}

// 具体类型重载：精确匹配时优先于模板
void overload_fwd(int arg) {
  std::cout << "  overload_fwd(int): concrete" << std::endl;
}

// =============================================================================
// 9. auto&&：变量声明中的转发引用
// =============================================================================
// auto&&（无需模板推导）也是转发引用。
// 既可以绑定到左值，也可以绑定到右值。

void demo_auto_ref() {
  int x = 42;

  auto&& r1 = x;           // int&  （x 是左值）
  auto&& r2 = 42;          // int&& （42 是右值）
  auto&& r3 = std::move(x); // int&& （std::move 产生 xvalue）

  std::cout << "  auto&& r1 (左值): is_rref="
            << std::is_rvalue_reference_v<decltype(r1)> << std::endl;
  std::cout << "  auto&& r2 (右值): is_rref="
            << std::is_rvalue_reference_v<decltype(r2)> << std::endl;

  static_assert(std::is_lvalue_reference_v<decltype(r1)>);
  static_assert(std::is_rvalue_reference_v<decltype(r2)>);
}

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 07.2 章：完美转发 ===\n" << endl;

  // --- 测试 1：引用折叠 ---
  cout << "[Test 1] 引用折叠：" << endl;
  ReferenceCollapseDemo<int>::print_type();
  ReferenceCollapseDemo<int&>::print_type();
  ReferenceCollapseDemo<int&&>::print_type();

  static_assert(is_same_v<int&, int&>);
  static_assert(is_same_v<int&&, int&&>);
  // 折叠：(int&)&& -> int&
  static_assert(is_same_v<int&, int&>);
  // 折叠：(int&&)&& -> int&&
  static_assert(is_same_v<int&&, int&&>);

  // --- 测试 2：remove_ref_t ---
  cout << "\n[Test 2] remove_ref_t：" << endl;
  static_assert(is_same_v<remove_ref_t<int>, int>);
  static_assert(is_same_v<remove_ref_t<int&>, int>);
  static_assert(is_same_v<remove_ref_t<int&&>, int>);
  static_assert(is_same_v<remove_ref_t<int const&>, int const>);
  cout << "  所有 remove_ref_t 检查通过" << endl;

  // --- 测试 3：自定义 forward ---
  cout << "\n[Test 3] 自定义 forward<T>：" << endl;
  int val = 10;
  cout << "  转发左值：" << endl;
  forwarder(val);
  cout << "  转发后 val（已修改）：" << val << endl;
  assert(val == 20);  // sink(int&) 修改了它

  cout << "  转发右值：" << endl;
  forwarder(42);

  // --- 测试 4：字符串转发 ---
  cout << "\n[Test 4] 字符串转发：" << endl;
  std::string s = "hello";
  cout << "  转发左值字符串：" << endl;
  forwarder_str(s);
  cout << "  转发后 s：'" << s << "'" << endl;

  cout << "  转发右值字符串：" << endl;
  forwarder_str(std::string("world"));

  // --- 测试 5：make_unique_custom ---
  cout << "\n[Test 5] make_unique_custom（完美转发）：" << endl;
  auto w1 = make_unique_custom<Widget>("bye", 42);     // 右值字符串
  auto w2 = make_unique_custom<Widget>(s, 99);         // 左值字符串
  cout << "  w1->name='" << w1->name << "', w2->name='" << w2->name << "'"
       << endl;

  // --- 测试 6：可变参数转发 ---
  cout << "\n[Test 6] 可变参数完美转发：" << endl;
  cout << "  call_with_forward(add, 3, 4) = "
       << call_with_forward(add, 3, 4) << endl;
  cout << "  call_with_forward(print_values, ...)：" << endl;
  std::string msg = "test";
  call_with_forward(print_values, 1, msg, 3.14);

  // --- 测试 7：转发陷阱 ---
  cout << "\n[Test 7] 对同一对象转发两次（避免！）：" << endl;
  int z = 5;
  forward_twice_bad(z);  // OK：第一次转发是左值，第二次是拷贝
  cout << "  forward_twice_bad 后 z：" << z << " (被 sink 修改)" << endl;

  // --- 测试 8：重载决议 ---
  cout << "\n[Test 8] 转发引用的重载决议：" << endl;
  overload_fwd(42);       // 调用具体 int 重载
  overload_fwd(3.14);     // 调用转发引用模板
  overload_fwd("hello");  // 调用转发引用模板

  // --- 测试 9：auto&& ---
  cout << "\n[Test 9] auto&& 转发引用：" << endl;
  demo_auto_ref();

  cout << "\n所有测试通过！" << endl;
  return 0;
}
