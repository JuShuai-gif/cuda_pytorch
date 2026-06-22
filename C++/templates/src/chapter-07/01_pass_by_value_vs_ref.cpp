// =============================================================================
// 第 07.1 章：传值 vs 传引用
//
// 在编写模板函数时，传值和传引用的选择对性能和正确性都有重要影响。
//
// 主题：
//   1. 传值：会拷贝，安全但可能开销大
//   2. 传 const&：避免拷贝，但可能导致悬空引用
//   3. 传 &&：对右值使用 move 语义
//   4. T&& 何时折叠：T& vs T&&
//   5. 传值 vs 传引用的重载决议
//   6. 传值时的 C 数组退化
//   7. std::decay vs std::remove_reference 用于模板参数
//   8. 实际模板函数设计指南
//
// 编译：g++ -std=c++20 -o 01_pass_by_value_vs_ref 01_pass_by_value_vs_ref.cpp
// =============================================================================

#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// 辅助工具：追踪对象的拷贝和移动
// =============================================================================

struct Tracker {
  static inline int copy_count   = 0;
  static inline int move_count   = 0;
  static inline int ctor_count   = 0;

  int id;

  Tracker(int i = 0) : id(i) { ++ctor_count; }
  Tracker(Tracker const& other) : id(other.id) { ++copy_count; }
  Tracker(Tracker&& other) noexcept : id(other.id) {
    other.id = -1;
    ++move_count;
  }
  Tracker& operator=(Tracker const& other) {
    id = other.id;
    ++copy_count;
    return *this;
  }
  Tracker& operator=(Tracker&& other) noexcept {
    id = other.id;
    other.id = -1;
    ++move_count;
    return *this;
  }

  static void reset() {
    copy_count = 0; move_count = 0; ctor_count = 0;
  }
  static void report() {
    std::cout << "  copies=" << copy_count << ", moves=" << move_count
              << ", ctors=" << ctor_count << std::endl;
  }
};

// =============================================================================
// 1. 传值：会进行拷贝
// =============================================================================
// 简单且安全。调用方的对象不受影响。但对于大类型，
// 这会触发昂贵的深拷贝。

template <typename T>
void consume_by_value(T obj) {
  obj.id = 999;  // 修改的是局部拷贝，不影响原对象
}

// =============================================================================
// 2. 传 const&：不拷贝，但不能修改
// =============================================================================

template <typename T>
void inspect_by_const_ref(T const& obj) {
  std::cout << "  Read: id=" << obj.id << std::endl;
  // obj.id = 999;  // 错误：不能修改 const 引用
}

// =============================================================================
// 3. 传 &（可变引用）
// =============================================================================

template <typename T>
void modify_by_ref(T& obj) {
  obj.id = 777;
}

// =============================================================================
// 4. 传 &&（右值引用）：move 语义
// =============================================================================

template <typename T>
void consume_by_rvalue(T&& obj) {
  T local(std::move(obj));  // 从临时对象中窃取资源
  local.id = 555;
}

// =============================================================================
// 5. 数组退化：传值 vs 传引用
// =============================================================================

// 传值：数组退化为指针
template <typename T>
void array_by_value(T arr) {
  // arr 是指针，不是数组
  std::cout << "  sizeof(arr) (val) = " << sizeof(arr)
            << " (pointer, not array!)" << std::endl;
}

// 传引用：保留数组类型（包括大小）
template <typename T, std::size_t N>
void array_by_ref(T (&arr)[N]) {
  std::cout << "  sizeof(arr) (ref) = " << sizeof(arr)
            << " (full array, N=" << N << ")" << std::endl;
}

// =============================================================================
// 6. 悬空引用的陷阱
// =============================================================================
// 从创建了临时对象的函数中返回 const&。

template <typename T>
T const& max_by_ref_dangerous(T const& a, T const& b) {
  return (a < b) ? b : a;
}
// 如果这样调用：max_by_ref_dangerous(std::string("a"), std::string("b"))
// 临时对象在完整表达式结束后被销毁，返回的引用变成悬空。
// 这是一个经典陷阱。

// 安全版本：返回值
template <typename T>
T max_by_val_safe(T a, T b) {
  return (a < b) ? b : a;
}

// =============================================================================
// 7. std::decay：剥离引用、cv 限定符以及数组/函数退化
// =============================================================================

template <typename T>
void demonstrate_decay(T&& arg) {
  using Raw    = T;
  using Decayed = std::decay_t<T>;

  std::cout << "  T = " << typeid(Raw).name() << std::endl;
  std::cout << "  decay_t<T> = " << typeid(Decayed).name() << std::endl;
  std::cout << "  is_lvalue_ref = " << std::is_lvalue_reference_v<T>
            << std::endl;
  std::cout << "  is_rvalue_ref = " << std::is_rvalue_reference_v<T>
            << std::endl;
}

// =============================================================================
// 8. 重载决议：传值 vs Ref vs Const Ref
// =============================================================================

// 以下三个重载展示了编译器如何选择：
template <typename T>
void overload_demo(T) {
  std::cout << "  overload_demo(T) -- by value" << std::endl;
}

template <typename T>
void overload_demo(T&) {
  std::cout << "  overload_demo(T&) -- by lvalue ref" << std::endl;
}

template <typename T>
void overload_demo(T const&) {
  std::cout << "  overload_demo(T const&) -- by const ref" << std::endl;
}

// 注意：上面的重载对很多调用会产生歧义。更好的做法是使用
// SFINAE 或 concepts，这里只是展示模式。

// =============================================================================
// 9. 基准测试拷贝行为
// =============================================================================
// 比较不同传递策略下的拷贝次数。

template <typename T>
void benchmark_by_val(T obj) {
  (void)obj;  // 消费
}

template <typename T>
void benchmark_by_const_ref(T const& obj) {
  (void)obj;
}

template <typename T>
void benchmark_by_rvalue(T&& obj) {
  T local(std::move(obj));
  (void)local;
}

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 07.1 章：传值 vs 传引用 ===\n" << endl;

  // --- 测试 1：传值（拷贝） ---
  cout << "[Test 1] 传值：" << endl;
  {
    Tracker::reset();
    Tracker t(42);
    consume_by_value(t);
    Tracker::report();
    cout << "  原对象 id 未变：" << t.id << endl;
    assert(t.id == 42);  // 原对象未变
    assert(Tracker::copy_count == 1);  // 一次拷贝
  }

  // --- 测试 2：传 const&（不拷贝） ---
  cout << "\n[Test 2] 传 const&：" << endl;
  {
    Tracker::reset();
    Tracker t(42);
    inspect_by_const_ref(t);
    Tracker::report();
    assert(Tracker::copy_count == 0);
    assert(Tracker::move_count == 0);
  }

  // --- 测试 3：传可变引用 ---
  cout << "\n[Test 3] 传可变引用：" << endl;
  {
    Tracker::reset();
    Tracker t(42);
    modify_by_ref(t);
    Tracker::report();
    cout << "  修改后的 id：" << t.id << endl;
    assert(t.id == 777);
    assert(Tracker::copy_count == 0);
  }

  // --- 测试 4：传右值引用（移动） ---
  cout << "\n[Test 4] 传右值引用：" << endl;
  {
    Tracker::reset();
    consume_by_rvalue(Tracker(42));
    Tracker::report();
    assert(Tracker::move_count >= 1);
  }

  // --- 测试 5：数组退化 ---
  cout << "\n[Test 5] 数组退化：" << endl;
  int arr[10] = {};
  array_by_value(arr);    // 退化为 int*
  array_by_ref(arr);      // 保持 int[10]
  cout << "  实际 sizeof(arr) = " << sizeof(arr) << endl;
  assert(sizeof(arr) == 10 * sizeof(int));

  // --- 测试 6：悬空引用意识 ---
  cout << "\n[Test 6] 悬空引用演示：" << endl;
  // 不会用临时对象调用 max_by_ref_dangerous -- 那是 UB！
  int a = 10, b = 20;
  int const& safe_ref = max_by_ref_dangerous(a, b);
  cout << "  max_by_ref_dangerous(lval, lval) = " << safe_ref
       << " (safe: both lvalues)" << endl;
  assert(safe_ref == 20);

  // max_by_ref_dangerous(10, 20) 会产生悬空引用 -- 不要这样做
  cout << "  警告：max_by_ref_dangerous(10, 20) 会产生悬空引用！" << endl;

  // 安全版本
  auto safe_val = max_by_val_safe(10, 20);
  cout << "  max_by_val_safe(10, 20) = " << safe_val
       << " (safe: returns by value)" << endl;
  assert(safe_val == 20);

  // --- 测试 7：std::decay ---
  cout << "\n[Test 7] std::decay 演示：" << endl;
  int xx = 42;
  cout << "  传递左值 int：" << endl;
  demonstrate_decay(xx);
  cout << "  传递右值 int：" << endl;
  demonstrate_decay(42);

  static_assert(is_same_v<decay_t<int&>, int>);
  static_assert(is_same_v<decay_t<int&&>, int>);
  static_assert(is_same_v<decay_t<int const&>, int>);
  static_assert(is_same_v<decay_t<int[5]>, int*>);

  // --- 测试 8：拷贝次数基准测试 ---
  cout << "\n[Test 8] 拷贝次数基准测试：" << endl;
  {
    Tracker::reset();
    Tracker t(42);
    benchmark_by_val(t);
    cout << "  by_val: ";
    Tracker::report();
    assert(Tracker::copy_count == 1);
  }
  {
    Tracker::reset();
    Tracker t(42);
    benchmark_by_const_ref(t);
    cout << "  by_const_ref: ";
    Tracker::report();
    assert(Tracker::copy_count == 0);
  }
  {
    Tracker::reset();
    benchmark_by_rvalue(Tracker(42));
    cout << "  by_rvalue: ";
    Tracker::report();
    assert(Tracker::move_count >= 1);
  }

  // --- 测试 9：std::string 传值 vs const ref 的性能模式 ---
  cout << "\n[Test 9] std::string 性能模式：" << endl;
  // 传值：拷贝字符串（堆分配）
  auto consume_str_by_val = [](string s) {
    cout << "    by_val 收到：'" << s << "'" << endl;
  };
  // 传 const ref：不拷贝
  auto consume_str_by_cref = [](string const& s) {
    cout << "    by_cref 收到：'" << s << "'" << endl;
  };
  // 传值 + move（sink 参数的现代模式）
  auto consume_str_sink = [](string s) {
    string local(std::move(s));
    cout << "    sink 收到：'" << local << "'" << endl;
  };

  string long_str(1000, 'x');  // 大字符串
  cout << "  用左值调用 by_val..." << endl;
  consume_str_by_val(long_str);     // 拷贝
  cout << "  用右值调用 by_val..." << endl;
  consume_str_by_val(string("hi")); // 移动（或拷贝消除）

  cout << "  调用 by_cref..." << endl;
  consume_str_by_cref(long_str);    // 不拷贝

  cout << "  用左值调用 sink..." << endl;
  consume_str_sink(long_str);       // 拷贝（long_str 仍然有效）
  cout << "    long_str 仍然有效：'" << long_str.substr(0, 3) << "...'"
       << endl;

  cout << "  用右值调用 sink..." << endl;
  consume_str_sink(string("world"));  // 移动

  cout << "\n所有测试通过！" << endl;
  return 0;
}
