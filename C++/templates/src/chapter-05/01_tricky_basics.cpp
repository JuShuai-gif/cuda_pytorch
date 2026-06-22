// =============================================================================
// 第 05.1 章：棘手基础 -- typename、template、this->、零初始化
//
// 编写模板代码时，需要某些关键字和模式来帮助编译器
// 正确解析依赖名。本文件涵盖：
//   1. 依赖类型名的 `typename`
//   2. 依赖成员模板的 `template`
//   3. 依赖基类成员访问的 `this->`
//   4. 模板参数 T{} 的零初始化
//   5. `.template` 和 `::template` 和 `->template` 语法
//   6. 类模板的默认模板实参
//   7. 默认实参中的模板参数
//
// 编译：g++ -std=c++20 -o 01_tricky_basics 01_tricky_basics.cpp
// =============================================================================

#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// 1. 依赖类型名的 typename
// =============================================================================
// 当嵌套名依赖于模板参数时，编译器无法知道它是类型还是值。
// 使用 `typename` 声明它是一个类型。

template <typename T>
struct ContainerTraits {
  // T::value_type 是依赖名；必须使用 typename
  using value_type = typename T::value_type;
  using iterator   = typename T::iterator;
  using const_iterator = typename T::const_iterator;

  // 同样适用于依赖模板别名
  using size_type = typename T::size_type;
};

// 没有 typename，这将失败：
// template <typename T>
// struct Bad {
//   T::value_type v;  // 错误：'T::value_type' 是类型但编译器
// };                  //       认为它可能是静态成员

// =============================================================================
// 2. 依赖成员模板的 template
// =============================================================================
// 在依赖对象上调用成员函数模板时，编译器将 '<' 解析为小于号，
// 除非前面有 `template` 关键字。

template <typename T>
struct MemberTemplateDemo {
  T obj;

  // obj.template method<int>() 是必需的，因为 obj 的类型依赖 T
  void call_member_template() {
    // obj.doSomething<int>();  // 错误：< 被解析为小于号
    obj.template doSomething<int>();  // 正确：template 关键字
    obj.template doSomething<double>();
  }
};

// 具有成员函数模板的具体类型
struct HasMemberTemplate {
  template <typename U>
  void doSomething() {
    std::cout << "  doSomething<" << typeid(U).name() << ">() 已调用"
              << std::endl;
  }
};

// =============================================================================
// 3. 依赖基类访问的 this->
// =============================================================================
// 依赖基类的成员在阶段 1 查找期间不会被找到。
// 通过 `this->` 或 `Base<T>::` 访问它们。

template <typename T>
struct Base {
  int    value     = 42;
  double coefficient = 3.14;

  void hello() { std::cout << "Base::hello()\n"; }

  enum { flag = 1 };
};

template <typename T>
struct Derived : Base<T> {
  void demo_this_access() {
    // this->value = 100;  // OK：使 value 成为依赖名
    // Base<T>::value = 100;  // OK：显式限定
    // value = 100;  // 错误：阶段 1 中找不到 value

    std::cout << "  this->value = " << this->value << std::endl;
    std::cout << "  this->coefficient = " << this->coefficient << std::endl;
    this->hello();

    // 对于枚举常量，不需要 this->（可以使用 Base<T>::）
    std::cout << "  Base<T>::flag = " << Base<T>::flag << std::endl;
  }
};

// =============================================================================
// 4. 模板参数的零初始化
// =============================================================================
// 对于内置类型，T() 或 T{} 执行零初始化。对于类类型，
// 它调用默认构造函数。始终初始化模板参数以避免未初始化值
// 导致的未定义行为。

template <typename T>
class SafeValue {
 public:
  // 默认构造函数：对基础类型进行零初始化
  SafeValue() : val_{} {}  // T{} 保证对 int、double 等进行零初始化

  explicit SafeValue(T const& v) : val_(v) {}
  explicit SafeValue(T&& v) : val_(std::move(v)) {}

  T const& get() const { return val_; }
  T&       get() { return val_; }

  // 确保 val_ 从不会被未初始化使用
  void reset() { val_ = T{}; }

 private:
  T val_;  // 成员不在此处初始化；在构造函数中完成
};

// 演示：T{} vs 未初始化
template <typename T>
T make_zero() {
  return T{};  // 零初始化：int 为 0，double 为 0.0，指针为 nullptr
}

template <typename T>
T make_uninitialized() {
  T val;  // 危险：对基础类型未初始化，对类类型默认初始化
  return val;  // 对于 int/double：不确定的值
}

// =============================================================================
// 5. .template、::template、->template 语法
// =============================================================================
// `template` 关键字可以出现在三种访问器上下文中。

template <typename T>
struct Outer {
  T inner;

  void demo_syntax() {
    // 1. .template -- 通过点进行成员访问
    inner.template method<int>();

    // 2. ->template -- 通过指针进行成员访问
    // （需要指针；这里仅用取地址演示语法）
    T* ptr = &inner;
    ptr->template method<double>();

    // 3. ::template -- 带作用域解析的限定名
    T::template nested_static<int>();
  }
};

struct WithTemplates {
  template <typename U>
  void method() {
    std::cout << "  method<" << typeid(U).name() << ">() 已调用" << std::endl;
  }

  template <typename U>
  static void nested_static() {
    std::cout << "  nested_static<" << typeid(U).name() << ">() 已调用"
              << std::endl;
  }
};

// =============================================================================
// 6. 默认模板实参（类模板）
// =============================================================================

template <typename T, typename Alloc = std::allocator<T>>
class SimpleContainer {
 public:
  using value_type = T;
  using allocator_type = Alloc;

  void push(T const& val) {
    data_.push_back(val);
  }

  T const& operator[](std::size_t i) const { return data_[i]; }
  std::size_t size() const { return data_.size(); }

 private:
  std::vector<T, Alloc> data_;
};

// =============================================================================
// 7. 非类型 Auto 和模板内的模板
// =============================================================================

// C++17 auto 非类型 + 类模板内 using
template <typename T, auto DefaultValue>
struct AutoDefault {
  static constexpr auto default_value = DefaultValue;

  T val{static_cast<T>(DefaultValue)};

  T get() const { return val; }
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 05.1 章：棘手基础 ===\n" << endl;

  // --- 测试 1：依赖类型的 typename ---
  cout << "[测试 1] 使用 typename 的 ContainerTraits：" << endl;
  using VecInt = std::vector<int>;
  using Traits = ContainerTraits<VecInt>;
  static_assert(is_same_v<Traits::value_type, int>);
  static_assert(is_same_v<Traits::size_type, std::size_t>);
  cout << "  value_type = int, size_type = size_t：已确认" << endl;

  // --- 测试 2：成员模板的 template 关键字 ---
  cout << "\n[测试 2] 成员函数模板的 template 关键字：" << endl;
  MemberTemplateDemo<HasMemberTemplate> mtd;
  mtd.call_member_template();

  // --- 测试 3：依赖基类访问的 this-> ---
  cout << "\n[测试 3] 依赖基类访问的 this->：" << endl;
  Derived<int> d;
  d.demo_this_access();

  // --- 测试 4：零初始化 ---
  cout << "\n[测试 4] 模板类型的零初始化：" << endl;

  SafeValue<int> si;
  cout << "  SafeValue<int> 默认值：" << si.get()
       << "（应为 0）" << endl;
  assert(si.get() == 0);

  SafeValue<double> sd;
  cout << "  SafeValue<double> 默认值：" << sd.get()
       << "（应为 0.0）" << endl;
  assert(sd.get() == 0.0);

  SafeValue<int*> sp;
  cout << "  SafeValue<int*> 默认值：" << sp.get()
       << "（应为 nullptr）" << endl;
  assert(sp.get() == nullptr);

  int zm = make_zero<int>();
  double zd = make_zero<double>();
  cout << "  make_zero<int>() = " << zm << "（应为 0）" << endl;
  cout << "  make_zero<double>() = " << zd << "（应为 0.0）" << endl;
  assert(zm == 0);
  assert(zd == 0.0);

  // 注意：make_uninitialized<int>() 返回不确定的值！
  // 我们不调用它，因为使用该值是未定义行为。

  // --- 测试 5：.template、->template、::template ---
  cout << "\n[测试 5] .template / ->template / ::template 语法：" << endl;
  Outer<WithTemplates> outer;
  outer.demo_syntax();

  // --- 测试 6：默认模板实参 ---
  cout << "\n[测试 6] 默认模板实参：" << endl;
  SimpleContainer<int> cont;
  cont.push(10);
  cont.push(20);
  cout << "  SimpleContainer<int> 大小 = " << cont.size() << endl;
  cout << "  cont[0] = " << cont[0] << ", cont[1] = " << cont[1] << endl;
  assert(cont.size() == 2);

  // --- 测试 7：auto 非类型 ---
  cout << "\n[测试 7] auto 非类型默认值：" << endl;
  AutoDefault<int, 42> ad_int;
  AutoDefault<double, 3.14> ad_double;
  cout << "  AutoDefault<int, 42>::value = " << ad_int.get() << endl;
  cout << "  AutoDefault<double, 3.14>::value = " << ad_double.get() << endl;
  assert(ad_int.get() == 42);

  // --- 测试 8：边界情况 -- std::string 的 T{} ---
  cout << "\n[测试 8] std::string 的 T{}（默认构造函数）：" << endl;
  SafeValue<std::string> s_str;
  cout << "  SafeValue<string> 默认值：\"" << s_str.get()
       << "\"（应为空）" << endl;
  assert(s_str.get().empty());

  cout << "\n所有测试通过！" << endl;
  return 0;
}
