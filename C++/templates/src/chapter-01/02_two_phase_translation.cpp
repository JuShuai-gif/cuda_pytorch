// =============================================================================
// 第 01.2 章：两阶段翻译（依赖名 vs 非依赖名）
//
// C++ 模板经历两个翻译阶段：
//
//   阶段 1（定义期 / 非依赖查找）：
//     - 在不了解模板参数 T 的情况下进行解析和语法检查。
//     - 不依赖 T 的名字立即解析。
//     - 非依赖代码中的错误在定义时捕获。
//
//   阶段 2（实例化期 / 依赖查找）：
//     - 当模板用具体类型实例化时。
//     - 依赖 T 的名字根据实际类型解析。
//     - 依赖名需要 `typename` 和 `template` 消歧义关键字。
//
// 本文件通过清晰的示例演示这两个阶段。
//
// 编译：g++ -std=c++20 -o 02_two_phase_translation 02_two_phase_translation.cpp
// =============================================================================

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// 第 1 节：非依赖名 vs 依赖名
// =============================================================================

// 一个带有嵌套类型和嵌套模板的辅助类
struct HelperA {
  using value_type = int;

  template <typename U>
  struct Nested {
    U val;
  };
};

template <typename T>
struct HelperB {
  using value_type = T;

  template <typename U>
  struct Nested {
    U val;
  };
};

// --- 示例：非依赖名（在阶段 1 解析）---
// 这里 HelperA::value_type 是具体类型，不依赖 T。
// 编译器在阶段 1 立即解析它。

template <typename T>
struct NonDependentDemo {
  // 阶段 1：HelperA 完全已知，不依赖 T -> 现在就解析
  using type = HelperA::value_type;  // OK：int

  // 如果 HelperA 拼写错误，错误会在阶段 1 被捕获：
  // using bad = HelperA_typo::value_type;  // 阶段 1 报错！
};

// --- 示例：依赖名（在阶段 2 解析）---
// 这里 HelperB<T>::value_type 依赖模板参数 T。
// 编译器在知道 T 之前无法解析它。
// 我们必须使用 `typename` 告诉编译器这是一个类型。

template <typename T>
struct DependentDemo {
  // 阶段 2：HelperB<T> 依赖 T -> 需要 'typename' 前缀
  // 如果没有 'typename'，编译器会将 value_type 当作静态成员或值，
  // 而非类型，从而导致编译错误。
  using type = typename HelperB<T>::value_type;  // 正确
  // using bad = HelperB<T>::value_type;         // 没有 typename 会报错！
};

// =============================================================================
// 第 2 节：`typename` 关键字
// =============================================================================
// 在表示类型的依赖限定名之前必须使用 `typename` 关键字。
// 它用于区分类型和非类型。
//
// 必须使用 typename 的三种场景：
//   1. 在依赖嵌套类型名之前
//   2. 在模板参数列表中（作为 'class' 的同义词）
//   3. 在依赖类型的 using 声明中

template <typename Container>
void print_first_element(Container const& c) {
  // 'typename' 告诉编译器 Container::value_type 是一个类型
  // 没有它："error: need 'typename' before ... because ... is a
  // dependent scope"
  typename Container::value_type const& first = c[0];
  std::cout << "第一个元素：" << first << std::endl;
}

// 即使在函数体内，依赖类型也可能需要 typename：
template <typename T>
void demonstrate_local_typename() {
  // 依赖类型的局部别名
  using elem_t = typename T::value_type;  // 需要 typename
  elem_t val{};
  std::cout << "sizeof(elem_t) = " << sizeof(val) << std::endl;
}

// =============================================================================
// 第 3 节：`template` 关键字
// =============================================================================
// 在调用成员模板时，依赖限定名之后必须使用 `template` 关键字。
// 没有它，'<' 会被解析为小于运算符。

template <typename T>
void demonstrate_template_keyword() {
  // T::Nested<int>  -- 没有 'template'，< 被解析为小于号
  // T::template Nested<int>  -- 正确：Nested 是成员模板

  typename T::template Nested<int> obj;  // typename 表示类型，template 表示成员模板
  obj.val = 42;
  std::cout << "嵌套 int 值：" << obj.val << std::endl;

  // 另一个常见模式：调用成员函数模板
  // obj.template foo<int>();  // <-- 成员函数模板调用前需要 template 关键字
}

// =============================================================================
// 第 4 节：两阶段查找错误（说明性）
// =============================================================================

// --- 非依赖错误：在阶段 1 捕获 ---
// 取消注释即可看到错误：
// template <typename T>
// void phase1_error() {
//   undefined_function();  // 阶段 1 报错：非依赖，立即解析
// }

// --- 依赖错误：仅在阶段 2 实例化时捕获 ---
// 编译器接受这个模板定义，因为 T::foo 对某些 T 可能有效。
// 只有在用缺少 'foo' 的类型实例化模板时错误才会出现。

template <typename T>
void phase2_error_candidate() {
  T obj;
  // obj.this_does_not_exist();  // 依赖名：在实例化时报错
}

// =============================================================================
// 第 5 节：依赖基类访问
// =============================================================================
// 当从依赖模板的基类继承时，基类成员是依赖名。
// 通过 this-> 或 Base<T>:: 访问它们。

template <typename T>
struct Base {
  int base_value = 100;
  void base_func() { std::cout << "Base::base_func()\n"; }
};

template <typename T>
struct Derived : Base<T> {
  void demo() {
    // 没有 'this->'，编译器在阶段 1 找不到 base_value，
    // 因为 Base<T> 依赖 T（它可能被特化而没有 base_value）。
    std::cout << "通过 this-> 访问 base_value = " << this->base_value << std::endl;

    // 替代方式：显式限定
    std::cout << "通过 Base<T>:: 访问 base_value = " << Base<T>::base_value
              << std::endl;

    this->base_func();
  }
};

// =============================================================================
// 第 6 节：SFINAE / Concepts 与两阶段查找的关系
// =============================================================================
// C++20 concepts 的约束本身在阶段 1 检查，
// 但函数体仍然遵循两阶段规则。

template <typename T>
concept HasValueType = requires {
  typename T::value_type;  // 简单 requirement：T 必须有 ::value_type
};

template <HasValueType T>
void concept_demo(T const&) {
  // 在函数体内，依赖名仍然需要 typename
  typename T::value_type v{};
  std::cout << "有 value_type，size = " << sizeof(v) << std::endl;
}

// =============================================================================
// 第 7 节：POI（实例化点）和 ADL（参数依赖查找）
// =============================================================================
// 阶段 2 查找要考虑实例化点（POI），即代码中模板首次以具体实参使用的位置。
// ADL（Koenig 查找）在阶段 2 中使用模板实参的关联命名空间执行。

namespace demo_ns {
struct Widget {
  int id = 99;
};

// 当 T = Widget 时，此函数通过 ADL 被找到
inline std::ostream& operator<<(std::ostream& os, Widget const& w) {
  return os << "Widget(id=" << w.id << ")";
}
}  // namespace demo_ns

template <typename T>
void adl_print(T const& obj) {
  // operator<< 在阶段 2 通过 ADL 被找到，因为它在与 T（demo_ns）相同的命名空间中。
  // 没有 ADL，这将无法编译。
  std::cout << "ADL 打印：" << obj << std::endl;
}

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 01.2 章：两阶段翻译 ===\n" << endl;

  // --- 第 1 节：非依赖 vs 依赖 ---
  cout << "[第 1 节] NonDependentDemo::type 是 int："
       << is_same_v<NonDependentDemo<void>::type, int> << endl;
  cout << "[第 1 节] DependentDemo::type 是 double："
       << is_same_v<DependentDemo<double>::type, double> << endl;

  // --- 第 2 节：typename ---
  vector<int> vec{10, 20, 30};
  print_first_element(vec);
  demonstrate_local_typename<vector<int>>();

  // --- 第 3 节：template 关键字 ---
  cout << "\n[第 3 节] 成员模板 'template' 关键字：" << endl;
  demonstrate_template_keyword<HelperB<float>>();

  // --- 第 5 节：依赖基类 ---
  cout << "\n[第 5 节] 依赖基类访问：" << endl;
  Derived<int> d;
  d.demo();

  // --- 第 6 节：concepts ---
  cout << "\n[第 6 节] Concepts + 两阶段：" << endl;
  vector<string> svec{"hello"};
  concept_demo(svec);

  // --- 第 7 节：ADL 和 POI ---
  cout << "\n[第 7 节] ADL 查找（阶段 2）：" << endl;
  demo_ns::Widget w;
  adl_print(w);  // operator<< 通过 ADL 在 demo_ns 中被找到

  cout << "\n所有演示完成！" << endl;
  return 0;
}
