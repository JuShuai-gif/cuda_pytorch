// ============================================================================
// 01_template_terminology.cpp - 模板术语：替换 vs 实例化、模板类 vs 类模板
// ============================================================================
//
// 目的：
//   澄清 C++ 标准中关于模板的精确术语，这对理解编译器错误和
//   标准的措辞至关重要。
//
// 关键区别：
//
//   术语                   含义
//   ----------------------------------------------------------------
//   class template          生成类的蓝图
//   template class          显式实例化（在现代 C++ 中已弃用此术语；
//                           建议使用 "explicit instantiation"）
//   function template       生成函数的蓝图
//   template function       非标准术语；对生成的函数使用
//                           "function template specialization"
//
//   替换（Substitution）     用具体参数替换模板参数
//                           （发生在重载决议 / SFINAE 上下文中）
//
//   实例化（Instantiation）  从模板生成实际代码
//                           （在重载决议选出最佳候选后发生）
//
//   隐式实例化              编译器在首次使用时生成代码
//   显式实例化              程序员强制生成代码
//
//   特化（Specialization）  模板针对特定参数的特定版本
//                           （可以是隐式或显式的——完全特化或部分特化）
//
// 构建块图示：
//
//   template <typename T> struct Foo {};   ← "class template"（蓝图）
//   Foo<int>                               ← "implicit specialization"
//   template <> struct Foo<int> {};        ← "explicit (full) specialization"
//   template <typename T> struct Foo<T*> {}; ← "partial specialization"
//   template struct Foo<float>;            ← "explicit instantiation"
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <vector>
#include <string>

// ============================================================================
// 第 1 部分：类模板 vs 模板类
// ============================================================================

// --- 1a. 类模板（蓝图） ---
// "class template" 指的是模板本身 —— 参数化的蓝图，
// 而不是具体的类。
template <typename T>
class MyVector {
public:
    using value_type = T;
    explicit MyVector(std::size_t n) : size_(n) {}
    std::size_t size() const { return size_; }
private:
    std::size_t size_;
};

// --- 1b. 模板类（历史术语） ---
// 历史上，"template class" 指的是显式实例化：
//   template class MyVector<int>;  // 类模板的显式实例化
//
// C++ 标准已不再使用此术语。现代用法：
//   "类模板的显式实例化"

template class MyVector<double>;  // 显式实例化（不是 "template class"）

// --- 1c. 特化（隐式 vs 显式） ---
// 当你写 MyVector<int> 时，编译器隐式生成一个特化。
// 这是产生"隐式特化"的"隐式实例化"。

// 当你写 template<> struct MyVector<bool> { ... } 时，
// 这是"显式特化"（也称为"完全特化"）。

template <>
class MyVector<bool> {
    // MyVector 对 bool 的显式（完全）特化
    // 允许完全不同的实现
public:
    using value_type = bool;
    explicit MyVector(std::size_t n) : bits_((n + 63) / 64) {}
    std::size_t size() const { return bits_ * 64; }
private:
    std::size_t bits_;
};

// ============================================================================
// 第 2 部分：替换 vs 实例化
// ============================================================================
//
// 这两者经常混淆，但它们是根本不同的阶段：
//
//   替换（发生在模板参数推导过程中）：
//     - 模板参数被替换为具体类型/值
//     - 发生在函数签名的"直接上下文"中
//     - 失败 = SFINAE（替换失败不是错误）
//     - 编译器尝试所有可行的候选；替换失败会
//       静默地从重载集合中移除候选
//
//   实例化（在重载决议之后发生）：
//     - 编译器为选定的模板生成实际代码
//     - 发生在"实例化上下文"中（函数体等）
//     - 失败 = 硬错误（不是 SFINAE）
//     - 只有单一最佳匹配候选被实例化

// --- 2a. 替换上下文示例 ---

// 此 trait 检查 T 是否有嵌套的 ::value_type
template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {};
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                            此表达式在替换上下文中。
//                            如果 T::value_type 无效，SFINAE 会移除此部分特化。

// --- 2b. 实例化上下文示例 ---

template <typename T>
auto get_size(T const& obj) -> decltype(obj.size()) {
    // 返回类型 decltype(obj.size()) 在替换上下文中。
    // 如果 obj.size() 不存在，SFINAE 生效。
    //
    // 下面的函数体在实例化上下文中。
    // 如果 obj.size() 存在但其函数体有错误，则是硬错误。
    return obj.size();
}

// --- 2c. 演示：有效替换，无效实例化 ---

struct HasSize {
    std::size_t size() const { return 42UL; }
};

struct HasSizeButBroken {
    // size() 存在（所以替换成功），但函数体是无意义的
    // 这在实例化时会是硬错误，而不是 SFINAE
    int size() const {
        // return static_cast<int*>(nullptr); // 会是硬错误
        return 10;
    }
};

// ============================================================================
// 第 3 部分：推导 vs 替换
// ============================================================================

// 推导：从函数调用参数推导模板参数
// 替换：用推导的/显式的参数替换参数

template <typename T>
void deduce_and_substitute(T const& val) {
    // 第 1 步：推导
    //   deduce_and_substitute(42) → T 推导为 int
    //
    // 第 2 步：替换
    //   在整个函数签名和函数体中将 T 替换为 int
    //
    // 第 3 步：实例化
    //   为 void deduce_and_substitute<int>(int const&) 生成代码
    //
    std::cout << "Value: " << val << "\n";
}

// ============================================================================
// 第 4 部分：编译期验证
// ============================================================================

// 验证类型 trait 工作正常
static_assert(has_value_type<MyVector<int>>::value,
    "MyVector<int> should have value_type");
static_assert(!has_value_type<int>::value,
    "int should NOT have value_type");

// 验证显式特化是不同的类型
static_assert(!std::is_same_v<MyVector<int>, MyVector<bool>>,
    "Explicit specialization must be a distinct type");

// 验证显式实例化创建了完整类型
static_assert(sizeof(MyVector<double>) > 0,
    "Explicitly instantiated MyVector<double> must be complete");

// ============================================================================
// MAIN
// ============================================================================

int main() {
    // --- 术语在实践中的体现 ---

    // "class template" → MyVector 是一个类模板
    MyVector<int> vi(100);

    // "function template" → deduce_and_substitute 是一个函数模板
    deduce_and_substitute(42);

    // "substitution" → has_value_type 替换 T=int，找到 ::value_type → true
    std::cout << std::boolalpha;
    std::cout << "has_value_type<int>: "
              << has_value_type<int>::value << "\n";
    std::cout << "has_value_type<MyVector<int>>: "
              << has_value_type<MyVector<int>>::value << "\n";

    // "instantiation" → 编译器现在为 get_size<HasSize> 生成代码
    HasSize hs;
    std::cout << "get_size(HasSize): " << get_size(hs) << "\n";

    // "explicit specialization" → MyVector<bool> 使用自定义实现
    MyVector<bool> vb(128);
    std::cout << "MyVector<bool>.size(): " << vb.size() << "\n";

    std::cout << "\n术语演示完成。\n";
    return 0;
}
