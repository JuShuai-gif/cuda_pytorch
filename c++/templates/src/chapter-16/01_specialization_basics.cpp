// ============================================================================
// 01_specialization_basics.cpp - 全特化、偏特化与偏序规则
// ============================================================================
//
// 目的:
//   C++ 模板特化的全面演示:
//   全（显式）特化、偏特化以及
//   当多个候选者匹配时决定选择哪个特化的
//   偏序规则。
//
// 关键概念:
//   1. 全（显式）特化                  - template<> struct Foo<T> {}
//   2. 偏特化                          - template<T> struct Foo<T*> {}
//   3. 主模板                          - 未特化的模板
//   4. 偏序                            - 选择"最特化"的规则
//   5. 成员特化                        - 单个成员函数
//   6. 从特化继承                      - 使用特化作为策略
//
// 偏序规则:
//   - 一个偏特化比另一个"更特化"，如果
//     匹配第一个的每组参数也匹配第二个，
//     但反过来不成立。
//   - 编译器选择最特化的匹配偏特化。
//   - 如果两个偏特化不可比较（没有哪个比另一个
//     更特化），则程序有问题（歧义）。
//   - 全特化优先于偏特化。
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <vector>
#include <complex>

// ============================================================================
// 第 1 节: 全（显式）特化
// ============================================================================

/// \brief 主模板: 通用类型描述符。
template <typename T>
struct TypeDescriptor {
    static std::string describe() {
        return "通用类型: " + std::string(typeid(T).name());
    }
    static constexpr bool is_primary = true;
};

/// \brief int 的全特化。
template <>
struct TypeDescriptor<int> {
    static std::string describe() {
        return "类型: int（32 位有符号整数）";
    }
    static constexpr bool is_primary = false;
    static constexpr int kMin = -2147483648;
    static constexpr int kMax = 2147483647;
};

/// \brief double 的全特化。
template <>
struct TypeDescriptor<double> {
    static std::string describe() {
        return "类型: double（64 位 IEEE 754 浮点）";
    }
    static constexpr bool is_primary = false;
    static constexpr int kMantissaBits = 53;
    static constexpr int kExponentBits = 11;
};

/// \brief std::string 的全特化。
template <>
struct TypeDescriptor<std::string> {
    static std::string describe() {
        return "类型: std::string（动态字符序列）";
    }
    static constexpr bool is_primary = false;
};

// 注意: std::vector<T> 的全特化需要
// 提供所有模板参数（T、Allocator）:
template <>
struct TypeDescriptor<std::vector<int>> {
    static std::string describe() {
        return "类型: std::vector<int>";
    }
    static constexpr bool is_primary = false;
};

// ============================================================================
// 第 2 节: 偏特化
// ============================================================================

/// \brief 指针类型 T* 的偏特化。
template <typename T>
struct TypeDescriptor<T*> {
    static std::string describe() {
        return "指向: " + TypeDescriptor<T>::describe();
    }
    static constexpr bool is_primary = false;
};

/// \brief 引用类型 T& 的偏特化。
template <typename T>
struct TypeDescriptor<T&> {
    static std::string describe() {
        return "引用自: " + TypeDescriptor<T>::describe();
    }
    static constexpr bool is_primary = false;
};

/// \brief const T 的偏特化。
template <typename T>
struct TypeDescriptor<T const> {
    static std::string describe() {
        return "常量: " + TypeDescriptor<T>::describe();
    }
    static constexpr bool is_primary = false;
};

/// \brief 任意类型的 std::vector 的偏特化。
template <typename T, typename Alloc>
struct TypeDescriptor<std::vector<T, Alloc>> {
    static std::string describe() {
        return "std::vector 包含 " + TypeDescriptor<T>::describe();
    }
    static constexpr bool is_primary = false;
};

/// \brief std::complex<T> 的偏特化。
template <typename T>
struct TypeDescriptor<std::complex<T>> {
    static std::string describe() {
        return "复数<" + TypeDescriptor<T>::describe() + ">";
    }
    static constexpr bool is_primary = false;
};

// ============================================================================
// 第 3 节: 偏序规则
// ============================================================================

/// \brief 演示偏序: 哪个特化胜出？
///
/// 考虑这两个对类型进行分类的特征的偏特化:

// 主模板
template <typename T, typename = void>
struct TypeRank {
    static constexpr int value = 0;  // 通用
    static constexpr const char* name() { return "通用"; }
};

// 偏特化 1: 指针
template <typename T>
struct TypeRank<T*, void> {
    static constexpr int value = 1;  // 指针
    static constexpr const char* name() { return "指针"; }
};

// 偏特化 2: 指向常量的指针
template <typename T>
struct TypeRank<T const*, void> {
    static constexpr int value = 2;  // 指向常量的指针
    static constexpr const char* name() { return "指向常量的指针"; }
};

// 全特化: int*
template <>
struct TypeRank<int*, void> {
    static constexpr int value = 3;  // 全特化胜出
    static constexpr const char* name() { return "Int指针"; }
};

// --- 验证偏序 ---
// T const* 比 T* 更特化，因为:
//   - 如果某类型匹配 T const*，它也匹配 T*（带入 T=const U）
//   - 但反过来不行: int* 匹配 T* 但不匹配 T const*
static_assert(TypeRank<int*>::value == 3, "全特化胜出");
static_assert(TypeRank<int const*>::value == 2, "最特化的偏特化胜出");
static_assert(TypeRank<float*>::value == 1, "指针偏特化胜出");
static_assert(TypeRank<int>::value == 0, "主模板胜出");

// ============================================================================
// 第 4 节: 偏特化中的歧义
// ============================================================================
//
// 当两个偏特化不可比较时（没有哪个更特化），
// 程序是有问题的。本节展示一个无害的示例
//（被注释掉以避免编译错误）并解释如何解决它。

// 两个不兼容的偏特化:
// template <typename T> struct Ambiguous<T, T> { ... };      // 两个相同类型
// template <typename T> struct Ambiguous<T*, T*> { ... };    // 两个指针
// 对于 Ambiguous<int*, int*>，两者都匹配 → 歧义 → 编译错误

/// \brief 解决方案: 添加消歧约束。
/// 这就是 SFINAE 或 Concepts 的用武之地。
template <typename T, typename U, typename = void>
struct SafePair {
    static constexpr const char* name() { return "通用对"; }
};

// 仅当 T 和 U 是相同类型时匹配
template <typename T, typename U>
struct SafePair<T, U, std::enable_if_t<std::is_same_v<T, U>>> {
    static constexpr const char* name() { return "相同类型对"; }
};

// 仅当 T* 和 U*（都是指针）时匹配 — 但当 T==U 时不匹配
//（因为 enable_if 防止重叠）
template <typename T, typename U>
struct SafePair<T, U, std::enable_if_t<
    std::is_pointer_v<T> && std::is_pointer_v<U> &&
    !std::is_same_v<T, U>
>> {
    static constexpr const char* name() { return "不同指针对"; }
};

static_assert(std::string_view(SafePair<int, int>::name()) == "相同类型对");
static_assert(std::string_view(SafePair<int*, float*>::name()) == "不同指针对");
static_assert(std::string_view(SafePair<int, float>::name()) == "通用对");

// ============================================================================
// 第 5 节: 成员函数特化
// ============================================================================

template <typename T>
struct Calculator {
    static T add(T a, T b) {
        return a + b;  // 主模板: 使用 operator+
    }
};

/// \brief 仅对 float 特化 add()（而不是整个类）。
/// 这保持了 Calculator<float> 的其余部分与主模板相同。
template <>
float Calculator<float>::add(float a, float b) {
    // 针对 float 的自定义实现（例如带有 NaN 处理）
    if (a != a || b != b) return 0.0f;  // NaN 检查
    return a + b;
}

// 我们也可以为某个类型特化整个类:
template <>
struct Calculator<double> {
    static double add(double a, double b) {
        return a + b;
    }
    // double 版本可以有额外特定于 double 的成员
    static double sqrt_newton(double x) {
        double guess = x * 0.5;
        for (int i = 0; i < 10; ++i)
            guess = 0.5 * (guess + x / guess);
        return guess;
    }
};

// ============================================================================
// 第 6 节: 变量模板特化（C++14）
// ============================================================================

template <typename T>
constexpr bool is_numeric_v = false;

template <>
constexpr bool is_numeric_v<int> = true;

template <>
constexpr bool is_numeric_v<float> = true;

template <>
constexpr bool is_numeric_v<double> = true;

// 指针上的偏特化: false（指向数值的指针不是数值）
template <typename T>
constexpr bool is_numeric_v<T*> = false;

static_assert(is_numeric_v<int>);
static_assert(!is_numeric_v<int*>);
static_assert(!is_numeric_v<std::string>);

// ============================================================================
// 第 7 节: 非类型模板参数的特化
// ============================================================================

/// \brief 编译期数组描述符。
template <typename T, std::size_t N>
struct ArrayDescriptor {
    static std::string describe() {
        return "包含 " + TypeDescriptor<T>::describe()
            + " 的数组，共 " + std::to_string(N) + " 个元素";
    }
    static constexpr std::size_t size = N;
    using element_type = T;
};

/// \brief 大小为 0 的数组的偏特化。
template <typename T>
struct ArrayDescriptor<T, 0> {
    static std::string describe() {
        return "包含 " + TypeDescriptor<T>::describe() + " 的空数组";
    }
    static constexpr std::size_t size = 0;
    using element_type = T;
};

// int[4] 的全特化
template <>
struct ArrayDescriptor<int, 4> {
    static std::string describe() {
        return "4 元素 int 数组（SIMD 友好）";
    }
    static constexpr std::size_t size = 4;
    using element_type = int;
};

static_assert(ArrayDescriptor<float, 8>::size == 8);
static_assert(ArrayDescriptor<double, 0>::size == 0);

// ============================================================================
// 第 8 节: 编译期验证
// ============================================================================

// 全特化
static_assert(!TypeDescriptor<int>::is_primary);
static_assert(TypeDescriptor<float>::is_primary);  // 没有 float 的特化

// 偏特化选择
static_assert(!TypeDescriptor<int*>::is_primary);   // 选中 T* 偏特化
static_assert(!TypeDescriptor<int const*>::is_primary);  // T*（const 是 T 的一部分）

// 偏序
static_assert(TypeRank<float const*>::value == 2);  // 更特化

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 模板特化: 全特化、偏特化、偏序 ===\n\n";

    // 第 1 节: 全特化
    std::cout << "--- 全特化 ---\n";
    std::cout << TypeDescriptor<int>::describe() << "\n";
    std::cout << TypeDescriptor<double>::describe() << "\n";
    std::cout << TypeDescriptor<float>::describe() << "\n";  // 使用主模板

    // 第 2 节: 偏特化
    std::cout << "\n--- 偏特化 ---\n";
    std::cout << TypeDescriptor<int*>::describe() << "\n";
    std::cout << TypeDescriptor<float&>::describe() << "\n";
    std::cout << TypeDescriptor<int const>::describe() << "\n";
    std::cout << TypeDescriptor<std::vector<double>>::describe() << "\n";
    std::cout << TypeDescriptor<std::complex<float>>::describe() << "\n";

    // 第 3 节: 偏序
    std::cout << "\n--- 偏序 ---\n";
    std::cout << "TypeRank<int*>:          " << TypeRank<int*>::name()
              << " (值=" << TypeRank<int*>::value << ")\n";
    std::cout << "TypeRank<int const*>:    " << TypeRank<int const*>::name()
              << " (值=" << TypeRank<int const*>::value << ")\n";
    std::cout << "TypeRank<float*>:        " << TypeRank<float*>::name()
              << " (值=" << TypeRank<float*>::value << ")\n";
    std::cout << "TypeRank<int>:           " << TypeRank<int>::name()
              << " (值=" << TypeRank<int>::value << ")\n";

    // 第 4 节: 安全消歧
    std::cout << "\n--- 消歧 ---\n";
    std::cout << "SafePair<int,int>:       " << SafePair<int, int>::name() << "\n";
    std::cout << "SafePair<int*,float*>:   " << SafePair<int*, float*>::name() << "\n";
    std::cout << "SafePair<int,float>:     " << SafePair<int, float>::name() << "\n";

    // 第 5 节: 成员函数特化
    std::cout << "\n--- 成员特化 ---\n";
    std::cout << "Calculator<int>::add(3,4):    " << Calculator<int>::add(3, 4) << "\n";
    std::cout << "Calculator<float>::add(3,4):  " << Calculator<float>::add(3.0f, 4.0f) << "\n";
    std::cout << "Calculator<double>::sqrt(2):  " << Calculator<double>::sqrt_newton(2.0) << "\n";

    // 第 7 节: 非类型特化
    std::cout << "\n--- 非类型特化 ---\n";
    std::cout << ArrayDescriptor<int, 4>::describe() << "\n";
    std::cout << ArrayDescriptor<float, 8>::describe() << "\n";
    std::cout << ArrayDescriptor<double, 0>::describe() << "\n";

    std::cout << "\n特化基础演示完成。\n";
    return 0;
}
