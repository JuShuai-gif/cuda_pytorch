// ============================================================================
// 02_odr_pitfalls.cpp - ODR（单一定义规则）在模板中的陷阱
// ============================================================================
//
// 目的：
//   演示重度使用模板的代码库中常见的 ODR 违规，以及库设计中
//   惯用的解决方案。
//
// ODR 背景：
//   单一定义规则规定每个非内联函数、变量或类在整个程序中
//   必须恰好有一个定义。模板是隐式内联的，但其特化
//   仍然可能违反 ODR。
//
// 常见 ODR 陷阱：
//   1. 不同 TU 中的不同模板定义（宏守卫）
//   2. 模板类中的静态数据成员
//   3. 在头文件中定义的显式特化
//   4. 带静态局部变量的内联函数
//   5. 不同 TU 之间不同的编译期常量
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>

// ============================================================================
// 陷阱 1：条件编译导致不同定义
// ============================================================================
//
// 问题：两个翻译单元包含相同的头文件，但使用了不同的
// 预处理器定义，导致模板在每个 TU 中有不同的定义。
// 这是 ODR 违规。

// 模拟两个 TU 以不同方式定义 MY_CONFIG 时发生的情况：
// TU-A：#define MY_CONFIG 1 → 包含头文件 → 模板有一个定义
// TU-B：#define MY_CONFIG 0 → 包含头文件 → 模板有另一个定义
// → ODR 违规（未定义行为！）

// 坏模式（已注释 —— 如果启用将导致 ODR 违规）：
/*
template <typename T>
struct BadConfigurable {
# if MY_CONFIG
    static constexpr int value = 42;
# else
    static constexpr int value = 99;
# endif
};
*/

// 解决方案 1：使用模板特化替代预处理器
template <typename T, int ConfigVersion = 1>
struct GoodConfigurable;

template <typename T>
struct GoodConfigurable<T, 1> {
    static constexpr int value = 42;
};

template <typename T>
struct GoodConfigurable<T, 2> {
    static constexpr int value = 99;
};

// 解决方案 2：使用 if constexpr 配合 trait（单一定义，无 ODR 风险）
template <typename T>
struct BetterConfigurable {
    static constexpr int value =
        std::is_same_v<T, int> ? 42 : 99;
    // 无论哪个 TU，表达式始终相同 → 无 ODR 违规
};

// ============================================================================
// 陷阱 2：没有外部定义的静态数据成员
// ============================================================================
//
// 问题：在 C++17 inline 变量之前，static constexpr 成员没问题，
// 但非 constexpr 静态成员需要恰好在
// 一个 TU 中有外部定义。缺少之 = 未定义引用。
// 在头文件中没有 inline = ODR 违规。

template <typename T>
struct Counter {
    // C++17 inline static：在头文件中安全，无 ODR 问题
    inline static int instances = 0;

    // 前 C++17 风格：需要在一个 .cpp 中有外部定义
    // static int old_style_counter;

    Counter() { ++instances; }
    ~Counter() { --instances; }
};

// 前 C++17：此行必须恰好出现在一个翻译单元中
// template <typename T> int Counter<T>::old_style_counter = 0;

// ============================================================================
// 陷阱 3：头文件中的显式特化
// ============================================================================
//
// 问题：显式（完全）特化不是模板；
// 它是具体的类/函数。如果在没有 'inline' 的头文件中定义，
// 从多个 TU 包含会导致多个定义 → ODR 违规。

template <typename T>
struct Processor {
    static void process() {
        std::cout << "Generic implementation for unknown type\n";
    }
};

// 坏：头文件中没有 inline 的完全特化 → ODR 违规
// template <>
// struct Processor<int> {
//     static void process() { std::cout << "Int processor\n"; }
// };

// 好：在头文件中声明，在 .cpp 中定义
// 在头文件中：
template <>
struct Processor<int> {
    // static void process();  // 仅声明
};

// 恰好在一个 .cpp 文件中（此处模拟）：
// template <>
// void Processor<int>::process() { std::cout << "Int processor (defined once)\n"; }

// 更好：使用部分特化（仍然是模板，隐式内联）
template <typename T>
struct Processor<T*> {
    static void process() {
        std::cout << "Pointer specialization (partial, safe in header)\n";
    }
};

// 最佳（C++17）：将完全特化标记为 inline。
// 由于 process() 在上面被注释了，这里使用新的 inline static 函数。
template <>
struct Processor<double> {
    static inline void process() {
        std::cout << "Double processor (inline full specialization)\n";
    }
};

// ============================================================================
// 陷阱 4：通过不求值上下文导致的 ODR
// ============================================================================
//
// 问题：sizeof、decltype、noexcept 表达式是不求值的，
// 但如果它们引用了模板实例化，不同 TU
// 可能会看到不同的实例化。

// 安全：不求值上下文，无 ODR 问题（不需要定义）
template <typename T>
constexpr std::size_t size_of = sizeof(T);

// 危险：如果 UNIQUE_ID 在不同 TU 之间不同，这是 ODR 违规
// 每个 TU 得到 UniqueType<DIFFERENT_VALUE> 的不同实例化
// #define UNIQUE_ID __COUNTER__  // 每个 TU 不同 → ODR 违规
// using T1 = UniqueType<UNIQUE_ID>;

// 安全：使用 __LINE__？如果头文件在不同行被包含仍然危险。
// 最安全：使用显式管理的 ID 或依赖不依赖编译单元值的模板特化。

// ============================================================================
// 陷阱 5：函数模板显式特化的 ODR
// ============================================================================
//
// 与类模板特化类似，显式函数模板特化必须在
// 使用前声明，并恰好定义一次（或标记为 inline）。

template <typename T>
T identity(T val) { return val; }

// 显式特化的声明（在头文件中安全）
template <>
float identity<float>(float val);

// 定义必须恰好在一个 TU 中（或 inline）
template <>
inline float identity<float>(float val) { return val; }

// ============================================================================
// 解决方案：标准库模式
// ============================================================================
//
// CUTLASS 和 STL 使用以下模式来避免 ODR 问题：
//
// 1. 所有模板都是隐式内联的 → 在头文件中安全
// 2. 显式特化在头文件中声明，在 .cpp 中定义
// 3. 静态成员使用 C++17 'inline' 关键字
// 4. 条件代码使用 if constexpr，而非 #ifdef
// 5. 编译期常量是 constexpr（C++17+ 中 ODR 可用）

// ============================================================================
// 编译期验证
// ============================================================================

// 陷阱 1 的解决方案：无论配置如何都是单一定义
static_assert(GoodConfigurable<float, 1>::value == 42);
static_assert(GoodConfigurable<float, 2>::value == 99);
static_assert(BetterConfigurable<int>::value == 42);
static_assert(BetterConfigurable<double>::value == 99);

// 陷阱 2 的解决方案：inline static 可用
static_assert(std::is_same_v<decltype(Counter<int>::instances), int>);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== ODR 陷阱与解决方案 ===\n\n";

    // 陷阱 1：可配置模板（安全版本）
    std::cout << "GoodConfigurable<float,1>::value = "
              << GoodConfigurable<float, 1>::value << "\n";
    std::cout << "GoodConfigurable<float,2>::value = "
              << GoodConfigurable<float, 2>::value << "\n";

    // 陷阱 2：静态数据成员
    Counter<int> c1, c2, c3;
    std::cout << "Counter<int>::instances = "
              << Counter<int>::instances << "\n";

    // 陷阱 3：显式特化（使用 double 特化）
    Processor<double>::process();
    Processor<double*>::process();

    // 陷阱 5：函数模板特化
    std::cout << "identity<float>(3.14f) = "
              << identity<float>(3.14f) << "\n";

    std::cout << "\n所有 ODR 安全模式已演示。\n";
    return 0;
}
