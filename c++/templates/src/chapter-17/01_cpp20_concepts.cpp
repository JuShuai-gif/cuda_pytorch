// ============================================================================
// 01_cpp20_concepts.cpp - C++20 Concepts: requires 表达式、
//                           Concept 定义 与 约束模板
// ============================================================================
//
// 目的:
//   C++20 Concepts 的全面介绍 —
//   那个期待已久的、用可读的一流语言支持
//   取代大量 SFINAE 的特性。
//
// 关键特性:
//   1. concept 定义               - 命名的约束集
//   2. requires 表达式            - 测试有效性的 bool 表达式
//   3. requires 子句              - 模板参数上的约束
//   4. 约束 auto                  - 缩写函数模板
//   5. Concepts 的偏序            - 更受约束的胜出
//   6. 复合需求                   - { expr } -> concept
//   7. 嵌套需求                   - requires requires
//
// 为什么用 Concepts 替代 SFINAE:
//   - 更好的错误消息（约束失败，而非替换失败）
//   - 可读代码（concept 名称记录意图）
//   - 基于约束排序的重载决议
//   - 无需模板技巧的编译期求值
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <concepts>
#include <string>
#include <vector>
#include <list>
#include <iterator>
#include <ranges>

// ============================================================================
// 第 1 节: 基本 Concept 定义
// ============================================================================

/// \brief Concept: 类型可哈希（有 std::hash 特化）。
/// 使用 requires 表达式检查 std::hash<T>{}(val) 的有效性。
template <typename T>
concept Hashable = requires(T val) {
    { std::hash<T>{}(val) } -> std::convertible_to<std::size_t>;
};
// 复合需求 { expr } -> concept 同时检查:
//   1. 表达式有效
//   2. 结果类型满足 concept

static_assert(Hashable<int>);
static_assert(Hashable<std::string>);
static_assert(!Hashable<std::vector<int>>);  // 没有 std::hash<vector<int>>

/// \brief Concept: 类型是数值（整数或浮点）。
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

static_assert(Numeric<int>);
static_assert(Numeric<double>);
static_assert(!Numeric<std::string>);
static_assert(!Numeric<void*>);

/// \brief Concept: 类型可相加（有 operator+）。
template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;   // a + b 必须返回 T
};

static_assert(Addable<int>);
static_assert(Addable<std::string>);  // string + string 返回 string
static_assert(!Addable<std::vector<int>>);

// ============================================================================
// 第 2 节: requires 表达式 vs requires 子句
// ============================================================================
//
// requires 表达式:     requires (params) { requirements... }
//   - 在编译期求值为 bool
//   - 用于 concept 定义和 static_assert
//
// requires 子句:       template <typename T> requires Concept<T>
//   - 约束模板声明
//   - 出现在模板参数列表之后或函数签名之后

// --- 2a. requires 表达式（独立） ---
constexpr bool int_is_hashable = requires {
    std::hash<int>{}(std::declval<int>());
};
static_assert(int_is_hashable);

// --- 2b. requires 子句（约束模板） ---

template <typename T>
    requires Numeric<T>          // requires 子句
T generic_square(T val) {
    return val * val;
}

// 等价: 尾置 requires 子句
template <typename T>
T generic_cube(T val) requires Numeric<T> {
    return val * val * val;
}

// --- 2c. 缩写函数模板（约束 auto） ---
auto generic_double(Numeric auto val) {
    return val * 2;
}

// ============================================================================
// 第 3 节: 标准库 Concepts
// ============================================================================

// C++20 <concepts> 提供:
//   std::same_as<T, U>          - T 和 U 是相同类型
//   std::derived_from<D, B>    - D 派生自 B
//   std::convertible_to<F, T>   - F 可隐式转换为 T
//   std::integral<T>            - T 是整数类型
//   std::signed_integral<T>     - T 是有符号整数类型
//   std::unsigned_integral<T>   - T 是无符号整数类型
//   std::floating_point<T>      - T 是浮点类型
//   std::movable<T>             - T 可移动构造和移动赋值
//   std::copyable<T>            - T 可拷贝
//   std::semiregular<T>         - T 是半正规的
//   std::regular<T>             - T 是正规的
//   std::invocable<F, Args...>  - F 可用 Args... 调用
//   std::predicate<F, Args...>  - F 是谓词（返回 bool）

// --- 3a. 在函数模板中使用标准 concepts ---

template <std::integral T>
constexpr T gcd(T a, T b) {
    while (b != 0) {
        T t = b;
        b = a % b;
        a = t;
    }
    return a;
}

template <std::floating_point T>
T safe_divide(T a, T b) {
    return (b != T{0}) ? (a / b) : T{0};
}

static_assert(gcd(12, 8) == 4);
static_assert(gcd(17, 5) == 1);

// ============================================================================
// 第 4 节: 复合需求
// ============================================================================

/// \brief Concept: T 是类容器的。
/// 使用复合需求检查方法签名。
template <typename T>
concept ContainerLike = requires(T c, T const cc, std::size_t n) {
    typename T::value_type;                        // 有 value_type
    typename T::iterator;                          // 有 iterator
    { c.begin() } -> std::same_as<typename T::iterator>;
    { c.end() } -> std::same_as<typename T::iterator>;
    { cc.size() } -> std::convertible_to<std::size_t>;
    { c.empty() } -> std::convertible_to<bool>;
    // 嵌套需求: value_type 必须是可移动的
    requires std::movable<typename T::value_type>;
};

static_assert(ContainerLike<std::vector<int>>);
static_assert(ContainerLike<std::list<double>>);
static_assert(!ContainerLike<int>);

/// \brief 由 ContainerLike 约束的函数。
template <ContainerLike C>
auto first_element(C const& container) -> typename C::value_type {
    return *container.begin();
}

// ============================================================================
// 第 5 节: Concept 细化（Concept 层次结构）
// ============================================================================

/// \brief Concept 细化: RandomAccessContainer 是更受约束的
/// ContainerLike（增加了 operator[]）。
template <typename T>
concept RandomAccessContainer = ContainerLike<T> && requires(T c, std::size_t i) {
    { c[i] } -> std::convertible_to<typename T::value_type>;
};

static_assert(RandomAccessContainer<std::vector<int>>);
static_assert(!RandomAccessContainer<std::list<int>>);  // list 没有 operator[]

// ============================================================================
// 第 6 节: 带 Concepts 的重载决议
// ============================================================================
//
// 当多个约束重载匹配时，更受约束的那个
// 被优先选择。这是类模板特化偏序的
// concept 等价物。

/// \brief 重载 1: 任意 ContainerLike（较少约束）。
template <ContainerLike C>
void describe_container(C const& c) {
    std::cout << "类容器（大小=" << c.size() << "）\n";
}

/// \brief 重载 2: RandomAccessContainer（更多约束）。
template <RandomAccessContainer C>
void describe_container(C const& c) {
    std::cout << "随机访问容器（大小=" << c.size()
              << ", 首元素=" << c[0] << "）\n";
}

// 当两者都匹配时，更受约束的（RandomAccessContainer）胜出。

// ============================================================================
// 第 7 节: requires requires — 双重 requires
// ============================================================================
//
// "requires requires" 看起来奇怪，但有清晰的解释:
//   - 第一个 "requires": 引入 requires 子句
//   - 第二个 "requires": 引入 requires 表达式
//
// 替代方案: 直接用命名 concept。

// 不命名 concept: requires requires
template <typename T>
    requires requires(T a, T b) { a + b; }  // requires 子句 + requires 表达式
T add_two(T a, T b) {
    return a + b;
}

// 使用命名 concept（推荐）:
template <typename T>
    requires Addable<T>
T add_two_better(T a, T b) {
    return a + b;
}

// ============================================================================
// 第 8 节: auto 作为 Concept 占位符
// ============================================================================

// C++20 允许在变量声明中使用带 concepts 的 auto:
void auto_concept_demo() {
    Numeric auto x = 42;        // x 必须是数值型
    Numeric auto y = 3.14;      // y 必须是数值型
    // Numeric auto z = "hello"; // ❌ 错误: const char* 不是数值型

    auto result = x + y;
    std::cout << "auto concept: " << x << " + " << y << " = " << result << "\n";
}

// ============================================================================
// 第 9 节: GPU 相关类型的自定义 Concept
// ============================================================================

/// \brief Concept: 类型是有效的 GEMM 元素类型。
/// 在 CUTLASS 中，这会约束模板参数为
/// 仅有硬件加速的类型。
template <typename T>
concept GemmElementType =
    std::same_as<T, float> ||
    std::same_as<T, double> ||
    std::same_as<T, int> ||
    std::same_as<T, unsigned short>;  // fp16 占位符

/// \brief Concept: 类型是累加器类型（比元素更宽）。
template <typename T>
concept AccumulatorType =
    std::same_as<T, float> ||
    std::same_as<T, double> ||
    std::same_as<T, int>;

// 使用 concepts 约束 GEMM 函数
template <GemmElementType TA, GemmElementType TB, AccumulatorType TC>
void constrained_gemm(TA const* A, TB const* B, TC* C, int M, int N, int K) {
    std::cout << "约束 GEMM: "
              << typeid(TA).name() << " x "
              << typeid(TB).name() << " -> "
              << typeid(TC).name() << "\n";
}

// ============================================================================
// 第 10 节: 编译期验证
// ============================================================================

// Hashable
static_assert(Hashable<int>);
static_assert(Hashable<std::string>);
static_assert(!Hashable<std::vector<int>>);

// Numeric
static_assert(Numeric<int>);
static_assert(Numeric<float>);
static_assert(Numeric<double>);
static_assert(!Numeric<std::string>);

// ContainerLike
static_assert(ContainerLike<std::vector<int>>);
static_assert(ContainerLike<std::string>);
static_assert(!ContainerLike<int>);

// RandomAccessContainer
static_assert(RandomAccessContainer<std::vector<int>>);
static_assert(!RandomAccessContainer<std::list<int>>);

// GemmElementType
static_assert(GemmElementType<float>);
static_assert(!GemmElementType<std::string>);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== C++20 Concepts ===\n\n";

    // 第 2 节: 约束模板
    std::cout << "--- 约束模板 ---\n";
    std::cout << "generic_square(5) = " << generic_square(5) << "\n";
    std::cout << "generic_cube(3.0) = " << generic_cube(3.0) << "\n";
    std::cout << "generic_double(7) = " << generic_double(7) << "\n";

    // 第 3 节: 标准 concepts
    std::cout << "\n--- 标准 Concepts ---\n";
    std::cout << "gcd(36, 24) = " << gcd(36, 24) << "\n";
    std::cout << "safe_divide(10.0, 3.0) = " << safe_divide(10.0, 3.0) << "\n";
    std::cout << "safe_divide(5.0, 0.0) = " << safe_divide(5.0, 0.0) << "\n";

    // 第 4 节: ContainerLike
    std::cout << "\n--- ContainerLike ---\n";
    std::vector<int> v = {10, 20, 30};
    std::cout << "first_element(vector): " << first_element(v) << "\n";

    // 第 6 节: 重载决议
    std::cout << "\n--- 重载决议 ---\n";
    std::vector<int> rv = {1, 2, 3};
    std::list<int>   ll = {4, 5, 6};
    describe_container(rv);  // RandomAccessContainer 胜出
    describe_container(ll);  // ContainerLike（非随机访问）

    // 第 8 节: auto concept
    std::cout << "\n--- Auto Concept ---\n";
    auto_concept_demo();

    // 第 9 节: GemmElementType
    std::cout << "\n--- GEMM 元素 Concepts ---\n";
    constrained_gemm((float*)nullptr, (float*)nullptr, (float*)nullptr, 0, 0, 0);
    constrained_gemm((unsigned short*)nullptr, (unsigned short*)nullptr,
                     (float*)nullptr, 0, 0, 0);

    std::cout << "\nC++20 Concepts 演示完成。\n";
    return 0;
}
