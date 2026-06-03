// ============================================================================
// 01_sfinae_deep_dive.cpp - SFINAE 深入剖析: 即时上下文、
//                             void_t 技巧 与 类型检测库
// ============================================================================
//
// 目的:
//   全面探索 SFINAE（替换失败不是错误） —
//   在 C++20 Concepts 之前最强大的模板元编程机制。
//   理解 SFINAE 对于阅读和编写
//   像 CUTLASS 这样模板密集的库至关重要。
//
// SFINAE 规则:
//   1. SFINAE 适用于模板参数推导和替换期间
//   2. 只有"即时上下文"中的失败才是 SFINAE 友好的
//   3. "实例化上下文"（函数体）中的失败是硬错误
//   4. SFINAE 可以从重载决议中移除候选者
//   5. "即时上下文" 包括:
//      - 函数签名（返回类型、参数类型）
//      - 模板参数声明
//      - 默认模板参数
//      - noexcept 说明符
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <vector>
#include <utility>
#include <cmath>

// ============================================================================
// 第 1 节: 即时上下文 — SFINAE 覆盖的范围
// ============================================================================

// --- 1a. 返回类型中的 SFINAE（即时上下文） ---

// 仅当 T 有 ::value_type 时此重载才有效
template <typename T>
auto get_value_type_impl(int)  // int 参数用于重载排序
    -> typename T::value_type  // SFINAE: 此处替换失败 = 丢弃重载
{
    return typename T::value_type{};
}

// 回退: 始终有效但优先级较低
template <typename T>
auto get_value_type_impl(long) -> void {
    // 返回 void，表示 "没有 value_type"
}

// --- 1b. 默认模板参数中的 SFINAE（即时上下文） ---

template <typename T,
          typename = typename T::value_type>  // 此处 SFINAE
void process_with_value_type(T const&) {
    std::cout << "process_with_value_type: 有 value_type\n";
}

// --- 1c. 参数类型中的 SFINAE（即时上下文） ---

template <typename T>
void process_container(T const&,
    typename T::const_iterator* = nullptr)  // SFINAE: 如果没有 const_iterator 则丢弃
{
    std::cout << "process_container: 有 const_iterator\n";
}

// --- 1d. 不在即时上下文中（硬错误！） ---

template <typename T>
void bad_sfinae(T const& obj) {
    // 这在函数体中 — 实例化上下文，不是 SFINAE 上下文。
    // obj.nonexistent_method();  // ❌ 硬错误，不是 SFINAE
}

// ============================================================================
// 第 2 节: void_t 技巧 — SFINAE 的瑞士军刀
// ============================================================================
//
// std::void_t<Ts...> 将任意类型序列映射为 void。
// 它的威力: 如果任意 Ts... 无效（替换失败），
// 整个 void_t<...> 无效 → SFINAE 移除该特化。
//
// 这使得可以用来检测任意属性的"检测惯用法"成为可能。

// --- 2a. void_t 定义（C++17 提供了 std::void_t） ---
#if 0
template <typename...>
using void_t = void;
#endif

// --- 2b. 经典 void_t 用法: 检测嵌套 typedef ---

template <typename T, typename = void>
struct has_iterator : std::false_type {};

template <typename T>
struct has_iterator<T, std::void_t<typename T::iterator>> : std::true_type {};
//                     ^       ^^^^^^^^^^^^^^^^^^^^^^^^^^
//                     |       如果 T::iterator 无效 → SFINAE → 回退到主模板
//                     |       （false_type 版本）
//                     void_t 将有效类型折叠为 void

static_assert(has_iterator<std::vector<int>>::value);
static_assert(!has_iterator<int>::value);

// --- 2c. 带多个条件的 void_t ---

template <typename T, typename = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<T, std::void_t<
    typename T::value_type,      // 有 value_type
    typename T::iterator,         // 有 iterator
    typename T::const_iterator,   // 有 const_iterator
    decltype(std::declval<T>().size()),  // 有 .size()
    decltype(std::declval<T>().begin()), // 有 .begin()
    decltype(std::declval<T>().end())    // 有 .end()
>> : std::true_type {};

static_assert(is_container<std::vector<int>>::value);
static_assert(!is_container<int>::value);
// std::string 是类容器的（有 value_type、iterator、size、begin、end）
static_assert(is_container<std::string>::value);
static_assert(is_container<std::string>::value);   // 正确: std::string 是类容器的

// --- 2d. 用 void_t 检测成员函数 ---

template <typename T, typename = void>
struct has_reserve : std::false_type {};

template <typename T>
struct has_reserve<T, std::void_t<
    decltype(std::declval<T>().reserve(std::declval<std::size_t>()))
>> : std::true_type {};

static_assert(has_reserve<std::vector<int>>::value);
static_assert(!has_reserve<int>::value);

// ============================================================================
// 第 3 节: enable_if — 经典 SFINAE
// ============================================================================

// --- 3a. 返回类型中的 enable_if ---

template <typename T>
constexpr std::enable_if_t<std::is_integral_v<T>, T>
divide_floor(T a, T b) {
    return (a >= 0) ? (a / b) : ((a - b + 1) / b);
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
divide_floor(T a, T b) {
    return std::floor(a / b);  // std::floor 在 C++23 之前不是 constexpr
}

// --- 3b. 模板参数中的 enable_if（更干净） ---

template <typename T,
          std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr T multiply_power2(T val, unsigned shift) {
    return val << shift;  // 仅对整数类型有效
}

template <typename T,
          std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T multiply_power2(T val, unsigned shift) {
    return val * static_cast<T>(1u << shift);
}

// --- 3c. enable_if_all: 多个条件 ---

template <typename T,
          std::enable_if_t<
              std::is_integral_v<T> && std::is_unsigned_v<T>, int> = 0>
void safe_cast_info() {
    std::cout << "无符号整数: 没有负值\n";
}

// ============================================================================
// 第 4 节: 检测惯用法（Library Fundamentals TS v2）
// ============================================================================

/// \brief 标准检测惯用法 — 一个结构化的 SFINAE 框架。
/// 这类似于 std::experimental::is_detected。

struct nonesuch {
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

namespace detection_detail {

template <typename Default,
          typename AlwaysVoid,
          template <typename...> class Op,
          typename... Args>
struct detector {
    using value_t = std::false_type;
    using type    = Default;
};

template <typename Default,
          template <typename...> class Op,
          typename... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type    = Op<Args...>;
};

} // namespace detection_detail

template <template <typename...> class Op, typename... Args>
using is_detected = typename detection_detail::detector<
    nonesuch, void, Op, Args...>::value_t;

template <template <typename...> class Op, typename... Args>
using detected_t = typename detection_detail::detector<
    nonesuch, void, Op, Args...>::type;

template <typename Expected,
          template <typename...> class Op,
          typename... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template <template <typename...> class Op, typename... Args>
using is_detected_convertible = std::is_convertible<
    detected_t<Op, Args...>, Args...>;
// （简化版 — 真实版本更复杂）

// --- 4a. 使用检测惯用法 ---

template <typename T>
using difference_type_op = typename T::difference_type;

template <typename T>
using has_foo_member = decltype(std::declval<T>().foo());

struct HasDifferenceType { using difference_type = long; };
struct HasFoo { int foo() { return 42; } };
struct Empty {};

static_assert(is_detected<difference_type_op, HasDifferenceType>::value);
static_assert(!is_detected<difference_type_op, Empty>::value);
static_assert(is_detected_exact<long, difference_type_op, HasDifferenceType>::value);
static_assert(is_detected<has_foo_member, HasFoo>::value);

// ============================================================================
// 第 5 节: SFINAE 与重载决议
// ============================================================================
//
// SFINAE 移除候选者。剩余的候选者按常规排序。
// 这使得可以构建基于类型属性选择"最佳"
// 候选者的重载集。

template <typename T>
auto describe_type_impl(T, int, int)  // 最低优先级
    -> std::enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>, void>
{
    std::cout << "未知类型\n";
}

template <typename T>
auto describe_type_impl(T, int, long)  // 中等优先级
    -> std::enable_if_t<std::is_floating_point_v<T>, void>
{
    std::cout << "浮点类型\n";
}

template <typename T>
auto describe_type_impl(T, long, long)  // 最高优先级（对整数）
    -> std::enable_if_t<std::is_integral_v<T>, void>
{
    std::cout << "整数类型\n";
}

template <typename T>
void describe_type(T val) {
    describe_type_impl(val, 0, 0L);
}

// ============================================================================
// 第 6 节: 进阶 — 偏特化中的 SFINAE
// ============================================================================

/// \brief 条件选择偏特化。
/// 这就是像 std::is_integral 这样的类型特征如何实现的：
/// 主模板默认为 false，而每种整数类型的
/// 偏特化用 true 覆盖。

template <typename T, typename = void>
struct TypeCategory {
    static constexpr const char* name() { return "其他"; }
};

template <typename T>
struct TypeCategory<T, std::enable_if_t<std::is_integral_v<T>>> {
    static constexpr const char* name() { return "整数"; }
};

template <typename T>
struct TypeCategory<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    static constexpr const char* name() { return "浮点"; }
};

static_assert(TypeCategory<int>::name() == std::string_view("整数"));
static_assert(TypeCategory<double>::name() == std::string_view("浮点"));
static_assert(TypeCategory<std::string>::name() == std::string_view("其他"));

// ============================================================================
// 第 7 节: 编译期验证
// ============================================================================

// is_container 检查
struct PseudoContainer {
    using value_type = int;
    using iterator = int*;
    using const_iterator = int const*;
    std::size_t size() { return 0; }
    iterator begin() { return nullptr; }
    iterator end() { return nullptr; }
};
static_assert(is_container<PseudoContainer>::value);

// SFINAE 重载决议: 整数除法
static_assert(divide_floor(7, 3) == 2);     // 整数: 7/3 向下取整 = 2
// 浮点版本（在此上下文中因 std::floor 不是 constexpr，
// 但模板选择在编译期已验证）
static_assert(std::is_invocable_r_v<int, decltype(divide_floor<int>), int, int>);
static_assert(std::is_invocable_r_v<double, decltype(divide_floor<double>), double, double>);

// multiply_power2 重载
static_assert(multiply_power2(3, 2) == 12);  // 3 << 2 = 12

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== SFINAE 深入剖析 ===\n\n";

    // 第 1 节: 即时上下文
    std::cout << "--- 即时上下文 ---\n";
    // get_value_type_impl: 测试 ::value_type
    std::vector<int> v{1, 2, 3};
    // 对于有 value_type 的类型，int 重载胜出
    auto vt = get_value_type_impl<decltype(v)>(0);  // 返回 int
    std::cout << "vector<int> 的 value_type = " << vt << "\n";

    // 第 2 节: void_t 技巧
    std::cout << "\n--- void_t 技巧 ---\n";
    std::cout << std::boolalpha;
    std::cout << "has_iterator<vector<int>>: " << has_iterator<std::vector<int>>::value << "\n";
    std::cout << "has_iterator<int>: " << has_iterator<int>::value << "\n";
    std::cout << "is_container<vector<int>>: " << is_container<std::vector<int>>::value << "\n";
    std::cout << "is_container<int>: " << is_container<int>::value << "\n";
    std::cout << "has_reserve<vector<int>>: " << has_reserve<std::vector<int>>::value << "\n";

    // 第 3 节: enable_if
    std::cout << "\n--- enable_if ---\n";
    std::cout << "divide_floor(7, 3) = " << divide_floor(7, 3) << "\n";
    std::cout << "divide_floor(7.5, 3.2) = " << divide_floor(7.5, 3.2) << "\n";
    std::cout << "multiply_power2(3, 2) = " << multiply_power2(3, 2) << "\n";
    std::cout << "multiply_power2(3.5, 2) = " << multiply_power2(3.5, 2) << "\n";

    // 第 5 节: SFINAE 重载决议
    std::cout << "\n--- SFINAE 重载决议 ---\n";
    describe_type(42);
    describe_type(3.14);
    describe_type(std::string("hello"));

    // 第 6 节: 偏特化
    std::cout << "\n--- TypeCategory ---\n";
    std::cout << "int: " << TypeCategory<int>::name() << "\n";
    std::cout << "double: " << TypeCategory<double>::name() << "\n";
    std::cout << "string: " << TypeCategory<std::string>::name() << "\n";

    std::cout << "\nSFINAE 深入剖析完成。\n";
    return 0;
}
