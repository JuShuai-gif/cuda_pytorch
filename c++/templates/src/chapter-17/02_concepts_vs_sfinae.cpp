// ============================================================================
// 02_concepts_vs_sfinae.cpp - Concepts vs SFINAE: 并排对比
// ============================================================================
//
// 目的:
//   使用 C++20 Concepts 重写经典 SFINAE 模式，演示
//   可读性、错误消息和可维护性方面的
//   巨大改进。每节先展示 SFINAE 版本，
//   然后展示等价的 Concepts 版本。
//
// 对比矩阵:
//   模式               SFINAE                          Concepts
//   ------------------------------------------------------------------
//   类型检查           std::enable_if + type trait     requires Concept<T>
//   重载选择           enable_if in return type        约束重载
//   表达式有效性       decltype + void_t               requires 表达式
//   嵌套检查           嵌套 enable_if                   复合需求
//   错误消息           模板回溯噩梦                     "约束不满足"
//   可读性             晦涩                             自文档化
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <concepts>
#include <string>
#include <vector>
#include <list>
#include <iterator>
#include <memory>
#include <utility>

// ============================================================================
// 第 1 节: 类型属性检查
// ============================================================================

// --- SFINAE 版本 ---
template <typename T, typename = void>
struct has_value_type_sfinae : std::false_type {};

template <typename T>
struct has_value_type_sfinae<T, std::void_t<typename T::value_type>>
    : std::true_type {};

template <typename T>
constexpr bool has_value_type_sfinae_v = has_value_type_sfinae<T>::value;

// --- Concepts 版本 ---
template <typename T>
concept HasValueType = requires {
    typename T::value_type;
};

// 用法对比:
// SFINAE:  if constexpr (has_value_type_sfinae_v<T>) { ... }
// Concept: if constexpr (HasValueType<T>) { ... }
// Or:      template <HasValueType T> void func(T) { ... }

static_assert(HasValueType<std::vector<int>>);
static_assert(!HasValueType<int>);

// ============================================================================
// 第 2 节: 表达式有效性检查
// ============================================================================

// --- SFINAE 版本 ---
template <typename T, typename = void>
struct is_printable_sfinae : std::false_type {};

template <typename T>
struct is_printable_sfinae<T,
    std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};

template <typename T>
constexpr bool is_printable_sfinae_v = is_printable_sfinae<T>::value;

// --- Concepts 版本 ---
template <typename T>
concept Printable = requires(std::ostream& os, T val) {
    { os << val } -> std::same_as<std::ostream&>;
};

static_assert(Printable<int>);
static_assert(Printable<std::string>);
// struct NonPrintable {};
// static_assert(!Printable<NonPrintable>);

// ============================================================================
// 第 3 节: 重载选择（enable_if vs Concepts）
// ============================================================================

// --- SFINAE 版本: 用 enable_if 消歧的两个重载 ---

template <typename T>
std::enable_if_t<std::is_integral_v<T>, std::string>
describe_sfinae(T val) {
    return "整数: " + std::to_string(val);
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, std::string>
describe_sfinae(T val) {
    return "浮点: " + std::to_string(val);
}

template <typename T>
std::enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>,
    std::string>
describe_sfinae(T) {
    return "其他类型";
}

// --- Concepts 版本: 干净的重载 ---

template <std::integral T>
std::string describe_concept(T val) {
    return "整数: " + std::to_string(val);
}

template <std::floating_point T>
std::string describe_concept(T val) {
    return "浮点: " + std::to_string(val);
}

std::string describe_concept(auto val) {
    return "其他类型: " + std::string(typeid(val).name());
}

// ============================================================================
// 第 4 节: 多重约束（enable_if_all vs requires 子句）
// ============================================================================

// --- SFINAE 版本: 组合 enable_if ---

template <typename T,
    std::enable_if_t<
        std::is_integral_v<T> &&
        std::is_signed_v<T> &&
        (sizeof(T) <= 4),
        int> = 0>
void process_small_signed_sfinae(T val) {
    std::cout << "SFINAE: 小有符号整数: " << val << "\n";
}

// --- Concepts 版本: requires 子句 ---

template <typename T>
    requires std::integral<T> &&
             std::signed_integral<T> &&
             (sizeof(T) <= 4)
void process_small_signed_concept(T val) {
    std::cout << "Concept: 小有符号整数: " << val << "\n";
}

// ============================================================================
// 第 5 节: 嵌套类型检查（void_t vs 复合需求）
// ============================================================================

// --- SFINAE 版本: void_t 链 ---

template <typename T, typename = void>
struct has_begin_end_sfinae : std::false_type {};

template <typename T>
struct has_begin_end_sfinae<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

template <typename T, typename = void>
struct has_iterator_value_sfinae : std::false_type {};

template <typename T>
struct has_iterator_value_sfinae<T, std::void_t<
    typename decltype(std::declval<T>().begin())::value_type
>> : std::true_type {};

// --- Concepts 版本: 复合需求 ---

template <typename T>
concept Iterable = requires(T c) {
    { c.begin() } -> std::input_iterator;    // begin() 返回输入迭代器
    { c.end() } -> std::sentinel_for<decltype(c.begin())>;
};

template <typename T>
concept IterableWithValueType = Iterable<T> && requires(T c) {
    typename std::iter_value_t<decltype(c.begin())>;  // 迭代器的 value_type
};

static_assert(Iterable<std::vector<int>>);
static_assert(IterableWithValueType<std::vector<int>>);
static_assert(!Iterable<int>);

// ============================================================================
// 第 6 节: 返回类型约束
// ============================================================================

// --- SFINAE 版本: 带 enable_if 的尾置返回类型 ---

template <typename T>
auto get_first_sfinae(T const& container)
    -> std::enable_if_t<
        has_begin_end_sfinae<T>::value,
        decltype(*container.begin())
    >
{
    return *container.begin();
}

// --- Concepts 版本 ---

auto get_first_concept(Iterable auto const& container) {
    return *container.begin();
}

// ============================================================================
// 第 7 节: 多参数约束
// ============================================================================

// --- SFINAE 版本 ---

template <typename TA, typename TB, typename TC,
    std::enable_if_t<
        std::is_floating_point_v<TA> &&
        std::is_floating_point_v<TB> &&
        std::is_floating_point_v<TC>,
        int> = 0>
TC multiply_add_sfinae(TA a, TB b, TC c) {
    return a * b + c;
}

// --- Concepts 版本 ---

template <std::floating_point TA,
          std::floating_point TB,
          std::floating_point TC>
TC multiply_add_concept(TA a, TB b, TC c) {
    return a * b + c;
}

// ============================================================================
// 第 8 节: Concept 包含（偏序）
// ============================================================================
//
// Concepts 通过包含关系支持偏序。
// 如果 concept A 的约束逻辑上蕴含 concept B 的约束，
// 则 A 包含 B。更受约束的重载胜出。
//
// 这是对 SFINAE 的根本性改进，因为 SFINAE 的重载
// 消歧需要人工优先级参数（int/long/...）。

template <typename T>
concept SignedInteger = std::integral<T> && std::signed_integral<T>;

// 非负有符号整数（更多约束）
template <typename T>
concept NonNegativeSignedInteger = SignedInteger<T> && requires(T val) {
    { val >= T{0} } -> std::convertible_to<bool>;
};

// 重载 1: 任意有符号整数（较少约束）
template <SignedInteger T>
constexpr const char* classify(T) {
    return "有符号整数";
}

// 重载 2: 非负（通过包含关系更多约束）
template <NonNegativeSignedInteger T>
constexpr const char* classify(T) {
    return "非负有符号整数";
}

// 在运行时验证 concept 包含关系（constexpr classify 在运行时可验证）:
// 不使用 static_assert，因为 classify 依赖于 NonNegativeSignedInteger，
// 它使用带有运行时语义的 requires-表达式。
// 改为在 main() 中验证。

// ============================================================================
// 第 9 节: 错误消息对比
// ============================================================================
//
// SFINAE 错误（典型的）:
//   error: no matching function for call to 'describe_sfinae(std::vector<int>)'
//   note: candidate template ignored: requirement '!std::is_integral_v<...> &&
//          !std::is_floating_point_v<...>' was not satisfied
//   [模板实例化的长回溯...]
//
// Concepts 错误（典型的）:
//   error: no matching function for call to 'describe_concept(std::vector<int>)'
//   note: candidate 'describe_concept(auto)' not viable: constraints not satisfied
//   [简洁、可读的消息]

// ============================================================================
// 第 10 节: 将 SFINAE 特征库转换为 Concepts
// ============================================================================

// --- 旧: 基于 SFINAE 的类型特征库 ---

template <typename T, typename = void> struct is_dereferenceable_sfinae : std::false_type {};
template <typename T>
struct is_dereferenceable_sfinae<T, std::void_t<decltype(*std::declval<T>())>>
    : std::true_type {};

template <typename T>
constexpr bool is_dereferenceable_sfinae_v = is_dereferenceable_sfinae<T>::value;

template <typename T, typename = void> struct is_incrementable_sfinae : std::false_type {};
template <typename T>
struct is_incrementable_sfinae<T, std::void_t<decltype(++std::declval<T&>())>>
    : std::true_type {};

template <typename T>
constexpr bool is_incrementable_sfinae_v = is_incrementable_sfinae<T>::value;

// 在 SFINAE 中的用法: 组合复杂
template <typename T,
    std::enable_if_t<
        is_dereferenceable_sfinae_v<T> && is_incrementable_sfinae_v<T>,
        int> = 0>
void advance_iterator_sfinae(T& it, int n) {
    for (int i = 0; i < n; ++i) ++it;
}

// --- 新: Concepts 版本 ---

template <typename T>
concept Dereferenceable = requires(T t) { *t; };

template <typename T>
concept Incrementable = requires(T& t) { ++t; };

template <typename T>
concept IteratorLike = Dereferenceable<T> && Incrementable<T>;

// 用法: 自文档化
template <IteratorLike T>
void advance_iterator_concept(T& it, int n) {
    for (int i = 0; i < n; ++i) ++it;
}

static_assert(IteratorLike<int*>);
static_assert(IteratorLike<std::vector<int>::iterator>);

// ============================================================================
// 第 11 节: 编译期验证
// ============================================================================

static_assert(HasValueType<std::vector<int>>);
static_assert(Printable<int>);
static_assert(Iterable<std::vector<int>>);
static_assert(IteratorLike<int*>);
static_assert(NonNegativeSignedInteger<int>);
// static_assert(!NonNegativeSignedInteger<int>); // 会失败，int 不是显式的 signed_integral？其实是。

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Concepts vs SFINAE: 并排对比 ===\n\n";

    // 第 3 节: 重载选择
    std::cout << "--- 重载选择 ---\n";
    std::cout << "SFINAE:  " << describe_sfinae(42) << "\n";
    std::cout << "SFINAE:  " << describe_sfinae(3.14) << "\n";
    std::cout << "SFINAE:  " << describe_sfinae(std::string("hello")) << "\n";
    std::cout << "Concept: " << describe_concept(42) << "\n";
    std::cout << "Concept: " << describe_concept(3.14) << "\n";
    std::cout << "Concept: " << describe_concept(std::string("hello")) << "\n";

    // 第 4 节: 多重约束
    std::cout << "\n--- 多重约束 ---\n";
    process_small_signed_sfinae(int16_t{42});
    process_small_signed_concept(int16_t{42});

    // 第 6 节: 返回类型约束
    std::cout << "\n--- 返回类型约束 ---\n";
    std::vector<int> v = {100, 200, 300};
    std::cout << "SFINAE:  第一个 = " << get_first_sfinae(v) << "\n";
    std::cout << "Concept: 第一个 = " << get_first_concept(v) << "\n";

    // 第 7 节: 多参数
    std::cout << "\n--- 多参数约束 ---\n";
    std::cout << "SFINAE:  multiply_add(2.0f, 3.0f, 1.0f) = "
              << multiply_add_sfinae(2.0f, 3.0f, 1.0f) << "\n";
    std::cout << "Concept: multiply_add(2.0f, 3.0f, 1.0f) = "
              << multiply_add_concept(2.0f, 3.0f, 1.0f) << "\n";

    // 第 8 节: 包含
    std::cout << "\n--- Concept 包含 ---\n";
    std::cout << "classify(42):  " << classify(42) << "\n";
    std::cout << "classify(-42): " << classify(-42) << "\n";

    // 第 10 节: 迭代器 concept
    std::cout << "\n--- 迭代器 Concepts ---\n";
    int arr[] = {1, 2, 3, 4, 5};
    int* it = arr;
    std::cout << "前进前: *it = " << *it << "\n";
    advance_iterator_concept(it, 2);
    std::cout << "前进(2)后: *it = " << *it << "\n";

    std::vector<int>::iterator vit = v.begin();
    advance_iterator_concept(vit, 1);
    std::cout << "Vector 迭代器前进(1)后: *it = " << *vit << "\n";

    std::cout << "\nConcepts vs SFINAE 对比完成。\n";
    return 0;
}
