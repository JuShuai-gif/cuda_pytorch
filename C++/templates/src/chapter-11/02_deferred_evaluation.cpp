// ============================================================================
// 02_deferred_evaluation.cpp - 延迟求值：identity、lazy_enable_if
//                               和类型计算惰性
// ============================================================================
//
// 目的：
//   演示模板元编程的关键技术 —— 延迟求值（DEFERRED EVALUATION）。
//   在模板元编程中，我们经常希望延迟类型计算的实例化，直到确信
//   它会成功 —— 否则会在 SFINAE 拯救我们之前触发硬错误。
//
// 关键模式：
//   1. identity<T>                     - 延迟类型计算
//   2. lazy_enable_if / lazy_conditional - 延迟布尔求值
//   3. 延迟别名模板                    - ::type 间接引用
//   4. 带延迟求值的 void_t             - 检测惯用法
//
// 为什么这很重要：
//   在 SFINAE 上下文中，替换失败必须发生在"直接上下文"中。
//   如果类型计算被急切求值（例如在类模板体内），失败会成为硬错误。
//   延迟求值将计算保持在替换上下文中。
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <vector>
#include <memory>

// ============================================================================
// 第 1 部分：问题 —— 急切求值导致硬错误
// ============================================================================

// --- 1a. 急切求值（对 SFINAE 来说是坏的） ---

template <typename T>
struct eager_value_type {
    // 这在类模板实例化时急切求值。
    // 如果 T::value_type 不存在，它是硬错误 —— 不是 SFINAE。
    using type = typename T::value_type;
};

// 如果 T 没有 ::value_type，这会导致硬编译错误：
// eager_value_type<int>::type  // 爆炸 💥

// --- 1b. 惰性求值（对 SFINAE 来说是好的） ---

template <typename T, typename = void>
struct lazy_value_type {
    // 默认：根本没有 ::type 成员
    // SFINAE 会简单地将此特化移出考虑范围
};

template <typename T>
struct lazy_value_type<T, std::void_t<typename T::value_type>> {
    // ::type 仅在 T::value_type 存在时才存在
    // 计算被延迟到 std::void_t<...> 的替换
    using type = typename T::value_type;
};

// 这是安全的 —— 对 int 没有错误：
// lazy_value_type<int>::type  // 根本不存在 → SFINAE

// ============================================================================
// 第 2 部分：identity<T> —— 最简单的延迟工具
// ============================================================================
//
// identity<T> 是一个简单的元函数，只返回 T。
// 它的威力来自于强制编译器在替换过程中更晚地求值。

// --- 2a. 标准 identity（C++20 有 std::type_identity） ---

template <typename T>
struct identity {
    using type = T;
};

template <typename T>
using identity_t = typename identity<T>::type;

// --- 2b. identity 为什么重要：强制依赖上下文 ---

template <typename T>
struct Container {
    using value_type = T;

    // 没有 identity，这会是非依赖名称，
    // 会立即被查找 → 潜在的硬错误
    using iterator = T*;  // 没问题，T 是参数

    // identity_t<...> 使编译器等到 T 已知
    template <typename U = T>
    using deferred_value = identity_t<U>;
};

// --- 2c. 经典用法：在 SFINAE 中防止过早求值 ---

// 坏：默认模板参数中的急切求值
template <typename T,
          typename = typename T::value_type>  // 如果 T 没有 value_type 则是硬错误
void bad_f(T const&) {}

// 好：通过 identity（或 void_t）延迟
template <typename T,
          typename = void>
void good_f(T const&) {}

template <typename T>
void good_f(T const&,
            std::void_t<typename T::value_type>* = nullptr)
{
    // 此重载仅在 T 有 ::value_type 时存在
}

// ============================================================================
// 第 3 部分：lazy_enable_if —— 条件类型选择
// ============================================================================
//
// std::enable_if 可用但冗长。lazy_enable_if 是一种将条件求值
// 延迟到 ::type 访问点的模式。

// --- 3a. 标准 enable_if（急切条件） ---

template <bool B, typename T = void>
struct lazy_enable_if {
    // 当 B 为 false 时没有 ::type
};

template <typename T>
struct lazy_enable_if<true, T> {
    using type = T;
};

template <bool B, typename T = void>
using lazy_enable_if_t = typename lazy_enable_if<B, T>::type;

// 这实际上与 std::enable_if 相同！"惰性"方面
// 已经内置，因为 ::type 只在后面才被访问。

// --- 3b. lazy_conditional：延迟整个条件 ---

template <bool B, typename T, typename F>
struct lazy_conditional {
    using type = T;  // 如果为 true
};

template <typename T, typename F>
struct lazy_conditional<false, T, F> {
    using type = F;  // 如果为 false
};

template <bool B, typename T, typename F>
using lazy_conditional_t = typename lazy_conditional<B, T, F>::type;

// --- 3c. 为什么"惰性"：防止错误分支的实例化 ---
//
// 关键洞见：在 lazy_conditional<false, T, F> 中，
// T 永远不会被实例化。只访问 F 的定义。
// 这在 T 对 false 情况包含无效类型时至关重要。

struct ValidType {
    using value_type = int;
};

// 如果这被急切求值，会失败：
using SafeResult = lazy_conditional_t<
    true,
    typename ValidType::value_type,   // 仅在 true 时求值
    void                              // 仅在 false 时求值
>;

static_assert(std::is_same_v<SafeResult, int>);

// ============================================================================
// 第 4 部分：带延迟求值的 void_t
// ============================================================================
//
// std::void_t<Exprs...> 将任何有效表达式映射为 void。
// 其真正威力在于参数求值被延迟到使用 void_t 的模板替换时。
//
// 这使得"检测惯用法"成为可能（C++ Library Fundamentals TS v2）。

// --- 4a. 检测惯用法 ---

// nonesuch：表示"未检测到"的类型
struct nonesuch {
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

// 主模板：detector<Default, void_t<Op<Args...>>, Op, Args...>
template <typename Default,
          typename AlwaysVoid,
          template <typename...> class Op,
          typename... Args>
struct detector {
    using value_t = std::false_type;
    using type    = Default;
};

// 特化：当 Op<Args...> 有效时
template <typename Default,
          template <typename...> class Op,
          typename... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type    = Op<Args...>;
};

// --- 4b. 便捷别名 ---

template <template <typename...> class Op, typename... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template <template <typename...> class Op, typename... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

template <typename Expected,
          template <typename...> class Op,
          typename... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

// --- 4c. 使用检测惯用法 ---

// 操作：提取 T::value_type
template <typename T>
using value_type_op = typename T::value_type;

// 操作：提取 T::iterator
template <typename T>
using iterator_op = typename T::iterator;

// 操作：T::foo(int)
template <typename T>
using has_foo_with_int = decltype(std::declval<T>().foo(0));

struct WithValueType { using value_type = double; };
struct WithIterator  { using iterator = int*; };
struct WithNothing   {};

static_assert(is_detected<value_type_op, WithValueType>::value);
static_assert(!is_detected<value_type_op, WithNothing>::value);
static_assert(is_detected<iterator_op, std::vector<int>>::value);
static_assert(is_detected_exact<double, value_type_op, WithValueType>::value);

// ============================================================================
// 第 5 部分：高级 —— 延迟函数返回类型检测
// ============================================================================

// 有时我们想知道一个方法是否存在并返回特定类型。
// 这需要嵌套的延迟求值。

// 检查：T 是否有返回 int 的 .compute() 方法？
template <typename T>
using has_compute_returning_int_op = std::enable_if_t<
    std::is_same_v<int, decltype(std::declval<T>().compute())>>;

// enable_if_t 将 is_same 的求值延迟到 declval 解析之后

struct ComputesInt    { int    compute() const { return 42; } };
struct ComputesDouble { double compute() const { return 3.14; } };

static_assert(is_detected<has_compute_returning_int_op, ComputesInt>::value);
static_assert(!is_detected<has_compute_returning_int_op, ComputesDouble>::value);

// ============================================================================
// 第 6 部分：compile_time_branch —— 在类型级别模拟 if constexpr
// ============================================================================
//
// 对于类型级别的决策（而非值级别），我们使用 lazy_conditional
// 或特化来避免实例化无效类型。

template <bool UseFastPath, typename T>
struct algorithm_selector {
    // 根据编译期标志使用不同的内部表示
    using storage_type = lazy_conditional_t<
        UseFastPath,
        std::unique_ptr<T[]>,        // 快速路径：原始数组
        std::vector<T>               // 慢速路径：vector
    >;

    // 关键：当 UseFastPath=false 时，unique_ptr<T[]> 永远不会被实例化；
    // 当 UseFastPath=true 时，vector<T> 永远不会被实例化。
};

static_assert(
    std::is_same_v<
        algorithm_selector<true, int>::storage_type,
        std::unique_ptr<int[]>
    >
);

static_assert(
    std::is_same_v<
        algorithm_selector<false, int>::storage_type,
        std::vector<int>
    >
);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 延迟求值模式 ===\n\n";

    // 第 1 部分：惰性 vs 急切
    std::cout << std::boolalpha;
    std::cout << "lazy_value_type<WithValueType> 存在："
              << is_detected<value_type_op, WithValueType>::value << "\n";
    std::cout << "lazy_value_type<int> 存在："
              << is_detected<value_type_op, int>::value << "\n";

    // 第 2 部分：identity
    std::cout << "identity_t<int>："
              << typeid(identity_t<int>).name() << "\n";

    // 第 4 部分：检测惯用法
    std::cout << "is_detected<value_type_op, std::vector<int>>："
              << is_detected<value_type_op, std::vector<int>>::value << "\n";

    // 第 6 部分：实践中的 lazy_conditional
    std::cout << "algorithm_selector<true,int> 使用 unique_ptr："
              << std::is_same_v<
                  algorithm_selector<true, int>::storage_type,
                  std::unique_ptr<int[]>
              > << "\n";

    std::cout << "\n延迟求值模式已演示。\n";
    return 0;
}
