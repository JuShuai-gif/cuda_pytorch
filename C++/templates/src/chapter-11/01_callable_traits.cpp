// ============================================================================
// 01_callable_traits.cpp - 可调用对象检测与类型 traits
// ============================================================================
//
// 目的：
//   深入理解 std::declval、std::invoke_result、std::result_of
//   （已弃用）以及自定义可调用检测 traits。这些是 SFINAE、
//   concepts 和模板元编程的基础构建块。
//
// 关键工具：
//   std::declval<T>()        - 在不求值上下文中创建 T 的"虚拟"引用
//   std::invoke_result<F, Args...> - INVOKE 的结果类型（C++17）
//   std::result_of<F(Args...)> - C++17 之前（C++17 弃用，C++20 移除）
//   std::is_invocable<F, Args...> - F 能否用 Args 调用？
//   std::is_invocable_r<R, F, Args...> - F 能否用 Args 调用并返回 R？
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <functional>
#include <utility>
#include <string>
#include <memory>
#include <optional>

// ============================================================================
// 第 1 部分：std::declval —— 基础
// ============================================================================
//
// std::declval<T>() 将任意类型 T 转换为引用类型，
// 允许我们在不求值上下文（decltype、sizeof、noexcept）中
// 使用成员函数，而无需构造真实对象。
//
// 这在以下场景中至关重要：
//   - T 的默认构造函数被删除或为 private
//   - T 是抽象类
//   - 我们只需要表达式的类型，不需要其值

// --- 1a. 为什么需要 declval ---

struct NonDefaultConstructible {
    NonDefaultConstructible() = delete;
    explicit NonDefaultConstructible(int v) : value(v) {}
    int get_value() const { return value; }
    void set_value(int v) { value = v; }
private:
    int value;
};

struct AbstractBase {
    virtual ~AbstractBase() = default;
    virtual int compute() const = 0;
};

// 没有 declval，我们无法询问"get_value() 返回什么？"
// 因为我们不能创建 NonDefaultConstructible。
//
// 使用 declval：decltype(std::declval<NonDefaultConstructible>().get_value()) → int

static_assert(std::is_same_v<
    decltype(std::declval<NonDefaultConstructible>().get_value()),
    int>);

// declval 也可用于抽象类型
static_assert(std::is_same_v<
    decltype(std::declval<AbstractBase>().compute()),
    int>);

// --- 1b. 用于引用类型的 declval ---
// std::declval<T>() 返回 T&&（右值引用）
// std::declval<T&>() 返回 T&（左值引用）

using LRef = decltype(std::declval<int&>());    // int&
using RRef = decltype(std::declval<int&&>());   // int&&
using Plain = decltype(std::declval<int>());    // int&&

static_assert(std::is_lvalue_reference_v<LRef>);
static_assert(std::is_rvalue_reference_v<RRef>);
static_assert(std::is_rvalue_reference_v<Plain>);

// ============================================================================
// 第 2 部分：invoke_result —— 现代可调用结果检测
// ============================================================================
//
// std::invoke_result<F, Args...>::type 告诉你用 Args... 调用 F 时的
// 返回类型，使用 INVOKE 协议（统一处理成员函数指针、成员数据指针、
// 自由函数和函数对象）。

// --- 2a. 自由函数 ---
int free_func(double x, int y) { return static_cast<int>(x) + y; }

static_assert(std::is_same_v<
    std::invoke_result_t<decltype(&free_func), double, int>,
    int>);

// --- 2b. 成员函数 ---
struct Widget {
    std::string name() const { return "Widget"; }
    int         value = 42;
};

// INVOKE 协议：invoke_result<decltype(&Widget::name), Widget> →
// 调用 w.name()，其中 w 是 Widget
static_assert(std::is_same_v<
    std::invoke_result_t<decltype(&Widget::name), Widget>,
    std::string>);

// --- 2c. 成员数据指针 ---
// 对成员数据指针的 INVOKE 返回对成员的引用。
// 对于非 const Widget，在 Widget 上调用的 &Widget::value 返回 int&。
// （跳过 static_assert 以避免 decltype 的编译器特定行为）

// --- 2d. Lambda / 函数对象 ---
auto lambda = [](int a, int b) -> double { return a * b * 0.5; };

static_assert(std::is_same_v<
    std::invoke_result_t<decltype(lambda), int, int>,
    double>);

// --- 2e. std::function ---
static_assert(std::is_same_v<
    std::invoke_result_t<std::function<int(double)>, double>,
    int>);

// ============================================================================
// 第 3 部分：is_invocable —— 可调用检测
// ============================================================================
//
// std::is_invocable<F, Args...> → bool：F 能否用 Args 调用？
// std::is_invocable_r<R, F, Args...> → bool：F 能否用 Args 调用，
//                                        并返回可转换为 R 的值？

// --- 检查可调用性 ---
static_assert(std::is_invocable_v<decltype(&free_func), double, int>);
static_assert(!std::is_invocable_v<decltype(&free_func), std::string>);
//                                                         ^^^^^^^^^^
//                                               错误的参数类型

// --- 检查返回类型兼容性 ---
static_assert(std::is_invocable_r_v<int, decltype(&free_func), double, int>);
static_assert(std::is_invocable_r_v<double, decltype(&free_func), double, int>);
// free_func 能否被视为返回 long？（int → long 可转换）
static_assert(std::is_invocable_r_v<long, decltype(&free_func), double, int>);

// ============================================================================
// 第 4 部分：构建自定义可调用 traits
// ============================================================================
//
// 这些是像 CUTLASS 这样的库用来约束模板参数
// 和选择实现的构建块。

// --- 4a. has_call_operator：检测 T 是否有 operator() ---
template <typename T, typename = void>
struct has_call_operator : std::false_type {};

template <typename T>
struct has_call_operator<T, std::void_t<decltype(&T::operator())>>
    : std::true_type {};

struct Functor {
    void operator()(int) const {}
};

struct NoFunctor {
    void some_method() {}
};

static_assert(has_call_operator<Functor>::value);
static_assert(!has_call_operator<NoFunctor>::value);
static_assert(has_call_operator<decltype(lambda)>::value);

// --- 4b. return_type：获取可调用对象的返回类型 ---
template <typename F, typename... Args>
using return_type = std::invoke_result_t<F, Args...>;

template <typename F, typename... Args>
using return_type_t = typename return_type<F, Args...>::type;
// 实际上 invoke_result_t 已存在；这里只是为了说明模式。

// --- 4c. is_unary_predicate：F(T) → bool ---
template <typename F, typename T>
concept UnaryPredicate = std::is_invocable_r_v<bool, F, T>;
// C++20 concept —— 详情见第 17 章
// 使用 SFINAE 的 C++20 之前等价写法：
template <typename F, typename T, typename = void>
struct is_unary_predicate : std::false_type {};

template <typename F, typename T>
struct is_unary_predicate<F, T,
    std::enable_if_t<std::is_invocable_r_v<bool, F, T>>>
    : std::true_type {};

bool check_positive(int x) { return x > 0; }

static_assert(is_unary_predicate<decltype(&check_positive), int>::value);
static_assert(!is_unary_predicate<decltype(&free_func), double>::value);
//                                free_func 返回 int，不是 bool

// --- 4d. 参数数量检测：一个可调用对象接受多少个参数？ ---
// 仅对函数类型和函数指针有效。
template <typename F>
struct function_arity;

template <typename R, typename... Args>
struct function_arity<R(Args...)> {
    static constexpr std::size_t value = sizeof...(Args);
};

template <typename R, typename... Args>
struct function_arity<R(*)(Args...)> : function_arity<R(Args...)> {};

static_assert(function_arity<decltype(&free_func)>::value == 2);
static_assert(function_arity<int(double)>::value == 1);

// ============================================================================
// 第 5 部分：实际应用 —— 通用调用器
// ============================================================================
//
// 使用 invoke_result 构建一个通用包装器，在失败时返回
// optional<T>（当可调用对象不能用给定参数调用时）。

template <typename F, typename... Args>
auto safe_invoke(F&& f, Args&&... args)
    -> std::optional<std::invoke_result_t<F, Args...>>
{
    if constexpr (std::is_invocable_v<F, Args...>) {
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    } else {
        return std::nullopt;
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 可调用对象 traits 与检测 ===\n\n";

    // 第 1 部分：declval
    std::cout << "declval 与 NonDefaultConstructible：类型在编译期存在\n";

    // 第 2 部分：invoke_result
    Widget w;
    std::cout << "invoke_result(&Widget::name, Widget)："
              << std::invoke(std::mem_fn(&Widget::name), w) << "\n";
    std::cout << "invoke_result(lambda, int, int)："
              << lambda(3, 4) << "\n";

    // 第 3 部分：is_invocable
    std::cout << std::boolalpha;
    std::cout << "is_invocable(free_func, double, int)："
              << std::is_invocable_v<decltype(&free_func), double, int> << "\n";
    std::cout << "is_invocable(free_func, string)："
              << std::is_invocable_v<decltype(&free_func), std::string> << "\n";

    // 第 4 部分：自定义 traits
    std::cout << "has_call_operator<Functor>："
              << has_call_operator<Functor>::value << "\n";
    std::cout << "is_unary_predicate(check_positive, int)："
              << is_unary_predicate<decltype(&check_positive), int>::value << "\n";

    // 第 5 部分：通用调用器
    auto result = safe_invoke(lambda, 10, 20);
    std::cout << "safe_invoke(lambda, 10, 20) = " << *result << "\n";

    std::cout << "\n可调用对象 traits 演示完成。\n";
    return 0;
}
