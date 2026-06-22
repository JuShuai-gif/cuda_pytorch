// ============================================================================
// 01_instantiation_model.cpp - 实例化点 (POI)、
//                               贪婪 vs 惰性实例化
// ============================================================================
//
// 目的:
//   演示 C++ 模板实例化模型：模板特化在哪里和何时
//   被实例化，贪婪和惰性实例化策略之间的区别，
//   以及对代码组织和构建性能的影响。
//
// 关键概念:
//   1. 实例化点 (POI)              - 代码生成发生的位置
//   2. 贪婪（急切）实例化          - 首次使用时即实例化
//   3. 惰性（按需）实例化          - 仅在需要时实例化
//   4. 隐式 vs 显式实例化          - 编译器驱动 vs 程序员驱动
//   5. 实例化深度                 - 递归实例化链
//   6. "实例化上下文"              - 在 POI 处可见的内容
//
// 编译器行为:
//   不同编译器（GCC、Clang、MSVC、NVCC）在精确的
//   POI 位置和贪婪程度上存在细微差异。CUTLASS 必须在所有平台上工作。
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// ============================================================================
// 第 1 节: 函数模板的实例化点
// ============================================================================
//
// 对于函数模板特化，POI 紧跟在
// 包含触发实例化的调用的命名空间作用域定义或翻译单元
// 之后。
//
// 这意味着在模板定义之后但在 POI 之前
// 声明的名字是可见的（不同于模板体中的
// 非依赖名）。

namespace poi_demo {

    // --- 1a. 在 POI 处可见的名字 ---

    // 在模板之后但在实例化调用之前声明
    void helper_v1(int x) {
        std::cout << "helper_v1(int): " << x << "\n";
    }

    template <typename T>
    void call_helper(T val) {
        // helper_v1 是依赖名（通过 val 依赖于 T）；
        // 它将在 POI 处被查找 — 因此 helper_v1(int) 是可见的
        helper_v1(val);
    }

    // 在模板之后声明，但在 POI 处仍然可见
    void helper_v2(double x) {
        std::cout << "helper_v2(double): " << x << "\n";
    }

    template <typename T>
    void call_helper_v2(T val) {
        helper_v2(val);  // 找到了上面声明的重载
    }

    // --- 1b. "POI 遮蔽" 陷阱 ---

    // print() 函数必须在模板之前声明，
    // 使用它的模板才能在 POI 处通过阶段 2 查找找到它。
    void print(int x) {
        std::cout << "print(int) 来自模板之后: " << x << "\n";
    }

    template <typename T>
    void print_value(T val) {
        print(val);  // 依赖名 — ADL + POI 查找
    }

} // namespace poi_demo

// ============================================================================
// 第 2 节: 贪婪 vs 惰性实例化
// ============================================================================
//
// 隐式实例化可以是贪婪的或惰性的：
//
//   贪婪: 当类模板被隐式实例化时，所有
//         成员函数声明都会被实例化（但不会实例化
//         它们的定义/体）。这可以尽早捕获错误。
//
//   惰性: 成员函数定义仅在
//         实际使用时（被调用、odr-used、取地址）才被实例化。
//
// 这种区别对于 SFINAE 和编写
// 处理不完整类型的模板至关重要。

// --- 2a. 成员函数的惰性实例化 ---

template <typename T>
struct LazyDemo {
    // 这个函数从未被调用，所以它的体永远不会被实例化。
    // 它可以包含对某些 T 会失败的代码而不会报错。
    void never_called() {
        T::nonexistent_method();  // 对 int 会失败 — 但从未被实例化！
    }

    // 这个函数被调用了，所以它的体会被实例化
    void called() {
        std::cout << "LazyDemo::called() sizeof(T)=" << sizeof(T) << "\n";
    }
};

// --- 2b. 成员声明的贪婪实例化 ---

template <typename T>
struct GreedyDemo {
    // 成员声明被贪婪地实例化。
    // 如果 T 是不完整的，这会立即失败。
    T member;  // 贪婪: T 必须是完整的

    // 但成员函数定义仍然是惰性的
    void method() {
        T::static_method();  // 惰性: 仅在 method() 被调用时检查
    }
};

// --- 2c. 演示区别 ---

struct Complete {
    static void static_method() {}
};

// 没问题: Complete 是完整的
static GreedyDemo<Complete> gd_complete;

// 会失败: Incomplete 是不完整的
// struct Incomplete;
// static GreedyDemo<Incomplete> gd_incomplete;  // ❌ 贪婪失败于 T member;

// ============================================================================
// 第 3 节: 实例化深度与递归模板
// ============================================================================
//
// 模板可以递归地实例化其他模板。
// 标准规定最小实例化深度为 1024
//（由实现定义）。超过这个深度是有问题的。
// CUTLASS 在复杂的 epilogue 树中
// 小心控制实例化深度。

// --- 3a. 递归模板元编程 ---

// 编译期阶乘（经典递归模板）
template <int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0> {
    static constexpr int value = 1;
};

static_assert(Factorial<5>::value == 120);
static_assert(Factorial<10>::value == 3628800);

// --- 3b. 实例计数（跟踪实例化深度） ---

template <int N>
struct InstantiationDepth {
    static constexpr int depth = 1 + InstantiationDepth<N - 1>::depth;
};

template <>
struct InstantiationDepth<0> {
    static constexpr int depth = 0;
};

static_assert(InstantiationDepth<5>::depth == 5);

// ============================================================================
// 第 4 节: 实例化上下文
// ============================================================================
//
// "实例化上下文" 决定了错误消息的样子。
// 当模板实例化失败时，编译器报告从 POI 往回
// 经过所有中间实例化的错误链。
//
// NVCC（CUDA 编译器）有不同的实例化模型：
// 模板分别为 host 和 device 代码单独实例化。

/// \brief 模拟 host/device 双重实例化。
/// 在 CUDA 中，__host__ __device__ 函数被实例化两次。
template <typename T>
struct DualInstantiation {
    // 在真实 CUDA 中: __host__ __device__
    void process(T val) {
        std::cout << "处理 " << val << "\n";
    }
};

// ============================================================================
// 第 5 节: 显式实例化与 POI
// ============================================================================
//
// 显式实例化在显式实例化声明的点
// 创建一个额外的 POI。
//
// extern template 抑制隐式实例化
//（POI 被移动到显式实例化定义处）。

template <typename T>
struct ExplicitPOIDemo {
    void method() {
        // 在显式实例化的 POI 处，在显式实例化之前
        // 声明的名字是可见的。
        std::cout << "ExplicitPOIDemo::method()\n";
    }
};

// 显式实例化: 在此处创建一个 POI
// 在此处可见的所有名字都可用于阶段 2 查找
template struct ExplicitPOIDemo<int>;

// ============================================================================
// 第 6 节: 对库设计的实际影响
// ============================================================================
//
// 1. 头文件: 在头文件中定义模板。定义必须在
//    POI 处可见（POI 在用户的 TU 中）。
//
// 2. 显式实例化: 放在专用的 .cpp 文件中，
//    以控制 POI 并减少编译时间。
//
// 3. 不完整类型: 要小心 — 惰性实例化意味着
//    错误可能远离其根源才暴露出来。
//
// 4. SFINAE: 依赖惰性实例化 — 带有替换失败的
//    成员函数定义会被静默丢弃。

// ============================================================================
// 第 7 节: 编译期验证
// ============================================================================

// 惰性: never_called 没有被实例化，所以对 int 不会出错
static_assert(sizeof(LazyDemo<int>) > 0);

// 贪婪: Complete 结构体没问题，因为在使用时它是完整的
static_assert(sizeof(GreedyDemo<Complete>) > 0);

// 阶乘深度
static_assert(Factorial<7>::value == 5040);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 实例化点与实例化模型 ===\n\n";

    // 第 1 节: POI 演示
    std::cout << "--- POI 演示 ---\n";
    poi_demo::call_helper(42);     // 找到 helper_v1(int)
    poi_demo::call_helper_v2(3.14); // 找到 helper_v2(double)
    poi_demo::print_value(100);    // 在 POI 处找到 print(int)

    // 第 2 节: 惰性 vs 贪婪
    std::cout << "\n--- 惰性 vs 贪婪 ---\n";
    LazyDemo<int> ld;
    ld.called();
    // ld.never_called();  // 对 int 能编译但体中有无效代码

    GreedyDemo<Complete> gd;
    gd.method();

    // 第 3 节: 实例化深度
    std::cout << "\n--- 实例化深度 ---\n";
    std::cout << "阶乘<10> = " << Factorial<10>::value << "\n";
    std::cout << "实例化深度<5> = " << InstantiationDepth<5>::depth << "\n";

    // 第 4 节: 双重实例化模拟
    std::cout << "\n--- 双重实例化 ---\n";
    DualInstantiation<int> di;
    di.process(42);

    // 第 5 节: 显式 POI
    std::cout << "\n--- 显式实例化 POI ---\n";
    ExplicitPOIDemo<int> epd;
    epd.method();

    std::cout << "\n实例化模型演示完成。\n";
    return 0;
}
