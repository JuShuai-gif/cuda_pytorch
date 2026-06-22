// ============================================================================
// 01_template_template_params.cpp - 模板模板参数
// ============================================================================
//
// 目的：
//   全面探索模板模板参数（TTPs），这是一项强大但经常被
//   低估的 C++ 特性。TTPs 允许一个模板接受另一个模板作为参数，
//   从而实现高阶模板元编程。
//
// 关键概念：
//   1. 基本语法：template <template <typename> class Container>
//   2. TTPs 中的默认参数
//   3. 可变参数 TTPs（C++17）
//   4. 匹配规则（C++17 放宽匹配）
//   5. 限制与解决方案
//
// CUTLASS 中的实际使用：
//   Epilogue 操作符使用 TTPs 来接受不同的操作树。
//   内存布局和分块迭代器也利用了 TTPs。
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <vector>
#include <list>
#include <deque>
#include <memory>

// ============================================================================
// 第 1 部分：基本模板模板参数语法
// ============================================================================

/// \brief 一个简单的容器包装器，以容器模板而非特定容器实例化
/// 作为参数。
///
/// 旧方式（TTP 之前）：直接传递容器类型
///   ContainerWrapper<std::vector<int>> → 锁定为 int
///
/// TTP 方式：传递容器模板，在内部用 T 实例化
///   ContainerWrapper<int, std::vector> → 灵活的元素类型
template <typename T,
          template <typename, typename> class Container = std::vector>
class ContainerWrapper {
public:
    using container_type = Container<T, std::allocator<T>>;
    using value_type = T;

    void push(T const& val) { data_.push_back(val); }
    T    back() const { return data_.back(); }
    std::size_t size() const { return data_.size(); }

private:
    container_type data_;
};

// ============================================================================
// 第 2 部分：C++17 放宽匹配
// ============================================================================
//
// 在 C++17 之前，模板模板参数要求参数的数量和种类完全匹配。
// std::vector 有两个模板参数（T, Allocator），所以
// template <typename> class Container 不能匹配 std::vector。
//
// C++17 引入了 "typename..." 作为 TTP 参数，允许匹配
// 任意数量模板参数的模板。

/// \brief C++17 之前：必须匹配确切的参数数量
/// std::vector 不会匹配，因为它有 2 个参数（T, Alloc）
template <typename T,
          template <typename> class Container>   // ❌ 不能匹配 std::vector
class OldStyleWrapper {
    using type = Container<T>;  // 假设只有一个参数的模板
};

/// \brief C++17：匹配任意模板，无论参数数量
template <typename T,
          template <typename...> class Container>  // ✅ 匹配 std::vector、std::list 等
class ModernWrapper {
public:
    using type = Container<T>;
};

// 现代方法适用于所有标准容器：
using VecWrapper = ModernWrapper<int, std::vector>::type;   // std::vector<int>
using ListWrapper = ModernWrapper<int, std::list>::type;    // std::list<int>

static_assert(std::is_same_v<VecWrapper, std::vector<int>>);
static_assert(std::is_same_v<ListWrapper, std::list<int>>);

// ============================================================================
// 第 3 部分：实用的 TTP —— 可重新绑定的分配器
// ============================================================================

/// \brief 一个将数据存储在用户指定容器模板中的类型。
/// 容器模板是可重新绑定的：用户指定容器形状，
/// 库选择元素类型。
template <typename T,
          template <typename...> class ContainerTemplate = std::vector>
class DataStore {
public:
    using container_type = ContainerTemplate<T>;
    container_type data;

    void fill(T const& val, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i)
            push_back_impl(data, val);
    }

private:
    // SFINAE 友好的 push_back：适用于 vector/list/deque
    template <typename C>
    static auto push_back_impl(C& c, T const& val)
        -> decltype(c.push_back(val), void())
    {
        c.push_back(val);
    }
};

// ============================================================================
// 第 4 部分：带非类型参数的 TTP
// ============================================================================

/// \brief 接受带非类型参数的模板的模板。
/// 用于在编译期选择不同的数组大小。
template <typename T, std::size_t N>
struct FixedArray {
    T data[N];
    static constexpr std::size_t size() { return N; }
};

/// \brief 接受一个带单个类型和单个非类型参数的模板的包装器
template <typename T,
          std::size_t N,
          template <typename, std::size_t> class ArrayTemplate>
class ArrayWrapper {
public:
    using array_type = ArrayTemplate<T, N>;
    array_type arr;

    static constexpr std::size_t capacity() { return N; }
};

using WrappedFixedArray = ArrayWrapper<int, 16, FixedArray>;
static_assert(WrappedFixedArray::capacity() == 16);

// ============================================================================
// 第 5 部分：基于策略的 TTP 设计
// ============================================================================

/// \brief 带一个类型参数的策略模板。
/// 不同策略为相同接口提供不同行为。
template <typename T>
struct AddPolicy {
    static constexpr T combine(T a, T b) { return a + b; }
};

template <typename T>
struct MultiplyPolicy {
    static constexpr T combine(T a, T b) { return a * b; }
};

template <typename T>
struct MaxPolicy {
    static constexpr T combine(T a, T b) { return (a > b) ? a : b; }
};

/// \brief 接受策略模板作为模板模板参数的 Reducer。
/// 这允许在编译期选择组合策略，而不需要实例化所有策略。
template <typename T,
          template <typename> class Policy = AddPolicy>
class Reducer {
public:
    T reduce(T const* begin, T const* end) const {
        if (begin == end) return T{};
        T result = *begin++;
        while (begin != end) {
            result = Policy<T>::combine(result, *begin++);
        }
        return result;
    }
};

// ============================================================================
// 第 6 部分：真实 CUTLASS 模式 —— Epilogue 操作树
// ============================================================================
//
// CUTLASS epilogues 在主 GEMM 之后应用逐元素操作
// （例如 bias 加、激活函数）。这些操作形成一棵树，
// 通过模板模板参数来构建。

/// \brief 逐元素操作：恒等（直通）
template <typename T>
struct IdentityOp {
    static T apply(T val) { return val; }
    static constexpr const char* name() { return "Identity"; }
};

/// \brief 逐元素操作：ReLU
template <typename T>
struct ReluOp {
    static T apply(T val) { return (val > T{0}) ? val : T{0}; }
    static constexpr const char* name() { return "ReLU"; }
};

/// \brief 逐元素操作：bias 加
template <typename T>
struct BiasAddOp {
    T bias;
    explicit BiasAddOp(T b) : bias(b) {}
    T apply(T val) const { return val + bias; }
};

/// \brief Epilogue：组合两个逐元素操作。
/// 操作以模板模板参数形式传递，
/// 因此 epilogue 不绑定到特定的操作类型。
template <typename T,
          template <typename> class Op1 = IdentityOp,
          template <typename> class Op2 = IdentityOp>
class Epilogue {
public:
    T process(T val) const {
        // 应用 Op1，然后 Op2（从左到右组合）
        return Op2<T>::apply(Op1<T>::apply(val));
    }

    static void describe() {
        std::cout << "Epilogue: " << Op1<T>::name()
                  << " → " << Op2<T>::name() << "\n";
    }
};

// ============================================================================
// 第 7 部分：编译期验证
// ============================================================================

// ContainerWrapper 使用默认（std::vector）
using DefaultContainer = ContainerWrapper<int>;
static_assert(std::is_same_v<DefaultContainer::container_type,
    std::vector<int, std::allocator<int>>>);

// ContainerWrapper 使用 std::list
using ListContainer = ContainerWrapper<int, std::list>;
static_assert(std::is_same_v<ListContainer::container_type,
    std::list<int, std::allocator<int>>>);

// Reducer 使用不同策略
static_assert(AddPolicy<int>::combine(2, 3) == 5);
static_assert(MultiplyPolicy<int>::combine(2, 3) == 6);
static_assert(MaxPolicy<int>::combine(2, 3) == 3);

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 模板模板参数 ===\n\n";

    // 第 1 部分：ContainerWrapper
    ContainerWrapper<int, std::vector> cw;
    cw.push(42);
    std::cout << "ContainerWrapper back：" << cw.back() << "\n";

    // 第 3 部分：DataStore 使用用户指定的容器模板
    DataStore<int, std::vector> store_v;
    store_v.fill(7, 3);
    std::cout << "DataStore<vector> size：" << store_v.data.size() << "\n";

    DataStore<int, std::list> store_l;
    store_l.fill(7, 3);
    // std::list 没有 O(1) 的 .size()；我们使用 std::distance
    std::cout << "DataStore<list> first：" << store_l.data.front() << "\n";

    // 第 5 部分：基于策略的 reducer
    int values[] = {1, 2, 3, 4, 5};
    Reducer<int, AddPolicy> sum_reducer;
    std::cout << "Sum：" << sum_reducer.reduce(values, values + 5) << "\n";

    Reducer<int, MultiplyPolicy> prod_reducer;
    std::cout << "Product：" << prod_reducer.reduce(values, values + 5) << "\n";

    Reducer<int, MaxPolicy> max_reducer;
    std::cout << "Max：" << max_reducer.reduce(values, values + 5) << "\n";

    // 第 6 部分：Epilogue 组合
    Epilogue<float, ReluOp, IdentityOp> relu_epilogue;
    relu_epilogue.describe();
    std::cout << "ReLU(-3.5) = " << relu_epilogue.process(-3.5f) << "\n";
    std::cout << "ReLU( 3.5) = " << relu_epilogue.process(3.5f) << "\n";

    std::cout << "\n模板模板参数演示完成。\n";
    return 0;
}
