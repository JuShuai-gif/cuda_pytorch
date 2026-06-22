// ============================================================================
// 01_name_lookup.cpp - ADL、两阶段名字查找 与 依赖名字
// ============================================================================
//
// 目的:
//   深入探索 C++ 模板的名字查找规则：
//   实参依赖查找 (ADL)、两阶段名字查找、
//   以及依赖名字与非依赖名字的关键区别。
//
// 为什么重要:
//   模板编译错误通常源于名字查找问题。
//   理解名字何时被查找（阶段 1 vs 阶段 2）对于
//   调试和设计模板库至关重要。
//
// 两阶段查找:
//   阶段 1（模板定义时）:
//     - 非依赖名字被查找并绑定
//     - 非依赖表达式中的语法错误被检测
//     - 非依赖基类被检查
//
//   阶段 2（模板实例化时）:
//     - 依赖名字被查找（同时使用 ADL 和普通查找）
//     - 依赖基类被检查
//     - POI（实例化点）决定了哪些名字可见
//
// 关键规则:
//   - typename: 消歧依赖类型名
//   - template: 消歧依赖模板名
//   - this->:    强制对成员名进行依赖查找
//
// ============================================================================

#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// ============================================================================
// 第 1 节: 依赖名 vs 非依赖名
// ============================================================================

// --- 1a. 非依赖名（阶段 1 查找） ---

int global_value = 100;

template <typename T>
struct NonDepDemo {
    void print() {
        // "global_value" 是非依赖名 — 它不依赖于
        // 模板参数 T。它在模板定义点（阶段 1）被查找并绑定。
        std::cout << "非依赖 global_value = " << global_value << "\n";
    }
};

// --- 1b. 依赖名（阶段 2 查找） ---

template <typename T>
struct DepDemo {
    void print(T const& val) {
        // "val.size()" 是依赖表达式，因为 val 的类型
        // 依赖于 T。它在实例化点（阶段 2）
        // 通过普通查找和 ADL 一起被查找。
        std::cout << "依赖 val.size() = " << val.size() << "\n";

        // 没有 ADL 的话，编译器在实例化时之前不会知道
        // 用户自定义类型的成员函数。
    }
};

// ============================================================================
// 第 2 节: typename 关键字
// ============================================================================
//
// 在模板内部，当你写 T::something 时，编译器
// 默认假设 "something" 是一个值（静态成员、枚举值）。
// 使用 "typename" 来明确告诉编译器它是类型。

template <typename T>
struct TypenameDemo {
    // 没有 typename：编译器假设 T::value_type 是静态成员
    // T::value_type val;  // ❌ 错误: 依赖名被假定为非类型

    // 有了 typename：编译器知道 T::value_type 是类型
    typename T::value_type val = {};

    // 模板别名也需要
    using MyType = typename T::value_type;

    // 返回类型中也需要
    typename T::value_type get() const { return val; }

    // C++20: typename 在某些上下文中可以省略（如泛型 lambda）
    // 但在模板定义中对依赖类型仍然是必需的。
};

// ============================================================================
// 第 3 节: template 关键字
// ============================================================================
//
// 在模板内部，当你调用 T::foo<...>() 时，编译器
// 会把 < 解释为小于号。使用 "template" 来消歧。

template <typename T>
struct TemplateKeywordDemo {
    void call_template(T& obj) {
        // 错误写法: 编译器把 obj.foo< 看作比较运算
        // obj.foo<int>();  // ❌ 错误: < 被当作小于号

        // 正确写法: template 关键字告诉编译器 foo 是模板
        obj.template foo<int>();

        // 同样适用于 ::template 和 ->template
    }
};

// 带有成员模板的示例类型
struct HasMemberTemplate {
    template <typename U>
    void foo() {
        std::cout << "HasMemberTemplate::foo<" << typeid(U).name() << ">()\n";
    }
};

// ============================================================================
// 第 4 节: this-> 用于依赖基类成员
// ============================================================================
//
// 非依赖基类在阶段 1 被检查。
// 依赖基类（依赖于 T 的）在阶段 2 被检查。
// 要强制对基类成员进行阶段 2 查找，使用 this->。

template <typename T>
struct BaseTemplate {
    int base_value = 42;
    void base_method() { std::cout << "Base::base_method()\n"; }
};

template <typename T>
struct DerivedTemplate : BaseTemplate<T> {
    void demo() {
        // 错误写法: base_value 是非依赖名（不包含 T）
        //      但它来自依赖基类 → 编译器找不到它
        // std::cout << base_value;  // ❌ 错误: 找不到

        // 解决方案 1: this-> 强制依赖查找
        std::cout << "base_value = " << this->base_value << "\n";

        // 解决方案 2: 显式限定
        std::cout << "base_value = " << DerivedTemplate::base_value << "\n";

        // 解决方案 3: using 声明
        // using BaseTemplate<T>::base_value;

        this->base_method();
    }
};

// ============================================================================
// 第 5 节: 实参依赖查找 (ADL / Koenig 查找)
// ============================================================================
//
// ADL 将名字查找扩展到包含函数实参的命名空间。
// 这就是为什么你可以写 `std::cout << "hello"` 而不需要 `std::operator<<`
// 以及为什么 `swap(a, b)` 能找到用户定义的 swap 重载。

namespace MyLib {

struct Vector3 {
    float x, y, z;
};

// 与 Vector3 在同一命名空间的自由函数
std::ostream& operator<<(std::ostream& os, Vector3 const& v) {
    return os << "Vector3(" << v.x << ", " << v.y << ", " << v.z << ")";
}

// 另一个自由函数
float dot(Vector3 const& a, Vector3 const& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

} // namespace MyLib

// --- 5a. ADL 实战 ---

void adl_demo() {
    MyLib::Vector3 v1{1, 2, 3};
    MyLib::Vector3 v2{4, 5, 6};

    // ADL 找到了 MyLib 中的 operator<<，因为 v1 的类型在 MyLib 中
    std::cout << v1 << "\n";

    // ADL 找到了 MyLib 中的 dot() — 不需要 MyLib:: 前缀
    float d = dot(v1, v2);  // ADL: 搜索 MyLib 命名空间
    std::cout << "dot = " << d << "\n";
}

// --- 5b. 带模板的 ADL（阶段 2） ---

namespace LibA {
    struct TagA {};
    void process(TagA) { std::cout << "LibA::process(TagA)\n"; }
}

namespace LibB {
    struct TagB {};
    void process(TagB) { std::cout << "LibB::process(TagB)\n"; }
}

template <typename T>
void call_process(T const& obj) {
    // "process" 是依赖名（通过 ADL 依赖于 T）。
    // 阶段 2 查找包含基于 T 的命名空间的 ADL。
    process(obj);  // ADL 找到 LibA::process 或 LibB::process
}

// ============================================================================
// 第 6 节: 两阶段查找陷阱 — "解析" 问题
// ============================================================================
//
// 阶段 1 解析并检查非依赖代码。如果你写的代码
// 只对某些实例化有效，你需要让它变成依赖的，
// 这样解析就会推迟到阶段 2。

template <typename T>
struct TwoPhaseGotcha {
    void bad_approach() {
        // 这是非依赖表达式。编译器尝试在阶段 1
        // 解析 `size` 但失败了（作用域中没有 `size`）。
        // std::cout << size;  // ❌ 阶段 1 错误
    }

    void good_approach(T const& obj) {
        // 这依赖于 T → 推迟到阶段 2
        std::cout << obj.size();  // ✅ 可以，即使 size() 可能还不存在
    }
};

// ============================================================================
// 第 7 节: POI（实例化点）与名字可见性
// ============================================================================
//
// POI 决定了阶段 2 查找时哪些名字可见。
// 对于函数模板，POI 紧跟在包含它的命名空间作用域定义之后。
// 对于类模板，POI 在包含它的命名空间作用域之前。

int value_visible = 1;

template <typename T>
void poi_demo_1() {
    // 在 POI 处，value_visible 可见
    std::cout << "value_visible = " << value_visible << "\n";
}

// 在此处，poi_demo_1<int> 会在这里被实例化（POI）
// 而 value_visible (1) 是可见的。

// ============================================================================
// 第 8 节: 编译期验证
// ============================================================================

// 验证 typename 用法能编译
struct HasValueType {
    using value_type = int;
};

TypenameDemo<HasValueType> tdemo;
static_assert(std::is_same_v<decltype(tdemo.get()), int>);

// 验证 ADL 能找到正确的重载
// （在 main 中进行运行时测试）

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== 名字查找: ADL、两阶段查找、依赖名 ===\n\n";

    // 第 1 节: 依赖 vs 非依赖
    NonDepDemo<int> nd;
    nd.print();

    std::vector<int> v = {1, 2, 3};
    DepDemo<std::vector<int>> dd;
    dd.print(v);

    // 第 4 节: this-> 与依赖基类
    DerivedTemplate<int> dt;
    dt.demo();

    // 第 5 节: ADL
    std::cout << "\n--- ADL 演示 ---\n";
    adl_demo();

    std::cout << "\n--- 带模板的 ADL ---\n";
    call_process(LibA::TagA{});
    call_process(LibB::TagB{});

    // 第 3 节: template 关键字
    std::cout << "\n--- template 关键字 ---\n";
    HasMemberTemplate hmt;
    TemplateKeywordDemo<HasMemberTemplate> tkd;
    tkd.call_template(hmt);

    // 第 7 节: POI 演示
    std::cout << "\n--- POI 演示 ---\n";
    poi_demo_1<int>();

    std::cout << "\n名字查找演示完成。\n";
    return 0;
}
