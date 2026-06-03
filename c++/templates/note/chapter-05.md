# 第5章 模板基础技巧 —— 你需要知道的编译器约定

## 1. 核心问题

很多人写模板代码时频繁遇到这几类问题：
- 为什么写了 `T::iterator` 编译器不认识？
- 为什么在模板里调用成员函数要加 `this->`？
- 为什么我的模板代码写到 `.cpp` 文件里就 undefined reference？
- 为什么 CUTLASS 全是 `.h` 文件，一个 `.cpp` 都没有？

这些不是语法错误，是**编译器对模板的"翻译协议"**——编译器在看到模板定义时，对某些部分选择"暂不检查"，等实例化时再处理。但暂不检查不代表不检查语法，这里有一个精确的规则，叫"两阶段查找"（Two-Phase Lookup）。

掌握了这些规则，你就能写出让编译器满意的模板代码，而不是和它斗智斗勇。

## 2. 通俗解释（生活类比）

把编译器想象成一位餐厅经理。他看到菜单（源码）上写了"做一道菜"：

```
做菜(原料 T) {
    切 T
    炒 T
    装盘
}
```

经理在第一遍看菜单时（阶段一），会检查"装盘"这个词有没有语法问题——因为"装盘"不依赖 T，不管 T 是什么菜，"装盘"的操作都是一样的。但如果写的是"椭切 T"，经理会说："椭切是什么东西？你是不是写错了？"——这是阶段一的报错。

然后，当有顾客点"做菜(土豆)"时（阶段二），经理才去检查"切土豆"和"炒土豆"在不在菜谱上。如果菜谱里有"切土豆"但没"炒土豆"，这时候才报错。

这就是 C++ 模板的两阶段查找：
- **阶段一（定义时检查）：** 检查所有不依赖模板参数 `T` 的代码
- **阶段二（实例化时检查）：** 检查依赖模板参数 `T` 的代码

## 3. `typename` 与 `class` 的区别

```cpp
template<typename T>    // 这里用 typename
void foo() {}

template<class T>       // 这里用 class
void bar() {}
```

在声明模板参数时，`typename` 和 `class` **完全等价**。那为什么有俩？

历史原因：`template<class T>` 先被引入，Bjarne Stroustrup 当时觉得用 `class` 强调 T 是一个类型名很自然。后来为了避免新手产生"模板参数必须是 class"的误解，引入了 `typename`。现在 C++ 标准推荐使用 `typename`，但 CUTLASS 代码中两者混合使用，通常代码风格指南会建议统一用 `typename`。

真正不同的是 `typename` 的另一个用途——**依赖类型声明**：

```cpp
template<typename T>
void foo() {
    typename T::iterator it;   // 必须加 typename！
}

// 为什么？
// 编译器看到 T::iterator 时，不知道它是一个类型还是一个静态成员变量
// 因为 T 还没确定，所以 T::iterator 可能是一个 static int iterator = 0;
// 加上 typename 就是告诉编译器：这玩意儿是个类型，不是变量
```

```cpp
template<typename T>
void foo() {
    T::value_type * ptr;  
    // 歧义：这到底是 (T::value_type) * ptr（声明指针）？
    //       还是 T::value_type * ptr（乘法）？
    // 答案是声明指针，传指针
}

template<typename T>
void foo() {
    typename T::value_type * ptr;  // 明确：声明指针
}
```

在 CUTLASS 中你会看到铺天盖地的 `typename`，原因就在这里——CUTLASS 大量使用类型萃取（Type Traits），几乎每一行都在取模板类型内部的类型别名(`::type`, `::shape`, `::iterator` 等）。

## 4. `template` 关键字

和 `typename` 类似，当你在模板内部调用一个依赖模板的成员函数模板时，需要加 `template`：

```cpp
template<typename T>
void foo(T obj) {
    obj.template bar<int>();  // 必须加 template
}
```

因为编译器不知道 `obj.bar` 是一个成员函数模板还是成员变量。`obj.bar < 3` 可能被解析为 `(obj.bar) < 3`（比较大小），而不是 `obj.bar<3>()`（函数调用）。加上 `template` 就消除了歧义。

CUTLASS 中的典型场景：

```cpp
// cutlass/gemm/device/gemm.h 中的 dispatch 代码
using EpilogueOutputOp = typename EpilogueOutputOp_::template Params<ElementOutput_>;
```

这里 `template Params` 就是因为 `EpilogueOutputOp_` 是一个依赖类型，它的内部模板 `Params<...>` 需要在前面加 `template` 关键字才能被正确解析。

## 5. `this->` 的作用

```cpp
template<typename T>
class Base {
protected:
    void bar() {}
};

template<typename T>
class Derived : public Base<T> {
public:
    void foo() {
        // bar();        // ❌ 编译错误：找不到 bar
        this->bar();     // ✅ 正确
        // Base<T>::bar();  // ✅ 也行
    }
};
```

为什么 `bar()` 找不到？因为在阶段一（模板定义时），编译器不会检查依赖基类 `Base<T>`。`T` 还不知道是什么，万一存在一个 `Base<int>` 的偏特化里根本没有 `bar()` 呢？所以编译器选择在阶段一"假装没看到"。

加了 `this->` 后，`bar()` 变成了依赖表达式的一部分，编译器会推迟到阶段二（实例化时）再检查。这是两阶段查找的直接后果。

CUTLASS 中几乎每个类模板的成员函数内部都在用 `this->`：

```cpp
// 来自 cutlass 的线程块代码模式
template <typename ThreadblockShape_>
class Mma {
public:
    void operator()(...) {
        this->load_a();  // load_a 来自依赖基类
        this->load_b();
    }
};
```

## 6. 模板代码不能写到 `.cpp` 文件

这是 C++ 新手最常见的问题：

```cpp
// my_template.h
template<typename T>
T max(T a, T b);

// my_template.cpp
template<typename T>
T max(T a, T b) { return a > b ? a : b; }  // 实现写在 .cpp 里

// main.cpp
#include "my_template.h"
int result = max(1, 2);  // ❌ 链接错误: undefined reference to max<int>(int, int)
```

**原因：** 模板不是代码，是"生成代码的规则"。编译器在编译 `my_template.cpp` 时，没有看到任何人需要 `max<int>`，所以根本不会实例化它。到链接阶段，连接器找不到 `max<int>` 的符号，就报错了。

**解法有三：**

1. **单头文件模式（CUTLASS 的选择）：** 所有模板实现写在 `.h` 文件里
2. **显式实例化：** 在 `.cpp` 里写 `template int max<int>(int, int);`
3. **`.tcc` / `.impl.h` 模式：** 实现写在 `.tcc` 文件、`.h` 末尾 `#include "xxx.tcc"`

CUTLASS 选择了第一和第三种组合——绝大多数代码在 `.h` 文件中（inline），少量重型代码通过 `.inl` 文件包含的方式组织。

## 7. 零初始化与继承

```cpp
template<typename T>
class Holder {
    T value;      // 未初始化，值是垃圾
    T value_{};   // 零初始化（对内置类型也生效）
    T value_ = T{};  // 显式零初始化
    T value_ = T();  // 等价，但这有"most vexing parse"的风险
};
```

对于模板类型 `T`，你不知道它是内置类型（如 `int`）还是类类型。`int x;` 是未初始化的，`std::string s;` 是默认构造的（空字符串）。所以用 `T value_{}` 或 `T value_ = {}` 来确保值初始化——对 `int` 就是 0，对类类型就是默认构造。

## 8. 工业界真实用途

### 8.1 为什么 CUTLASS 全是单头文件

打开 `cutlass/include/cutlass/` 目录，你会看到一个 `.cpp` 文件都没有。原因不仅限于模板代码必须放在头文件——更深层的原因是 CUTLASS 的设计哲学：

- **所有决策都在编译期完成。** 运行时只执行已经彻底定型的代码。
- **每行代码都可能被内联。** 头文件实现让所有函数都是 inline candidate，编译器可以跨函数做优化。
- **无符号导出问题。** 没有 `.cpp` 就没有 `.o`，没有链接问题，不用处理跨编译单元的符号可见性。
- **HIP/CUDA 双后端。** CUTLASS 同时支持 CUDA 和 HIP（ROCm）。单头文件模式避免了对 `nvcc` 或 `hipcc` 的特殊链接逻辑。

### 8.2 为什么 `typename` 满天飞

在 CUTLASS 的一个函数模板里，你会看到类似这样的代码：

```cpp
template <typename GemmKernel_>
class Gemm {
public:
    using ElementA = typename GemmKernel_::ElementA;
    using ElementB = typename GemmKernel_::ElementB;
    using ElementC = typename GemmKernel_::ElementC;

    using UnderlyingKernel = typename GemmKernel_::Kernel;
    //                          ^^^^^^^^^
    //                         每个 :: 后面如果取类型，必须加 typename
    //                         因为编译器不知道 :: 后面是什么
};
```

这不是语法糖——这是编译期的强制要求。CUTLASS 在定义了足够多的"类型萃取"（Type Traits）后，typename 的使用是必然会爆炸的。

### 8.3 TensorRT 的单头文件模式

TensorRT 的开源 plugin 代码也是几乎全头文件模式。`plugin/common/kernel.h` 里把 GPU kernel 的实现直接写在头文件中，以便 `nvcc` 能从调用点出发做内联优化。

## 9. 与 CUTLASS 的联系

### 9.1 cutlass/include/cutlass/ 目录结构没有 .cpp 文件

你可以直接验证：在这个目录及其子目录中，几乎找不到 `.cpp` 文件。所有代码都在 `.h` 和 `.inl` 中。

这是因为 CUTLASS 的代码本质上是一个**编译期代码生成器**。它不需要运行时库——用户 `#include` 了 CUTLASS 头文件后，编译器在编译用户代码时直接生成 CUDA kernel 机器码。

### 9.2 模板依赖名的实际用例

打开 `cutlass/include/cutlass/gemm/kernel/default_gemm.h`：

```cpp
template <
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_
>
struct DefaultGemm {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using Swizzle = ThreadblockSwizzle_;

    // 从 Mma_ 中取类型——都是依赖名
    using IteratorA = typename Mma::IteratorA;     // typename 必须
    using IteratorB = typename Mma::IteratorB;     // typename 必须
    using WarpMma = typename Mma::WarpMma;         // typename 必须
    using SharedStorage = typename Mma::SharedStorage; // typename 必须
};
```

这就是 **Policy-based Design** 的一个典型实例：`DefaultGemm` 不知道 `Mma_` 具体是什么，它只需要从 `Mma_` 中提取一些类型别名。`typename` 是链接这个"不知道"到"等实例化时再说"的桥梁。

## 10. 常见坑点汇总

| 坑 | 现象 | 根因 | 解法 |
|---|---|---|---|
| `.cpp` 分离 | linker 报错 | 模板不是代码，是规则 | 全放头文件，或用显式实例化 |
| `typename` 缺失 | `T::value_type` 被当成静态变量 | 两阶段查找中的歧义 | 加 `typename` |
| `template` 缺失 | `<` 被当成小于号 | 编译器无法区分 `<` 的含义 | 加 `template` 关键字 |
| `this->` 缺失 | 依赖基类的方法找不到 | 第一阶段不检查依赖基类 | 加 `this->` |
| 零初始化缺失 | 内置类型随机值 | `int x;` 未初始化 | 用 `T x{}` 或 `T x = {}` |

## 11. 本章总结

C++ 模板的编译约定看起来像是"语言设计的瑕疵"，但实际上它们是**编译期代码生成器的精确规约**。

- `typename` 告诉你"这个依赖名是类型，不是变量"
- `this->` 告诉你"这个方法来自依赖基类，等实例化时再查"
- `.template` 告诉你"这是一个模板调用，不是小于号比较"
- 单头文件模式告诉你"模板不是实现，是生成规则，必须在每个编译单元里可见"

CUTLASS 把这些约定用到了极致——它的代码里 `typename` 无处不在，`this->` 行行皆有，头文件千行起步。这不是代码风格问题，而是模板元编程的**必然**。

> 关键认知：**C++ 的模板编译约定不是随便拍脑袋定的，而是编译器在"既要提前检查不必依赖类型的代码"和"又要推迟检查依赖类型的代码"之间做的精确妥协。** 理解了这一点，你就理解了为什么那些看似啰嗦的关键字（`typename`、`template`、`this->`）一个都不能省。
