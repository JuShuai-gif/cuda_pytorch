# 透过 gcc ranges 源码学 TMP —— 从十大特性到自定义 view

> 本笔记合并整理自两篇知乎文章：
> - 《学好现代 c++ 是不是一定要掌握模板元编程 (TMP)？》—— 以 gcc ranges 源码为例讲解 10 个高频 TMP 特性
> - 《为 gcc10 的 ranges 写自定义 view》—— 用 cycle / zip 两个例子演示自定义 view 的完整套路
>
> 两篇文章主题高度互补：前者是"读懂 ranges 源码需要的 TMP 知识地基"，后者是"用这套地基动手写一个 view"。先看懂源码，再动手扩展，正好串成一条完整学习路径。

## 写在前面：为什么从 ranges 源码学 TMP

- gcc10 是当时对 C++20 特性支持最全的编译器，包括 ranges 在内的诸多特性在 trunk 中即可试用。
- gcc10 的 ranges 头文件只用了 3000 多行代码，就实现了 C++20 规定的全部 views / adaptors 类。
- 它是 **C++ 标准库第一次出现的：连续的、规模较大、难度适中、又有代表性的 TMP 代码**，是学习 TMP 常规套路的绝佳素材。
- ranges 被包括 C++ 之父在内的人称为"下一代 STL"：它把业务逻辑中的循环、多重循环、边界判断进行高度提炼、科学拆分和高效封装，写出来省时省力、可读性更佳。

**结论先行**：不需要掌握 TMP 的全部知识点。基础又高频的特性只有 10 个，掌握后即使是 gcc ranges 这种超多模板的代码也能看懂。

---

# 第一部分：读懂 ranges 必备的 10 个 TMP 特性

## 1. 概念约束（Concepts Constraints）

概念是 C++20 引入的核心特性，用于在编译时对模板参数进行约束检查。比传统 SFINAE 更直观、更强大，错误信息也更友好。ranges 用概念定义 range、view、iterator 等核心抽象，确保只有满足语义要求的类型才能被使用。

```cpp
// ranges::view 定义 - 要求类型必须是 range、可移动、可默认构造
template<typename _Tp>
concept view = ranges::range<_Tp> && movable<_Tp>
            && default_initializable<_Tp> && enable_view<_Tp>;

// filter_view 的概念约束 - 确保谓词类型和 range 类型兼容
template<input_range _Range, typename _Pred>
  requires viewable_range<_Range> && is_object_v<_Pred>
        && indirect_unary_predicate<const _Pred*, iterator_t<_Range>>
class filter_view : public view_interface<filter_view<_Range, _Pred>>;
```

## 2. SFINAE 和类型特性（SFINAE and Type Traits）

即便有了概念，gcc 的 ranges 仍用传统 SFINAE 和类型特性处理边缘情况或提供向后兼容。SFINAE（Substitution Failure Is Not An Error）让模板在替换失败时从候选集中移除而不报错；类型特性用于在编译时查询类型属性。这是概念出现之前约束模板的主要手段。

```cpp
// 传统 SFINAE 技术检测 range 概念
template<typename _Tp>
struct __is_range_impl : false_type { };

template<typename _Tp>
  requires requires { ranges::begin(std::declval<_Tp&>()); }
struct __is_range_impl<_Tp> : true_type { };

// 使用类型特征来启用视图特性
template<typename _Tp>
struct enable_view : public __is_derived_from_view_interface<_Tp> { };
```

## 3. 标签分发和特性萃取（Tag Dispatching and Trait Extraction）

标签分发基于类型标签在编译时选择不同实现；特性萃取用于获取类型的各种属性。在 ranges 中，迭代器类别标签用于根据迭代器能力选择最优算法实现；特性萃取模板用于获取 range 的迭代器类型、值类型等信息。

```cpp
// 迭代器类别标签体系 - 用于算法优化
struct input_iterator_tag { };
struct output_iterator_tag { };
struct forward_iterator_tag        : input_iterator_tag { };
struct bidirectional_iterator_tag  : forward_iterator_tag { };
struct random_access_iterator_tag  : bidirectional_iterator_tag { };

// 范围迭代器特性萃取 - 获取 range 的迭代器和哨兵类型
template<range _Range>
using iterator_t = decltype(ranges::begin(std::declval<_Range&>()));
template<range _Range>
using sentinel_t = decltype(ranges::end(std::declval<_Range&>()));
```

## 4. 编译时条件分支（Compile-time Conditional Branches）

`if constexpr` 是 C++17 引入的编译时条件语句。与运行时 `if` 不同，它在编译时就丢弃未选择的分支，不产生运行时开销，也能避免编译不存在的代码路径——这在模板元编程中特别有用。

```cpp
// 根据 range 特性选择不同的 begin 实现
template<typename _Range>
constexpr auto _M_begin() {
  if constexpr (common_range<_Range>) {
    // 对于 common_range，begin 和 end 类型相同，直接返回
    return ranges::begin(_M_base());
  } else {
    // 否则使用包装的迭代器类型
    return _Iterator{ranges::begin(_M_base())};
  }
}
```

## 5. 变量模板和类型别名（Variable Templates and Type Aliases）

变量模板允许将常量值作为模板参数；类型别名可创建复杂类型表达式的简写。两者在 ranges 中广泛用于定义类型特征、简化类型名称，使代码更易读、易维护。

```cpp
// 范围特性变量模板 - 提供编译时常量布尔值
template<typename _Tp>
inline constexpr bool enable_borrowed_range = false;
template<typename _Tp>
inline constexpr bool __is_sized_range = false;

// 类型别名模板 - 简化复杂类型表达式
template<range _Range>
using range_difference_t = iter_difference_t<iterator_t<_Range>>;
template<range _Range>
using range_value_t = iter_value_t<iterator_t<_Range>>;
```

## 6. 完美转发和引用折叠（Perfect Forwarding and Reference Collapsing）

完美转发将参数以原始值类别（左值/右值）传递给其他函数；引用折叠规则处理 `T&&` 模板参数中的引用组合。在 ranges 适配器中，完美转发确保参数高效传递并保持其值类别。

```cpp
// 视图工厂函数中的完美转发 - 保持参数的值类别
template<typename _Range>
  requires viewable_range<_Range>
constexpr auto all(_Range&& __r) {
  if constexpr (view<remove_cvref_t<_Range>>) {
    // 如果已经是 view，直接转发
    return std::forward<_Range>(__r);
  } else {
    // 否则创建 ref_view 包装
    return ref_view{std::forward<_Range>(__r)};
  }
}
```

## 7. CRTP（奇异递归模板模式，Curiously Recurring Template Pattern）

CRTP 通过让派生类作为模板参数传给基类来实现静态多态。在 ranges 中，`view_interface` 用 CRTP 为所有视图类型提供公共接口实现，既避免了虚函数开销，又实现了代码复用。

```cpp
// 视图接口基类使用 CRTP
template<typename _Derived>
class view_interface {
private:
  // 将 this 转换为派生类型
  constexpr _Derived& _M_derived() noexcept {
    return static_cast<_Derived&>(*this);
  }
public:
  // 为前向 range 提供 empty 实现
  constexpr bool empty() requires forward_range<_Derived> {
    return ranges::begin(_M_derived()) == ranges::end(_M_derived());
  }
};

// 具体视图继承自 view_interface
template<view _Base>
class transform_view : public view_interface<transform_view<_Base>>;
```

## 8. 约束的偏特化（Constrained Partial Specialization）

约束的偏特化允许根据概念或其他约束条件为模板提供不同实现。C++17 之前这种功能主要靠标签分发和 SFINAE 实现；现在可用 `if constexpr` 或概念提供更清晰的实现。传统做法则是特化模板或使用 `std::enable_if`。

```cpp
// 现代做法：使用 if constexpr 的条件实现
template<input_iterator _Iter, sentinel_for<_Iter> _Sent>
constexpr iter_difference_t<_Iter>
distance(_Iter __first, _Sent __last) {
  if constexpr (sized_sentinel_for<_Sent, _Iter>) {
    // 对于大小已知的哨兵，直接计算距离
    return __last - __first;
  } else {
    // 否则遍历计数
    iter_difference_t<_Iter> __n = 0;
    while (__first != __last) { ++__first; ++__n; }
    return __n;
  }
}

// 传统做法：通过标签分发的多重重载
template<typename _Iter>
constexpr iter_difference_t<_Iter>
__distance(_Iter __first, _Iter __last, input_iterator_tag) {
  // 输入迭代器实现
}
template<typename _Iter>
constexpr iter_difference_t<_Iter>
__distance(_Iter __first, _Iter __last, random_access_iterator_tag) {
  // 随机访问迭代器实现
}
```

## 9. 定制点对象（Customization Point Objects, CPOs）

CPO 是 C++20 Ranges 引入的重要技术，提供统一的函数调用语法来访问定制点。它是 `constexpr` 的函数对象，重载 `operator()`，通过约束模板确保类型安全，支持完美转发，并提供编译时 `noexcept` 检查。

```cpp
// <bits/ranges_base.h> 中的 CPO 实现示例
namespace ranges {
namespace __detail {
  // begin CPO 的实现
  struct _Begin {
  private:
    // 编译时 noexcept 检查
    template<typename _Tp>
    static constexpr bool _S_noexcept() {
      if constexpr (is_lvalue_reference_v<_Tp>)
        return noexcept(ranges::begin(std::declval<_Tp&>()));
      else
        return false;
    }
  public:
    // CPO 的核心 - 函数调用运算符
    template<typename _Tp>
      requires requires(_Tp& __t) { ranges::begin(__t); }
    constexpr auto operator()(_Tp&& __t) const
      noexcept(_S_noexcept<_Tp&>())
      -> decltype(ranges::begin(__t))
    { return ranges::begin(__t); }

    // 对内置数组的特化
    template<typename _Tp, size_t _Nm>
    constexpr _Tp* operator()(_Tp (&__arr)[_Nm]) const noexcept
    { return __arr; }

    // 对 std::initializer_list 的特化
    template<typename _Tp>
    constexpr const _Tp* operator()(initializer_list<_Tp> __il) const noexcept
    { return __il.begin(); }
  };

  // 类似的 end CPO
  struct _End {
    template<typename _Tp>
      requires requires(_Tp& __t) { ranges::end(__t); }
    constexpr auto operator()(_Tp&& __t) const
      -> decltype(ranges::end(__t))
    { return ranges::end(__t); }
  };
}

// 通过内联命名空间暴露 CPO 实例
inline namespace __cust {
  inline constexpr __detail::_Begin begin{};
  inline constexpr __detail::_End   end{};
  inline constexpr __detail::_Size  size{};
  inline constexpr __detail::_Data  data{};
}
}
```

**CPO 的关键特性**：

- 统一的函数调用语法：`ranges::begin(container)`
- 编译时约束检查：确保参数满足特定概念
- 完美转发支持：保持值类别
- noexcept 正确性：编译时 noexcept 推导
- 支持特化：为特定类型提供优化实现

## 10. ADL 屏障（ADL Barriers）

ADL 屏障是一种命名空间设计技术，用于控制参数依赖查找（ADL）的范围，防止不期望的函数重载被找到。通过把实现细节放在**非内联**的命名空间中形成屏障，确保定制点查找只在预期命名空间中进行。

```cpp
namespace ranges {
  // ADL 屏障：实现细节放在非内联命名空间
  namespace __detail {
    // 实际的 CPO 实现类 - 对外部不可见
    struct _Begin { /* 实现同上 */ };
    struct _End   { /* 实现同上 */ };

    // 其他辅助函数也放在这里，避免 ADL 问题
    template<typename _Iter>
    constexpr auto __advance(_Iter& __it, iter_difference_t<_Iter> __n) {
      // 实现细节，不会被 ADL 找到
    }
  }

  // 屏障外层：通过内联命名空间暴露接口
  inline namespace __cust {
    // 只有这些 CPO 实例对外可见
    inline constexpr __detail::_Begin begin{};
    inline constexpr __detail::_End   end{};
  }

  // 另一个例子：ranges::swap 的 ADL 屏障设计
  namespace __detail {
    struct _Swap {
      template<typename _Tp, typename _Up>
        requires requires(_Tp&& __t, _Up&& __u) {
          swap(static_cast<_Tp&&>(__t), static_cast<_Up&&>(__u));
        }
      constexpr void operator()(_Tp&& __t, _Up&& __u) const
        noexcept(noexcept(swap(static_cast<_Tp&&>(__t), static_cast<_Up&&>(__u))))
      { swap(static_cast<_Tp&&>(__t), static_cast<_Up&&>(__u)); }
    };
  }
  // swap 放在单独的内联命名空间
  inline namespace __cust_swap {
    inline constexpr __detail::_Swap swap{};
  }
}
```

**ADL 屏障的工作原理**（使用示例）：

```cpp
namespace my_lib {
  struct MyContainer {
    int* data;
    size_t size;
    int* begin() { return data; }
    int* end()   { return data + size; }
  };
  // 自定义 begin 函数 - 但不会被 ranges::begin 调用
  void begin(MyContainer&) { /* 这个不会被调用 */ }
}

void example() {
  my_lib::MyContainer c;
  // 由于 ADL 屏障，只会调用 MyContainer 的成员函数 begin，
  // 不会找到 my_lib 命名空间中的自由函数 begin
  auto it = ranges::begin(c);
}
```

---

# 第二部分：实战 —— 为 gcc10 的 ranges 写自定义 view

掌握了上面 10 个特性，就有能力读懂、甚至扩展 ranges 源码。标准库提供的 view 有时不够用——C++20 连 `cycle`、`zip` 这样常见的 view 都没有。下面就以自定义这两个 view（准确说是 view adaptor）为例，演示完整套路。

## 目标效果：干支纪年

干支纪年法由 10 个『天干』和 12 个『地支』轮流组合而来，60 年一个轮回。正好用来演示 `cycle` 和 `zip`：

```cpp
namespace vs = std::ranges::views;

int main() {
  // 10 天干
  std::vector<std::string> v1 {
    "甲","乙","丙","丁","戊","己","庚","辛","壬","癸"
  };
  // 12 地支
  std::vector<std::string> v2 {
    "子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"
  };

  auto a = v1 | vs::cycle;              // 天干无限循环
  auto b = v2 | vs::cycle;             // 地支无限循环
  auto c = vs::zip(a, b) | vs::take(60); // 配对，取 60 个

  // 输出：甲子 乙丑 丙寅 丁卯 戊辰 己巳 庚午 辛未
  //       壬申 癸酉 甲戌 乙亥 丙子 丁丑 戊寅 ……
  for (auto&& [x, y] : c)
    std::cout << x << y << " ";
  std::cout << "\n";
  return 0;
}
```

其中的 `cycle` 和 `zip` 就是要实现的自定义 view adaptor。

## 自定义 view 的四个步骤

1. 定义一个 view 类，从 `view_base` 继承。标准库定义了 `view_base` 的子类 `view_interface`，直接从 `view_interface` 继承可以省一些事情（参见第一部分 CRTP）；
2. 为 view 类定义 iterator 和 sentinel，实现 `*`、`++` 之类的运算符；
3. 为 view 定义 `begin()` 和 `end()` 函数；
4. 为 view 定义 adaptor。

## 关于"代码丑化"（uglification）命名风格

gcc 标准库代码里有很多下划线，这种风格是**故意的**，叫做代码丑化（uglification），目的是避免与用户标识符冲突：

| 风格 | 含义 |
| --- | --- |
| `_M_xxx` | 类成员（member） |
| `_Xxx`（下划线 + 大写字母开头） | 类名或模板参数类型名 |
| `__xxx`（双下划线 + 小写字母） | 局部变量 |

刚开始看着很丑，习惯了其实还能忍受。"丑没关系，只要丑出风格丑出水平，那就是牛的东西。"

## cycle_view 的核心：在 `operator++` 中绕回

`cycle_view` 就是想要一个序列的无限循环：到了最后一个元素以后又绕回第一个。实现方法很简单——在 iterator 的 `++` 里做一个判断即可：

```cpp
constexpr _Iterator& operator++() {
  ++_M_current;
  if (_M_current == ranges::end(_M_parent->_M_base))
    // 到最后了，将迭代器放回到第一个
    _M_current = ranges::begin(_M_parent->_M_base);
  return *this;
}
```

`zip_view` 也没什么特别的：保存两个序列的指针，每次要返回两个值，用 `pair` 返回即可。

## 定义 adaptor：一个 lambda 搞定

实现 view 以后需要定义 adaptor。gcc10 中定义 adaptor 很简单，只要定义一个 lambda：

- 对于**构造函数只有一个参数**的 view，返回 `__adaptor::_RangeAdaptorClosure` 类型；
- 对于**构造函数参数在一个以上**的 view，返回 `__adaptor::_RangeAdaptor` 类型。

本文两个 view 刚好分别演示这两种情况。以 `cycle_view`（单参数）为例：

```cpp
// 推导指引
template<input_range _Range>
cycle_view(_Range&&) -> cycle_view<views::all_t<_Range>>;

namespace views {
  inline constexpr __adaptor::_RangeAdaptorClosure cycle
    = [] <viewable_range _Range> (_Range&& __r) {
        return cycle_view { std::forward<_Range>(__r) };
      };
}
```

## 小结与进阶方向

按照 gcc10 ranges 头文件里其它 view 的代码"依葫芦画瓢"，实现自己想要的 view 非常简单。但如果要做到工业级，还有很多事要做：

- 实现更多的运算符（不只是 `*` 和 `++`）；
- 为提高运行效率，针对各种 concept 进行重载（呼应第一部分的「约束的偏特化」「标签分发」）；
- 为提高拷贝速度，考虑如何减少成员变量。

总之，自己动手实现一遍才会有更深的体会。**ranges 的时代已经来了。**

---

## 两篇文章的连接点

| TMP 特性（第一部分） | 在自定义 view（第二部分）中的体现 |
| --- | --- |
| 概念约束 | `input_range`、`viewable_range` 约束 view 的模板参数 |
| CRTP | 从 `view_interface` 继承获得 `empty()`、`size()` 等默认实现 |
| 完美转发 | adaptor lambda 中 `std::forward<_Range>(__r)` |
| 类型别名 | `views::all_t<_Range>` 推导 view 持有的底层类型 |
| 定制点对象 / CPO | `ranges::begin/end` 在 `operator++` 中驱动迭代 |
| 约束的偏特化 | 针对不同 concept 重载以优化效率 |

读懂 ranges 源码（第一部分）→ 动手写 view（第二部分），正是学习现代 C++ TMP 最高效的闭环。
