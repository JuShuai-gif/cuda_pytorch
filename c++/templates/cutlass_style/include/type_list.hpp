#pragma once

#include <cstddef>
#include <type_traits>

namespace cutlass_style {

// ============================================================================
// TypeList<Ts...> - 编译期类型列表
// ============================================================================
//
// WHY: NVIDIA 工程师为什么发明 TypeList？
//
// 类比: TypeList 之于模板元编程，相当于 std::vector 之于运行时编程。
//       运行时我们用 vector<T> 存数据，编译期我们用 TypeList<Ts...> 存类型。
//
// GPU kernel 配置有大量组合:
//   - 3种数据类型 × 3种layout × 4种tile大小 × 3种架构 = 108种组合
//   这些组合必须在编译期确定，因为每个组合会实例化出完全不同的 PTX 代码。
//
// 传统做法: 写 108 个 if-else 分支。编译器仍然会为所有分支生成代码，
//           导致编译时间爆炸、二进制膨胀。
//
// CUTLASS 做法: 将 108 种组合存为 TypeList，用编译期算法 (find_if, filter)
//               在类型列表中选择，编译器只实例化被选中的那个 kernel。
//
// 模板展开后: TypeList<float, half, double> 不产生任何运行时代码。
//             类型列表本身是"零开销抽象"——只存在于编译器的符号表里。

template <typename... Ts>
struct TypeList {
  // 编译期长度 - 类比 std::vector::size()
  static constexpr std::size_t size = sizeof...(Ts);

  // 编译期判空 - 类比 std::vector::empty()
  static constexpr bool empty = (size == 0);
};

// ============================================================================
// at_t<Index, TypeList> - 按索引取类型
// ============================================================================
//
// 类比: TypeList 的 operator[]，就像 vector[3] 返回第4个元素。
//       但这里是编译期的——返回的是一个类型，不是值。
//
// 模板展开后:
//   at_t<2, TypeList<float, half, bf16, int8>>
//   → 递归展开 3 层
//   → 最终编译器看到: using result = bf16;

namespace detail {

// 递归实现: 每次剥离头部一个类型，Index 减 1
template <std::size_t Index, typename... Ts>
struct at_impl;

// 终止条件: Index == 0，当前头部类型就是答案
template <typename Head, typename... Tail>
struct at_impl<0, Head, Tail...> {
  using type = Head;
};

// 递归步骤: 跳过 Head，在 Tail 中找 Index-1
template <std::size_t Index, typename Head, typename... Tail>
struct at_impl<Index, Head, Tail...> {
  // WHY 递归而非循环: C++ 模板是纯函数式的。没有可变状态，
  // 只能用递归展开。编译器展开这个递归后，结果等同于直接写类型名。
  using type = typename at_impl<Index - 1, Tail...>::type;
};

} // namespace detail

template <std::size_t Index, typename TypeList>
struct at;

template <std::size_t Index, template <typename...> class List, typename... Ts>
struct at<Index, List<Ts...>> {
  using type = typename detail::at_impl<Index, Ts...>::type;
};

template <std::size_t Index, typename TypeList>
using at_t = typename at<Index, TypeList>::type;

// ============================================================================
// index_of_t<Query, TypeList> - 查找类型在列表中的位置
// ============================================================================
//
// 类比: 就像在 vector 中 std::find，但返回编译期常量。
//
// 用途: 多维 dispatch 时，需要知道某个配置在配置列表中的序号，
//       用于索引 shared memory 偏移表、寄存器分配表等。

namespace detail {

template <std::size_t Index, typename Query, typename... Ts>
struct index_of_impl;

template <std::size_t Index, typename Query, typename Head, typename... Tail>
struct index_of_impl<Index, Query, Head, Tail...> {
  static constexpr std::size_t value =
      std::is_same_v<Query, Head>
          ? Index
          : index_of_impl<Index + 1, Query, Tail...>::value;
};

template <std::size_t Index, typename Query>
struct index_of_impl<Index, Query> {
  // 未找到时返回极大值，相当于 npos
  static constexpr std::size_t value = static_cast<std::size_t>(-1);
};

} // namespace detail

template <typename Query, typename TypeList>
struct index_of;

template <typename Query, template <typename...> class List, typename... Ts>
struct index_of<Query, List<Ts...>> {
  static constexpr std::size_t value = detail::index_of_impl<0, Query, Ts...>::value;
};

template <typename Query, typename TypeList>
inline constexpr std::size_t index_of_v = index_of<Query, TypeList>::value;

// ============================================================================
// contains_v<Query, TypeList> - 编译期包含检测
// ============================================================================
//
// WHY: 用于 static_assert 验证用户传入的配置是否在支持列表中。
//      比如只支持 Sm80+ 的 kernel，如果用户传 Sm75，编译期直接报错，
//      而不是运行时崩溃。

template <typename Query, typename TypeList>
inline constexpr bool contains_v = (index_of_v<Query, TypeList> != static_cast<std::size_t>(-1));

// ============================================================================
// filter_t<Predicate, TypeList> - 编译期类型过滤
// ============================================================================
//
// 类比: Python 的 [x for x in list if pred(x)]，但在编译期执行。
//
// 实际场景: 从 108 种配置中，选出所有支持 Sm80 的配置。
//           filter_t 返回一个新 TypeList，只包含满足条件的类型。
//
// 模板展开后: 编译器对每个类型执行 Predicate<T>::value 检测，
//             生成一个新的 TypeList。整个过程在 -O0 也零开销。

namespace detail {

// Predicate<T> 必须提供 static constexpr bool value
template <template <typename> class Predicate, typename Result, typename... Remaining>
struct filter_impl;

template <template <typename> class Predicate, typename... Result, typename Head, typename... Tail>
struct filter_impl<Predicate, TypeList<Result...>, Head, Tail...> {
  using type = std::conditional_t<
      Predicate<Head>::value,
      typename filter_impl<Predicate, TypeList<Result..., Head>, Tail...>::type,
      typename filter_impl<Predicate, TypeList<Result...>, Tail...>::type>;
};

template <template <typename> class Predicate, typename... Result>
struct filter_impl<Predicate, TypeList<Result...>> {
  using type = TypeList<Result...>;
};

} // namespace detail

template <template <typename> class Predicate, typename TypeList>
struct filter;

template <template <typename> class Predicate, template <typename...> class List, typename... Ts>
struct filter<Predicate, List<Ts...>> {
  using type = typename detail::filter_impl<Predicate, TypeList<>, Ts...>::type;
};

template <template <typename> class Predicate, typename TypeList>
using filter_t = typename filter<Predicate, TypeList>::type;

// ============================================================================
// transform_t<Mapper, TypeList> - 编译期类型映射
// ============================================================================
//
// 类比: Python 的 map(func, list)，但在编译期。
//
// 实际场景: TypeList<Sm70, Sm75, Sm80> → TypeList<Sm70Traits, Sm75Traits, Sm80Traits>
//           对每个架构标签映射到对应的 trait 类。

namespace detail {

template <template <typename> class Mapper, typename... Ts>
struct transform_impl;

template <template <typename> class Mapper, typename... Result, typename Head, typename... Tail>
struct transform_impl<Mapper, TypeList<Result...>, Head, Tail...> {
  using type = typename transform_impl<Mapper, TypeList<Result..., typename Mapper<Head>::type>, Tail...>::type;
};

template <template <typename> class Mapper, typename... Result>
struct transform_impl<Mapper, TypeList<Result...>> {
  using type = TypeList<Result...>;
};

} // namespace detail

template <template <typename> class Mapper, typename TypeList>
struct transform;

template <template <typename> class Mapper, template <typename...> class List, typename... Ts>
struct transform<Mapper, List<Ts...>> {
  using type = typename detail::transform_impl<Mapper, TypeList<>, Ts...>::type;
};

template <template <typename> class Mapper, typename TypeList>
using transform_t = typename transform<Mapper, TypeList>::type;

// ============================================================================
// concat_t<TypeListA, TypeListB> - 编译期类型列表拼接
// ============================================================================

template <typename ListA, typename ListB>
struct concat;

template <typename... As, typename... Bs>
struct concat<TypeList<As...>, TypeList<Bs...>> {
  using type = TypeList<As..., Bs...>;
};

template <typename ListA, typename ListB>
using concat_t = typename concat<ListA, ListB>::type;

// ============================================================================
// for_each_type<TypeList, Lambda> - C++20 编译期类型遍历 (使用 constexpr lambda)
// ============================================================================
//
// WHY C++20: C++17 以前无法在 constexpr 上下文中使用 lambda。
//            C++20 的 constexpr lambda 让类型遍历变得像运行时 for 循环一样直观。
//
// 模板展开后:
//   for_each_type<TypeList<float, half, bf16>>([]<typename T>(){
//       Kernel<T>::launch();
//   });
//   → 编译器展开为 3 个独立的函数调用:
//       Kernel<float>::launch();
//       Kernel<half>::launch();
//       Kernel<bf16>::launch();
//   → 每个实例化都是独立编译的，可以利用所有编译期优化。

template <typename TypeList, typename Func>
constexpr void for_each_type(Func&& func) {
  // C++20: 使用折叠表达式展开
  [&]<template <typename...> class List, typename... Ts>(List<Ts...>) {
    (func.template operator()<Ts>(), ...);
  }(TypeList{});
}

} // namespace cutlass_style
