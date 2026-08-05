# reflection

最小反射：`reflect()` + 泛型运算符。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 242-248 页：

- C++ 无内建反射 → 用 `reflect()` 成员把成员暴露为 `std::tie` 元组；
- 用 `is_detected` + `enable_if_t` 为所有"可反射"类型自动生成
  `operator==`/`operator!=`/`operator<`/`operator<<`；
- 书中证明反射生成的汇编与手写运算符**完全一致**（零开销）。

## 构建与运行

```bash
cmake --build build --target ch08_reflection_example ch08_reflection_tests
./build/chapter08_metaprogramming/ch08_reflection_example
./build/chapter08_metaprogramming/ch08_reflection_tests
```

## 关键点

- 加/删成员只需改 `reflect()`，所有泛型运算符自动适配；
- 非反射类型（如 int）不会获得这些运算符（`static_assert` 验证）；
- 手写 `operator<` 复杂，`std::tie` 让字典序比较一行搞定。

## 现代补充

> 现代补充：成熟的反射库有 Boost.Hana（宏）、Boost.PFR（自动反射简单成员）、
> 以及 PFR 思想启发的手写方案；C++26 标准反射仍在演进。
