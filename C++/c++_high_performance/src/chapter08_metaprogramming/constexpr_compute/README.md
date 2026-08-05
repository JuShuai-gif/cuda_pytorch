# constexpr_compute

constexpr、if constexpr 与编译期验证。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 222-227 页：

- **`constexpr`**：输入为编译期常量时在编译期求值，否则退回运行期；
- **`std::integral_constant`**：要求模板参数是常量表达式，可**证明**编译期计算；
- **`if constexpr`**：消除假分支（普通 `if` 仍编译所有分支，会失败）。

## 构建与运行

```bash
cmake --build build --target ch08_constexpr_example ch08_constexpr_tests
./build/chapter08_metaprogramming/ch08_constexpr_example
./build/chapter08_metaprogramming/ch08_constexpr_tests
```

## 关键点

- `static_assert(sum(3,4,5) == 12)` 证明编译期求值；
- `integral_constant<int, sum(1,2,3)>` 编译通过（否则不编译）；
- `generic_mod` 用 `if constexpr`：float 走 `fmod`，int 走 `%`，互不干扰；
- 编译期哈希 `hash_function("abc")` 汇编为 `mov $294`（见 note/08 §13）。

## 注意

- 编译期求值有代价：复杂 constexpr 增加编译时间；
- constexpr 不代表一定编译期求值（运行期调用照常执行）。
