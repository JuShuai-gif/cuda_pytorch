# type_traits

类型萃取、decltype、enable_if、is_detected。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 214-221 页：

- **布尔型 traits**（`is_same_v`/`is_floating_point_v`）：编译期回答类型问题；
- **类型变换 traits**（`remove_pointer_t`/`add_pointer_t`/`decay_t`）：返回新类型；
- **`decltype`**：取变量/表达式类型；配合 `remove_reference_t` 处理引用；
- **`std::enable_if_t`**：按编译期谓词条件启用/禁用模板函数；
- **`std::experimental::is_detected`**：检查类是否含某成员（成员函数/成员变量/typedef）。

## 构建与运行

```bash
cmake --build build --target ch08_type_traits_example ch08_type_traits_tests
./build/chapter08_metaprogramming/ch08_type_traits_example
./build/chapter08_metaprogramming/ch08_type_traits_tests
```

## 关键点

- `sign_func` 对 unsigned 直接返回 1（`if constexpr` 消除死分支）；
- `is_detected` 探测 `to_string()`/`name_`，`print()` 按能力分派；
- `is_detected` 是实验性库（`<experimental/type_traits>`，GCC/Clang 提供）。

## 现代补充

> 现代补充：C++20 Concepts 提供了更优雅的约束语法（`requires` 子句），
> 可替代部分 enable_if 用法；`std::is_detected` 仍未入标准（建议用 concepts 或
> 自实现 detection idiom，见 note/08）。
