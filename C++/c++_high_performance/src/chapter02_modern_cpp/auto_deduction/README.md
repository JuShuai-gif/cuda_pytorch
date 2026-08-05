# auto_deduction

auto 类型推导的四种形式。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 44-47 页介绍了 `auto` 的四种形式及其语义：

| 形式 | 语义 | 注意 |
|---|---|---|
| `auto` | 值拷贝（退化掉引用/const） | 会复制对象 |
| `const auto&` | const 引用，可绑定临时对象 | 默认推荐 |
| `auto&` | 可变引用，不可绑定临时对象 | 仅当要修改时用 |
| `auto&&` | 转发引用，可绑定一切 | 仅用于转发 |

C++17 的复制消除保证：`auto x = Foo{};` 与 `Foo x{};` 等价，无临时对象（PDF 第 46 页）。

## 构建与运行

```bash
cmake --build build --target ch02_auto_example
./build/chapter02_modern_cpp/ch02_auto_example
```

## 关键点

- 用 `static_assert` 验证推导类型：`decltype(foo.val())` 为 `int`，`decltype(foo.cref())` 为 `const int&`；
- 值拷贝 `v` 在 `mr = 100` 之后仍为 42，引用 `mr`/`cr` 看到 100；
- `const auto&` 与 `auto&&` 都能绑定 `make_string()` 返回的临时对象并延长其生命周期；
- `auto&` 不能绑定临时对象（编译错误，此处未展示）。
