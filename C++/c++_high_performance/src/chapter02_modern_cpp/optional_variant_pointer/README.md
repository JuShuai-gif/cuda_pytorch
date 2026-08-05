# optional_variant_pointer

四种"可能为空"值表示的对比。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

表示"可能没有值"有四种常见方案：

| 方案 | 空表示 | 优点 | 缺点 |
|---|---|---|---|
| 哨兵值 | 特殊值（如 -1） | 零开销 | 值域被污染，易混淆 |
| 指针 | `nullptr` | 简单 | 需堆分配或指向外部存储 |
| `std::optional<T>` | 内部 bool | 类型安全，栈分配 | 一个 bool + padding 开销 |
| `std::variant<monostate,T>` | `monostate` | 类型安全，可扩展多状态 | 对齐到最大成员 |

PDF 第 80-82 页（optional）、238 页（variant）。

## 构建与运行

```bash
cmake --build build --target ch02_optional_variant_example \
      ch02_optional_variant_benchmark ch02_optional_variant_tests
./build/chapter02_modern_cpp/ch02_optional_variant_example
./build/chapter02_modern_cpp/ch02_optional_variant_tests
./build/chapter02_modern_cpp/ch02_optional_variant_benchmark
```

## 结果解释（GCC 13.3 Release）

尺寸：`int=4`，`int*=8`，`optional<int>=8`，`variant<monostate,int>=8`
（都对齐到 4 字节边界，optional 的 1 字节 bool 被 padding 到 4）。

Benchmark（1M 读取，全部存在）：

| 方案 | mean | 相对 |
|---|---|---|
| 哨兵值 | ~209 µs | 1.0x |
| `std::variant` | ~271 µs | 1.3x |
| 指针 | ~283 µs | 1.4x |
| `std::optional` | ~290 µs | 1.4x |

## 结论（限定本环境）

- 哨兵值最快但最不安全的表示；
- `optional`/`variant` 类型安全、栈分配，代价约 30-40% 的访问开销；
- 指针额外引入间接访存/堆分配；
- 在"值可能为空"的表达上，`optional`/`variant` 的安全收益通常远超这点开销。
