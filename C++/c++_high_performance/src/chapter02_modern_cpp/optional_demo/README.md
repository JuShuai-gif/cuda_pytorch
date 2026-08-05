# optional_demo

`std::optional` 的用法与开销。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 80-82 页：

- `std::optional` 是**栈分配**的、最多一个元素的容器；
- 内存开销 = 一个 `bool`（加 padding）；
- 适用场景：可选返回值（如线段交点）、可选成员变量（如帽子）；
- 比较规则：空 < 非空，两个空相等 → 排序时空 optional 排最前。

## 构建与运行

```bash
cmake --build build --target ch02_optional_example ch02_optional_benchmark ch02_optional_tests
./build/chapter02_modern_cpp/ch02_optional_example
./build/chapter02_modern_cpp/ch02_optional_tests
./build/chapter02_modern_cpp/ch02_optional_benchmark
```

## 结果解释

尺寸（本环境）：`sizeof(optional<int>)=8`（int 4 + bool 1 + padding 3），
`sizeof(optional<Point>)=24`（Point 16 + bool 1 + padding 7）。

Benchmark（1M 元素访问）：`vector<optional<int>>` 带 `has_value()` 检查
比直接 `vector<int>` 慢约 2.4 倍——差异来自额外的 bool 检查与元素变大
导致的缓存行翻倍。

## 注意事项

- 访问空 optional：`operator*` 是 UB，`value()` 才抛 `bad_optional_access`；
- 与哨兵值/指针/`variant` 的对比见 `optional_variant_pointer` 实验。
