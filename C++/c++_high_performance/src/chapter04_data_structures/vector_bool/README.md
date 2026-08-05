# vector_bool

`std::vector<bool>` 特化行为。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 127 页：`std::vector<bool>` 不是标准 bool 向量，而是**位数组**。
优点：内存紧凑（每 bool 1 bit），`count`/`find` 可一次处理 64 bit（极快）。
缺点：`operator[]` 返回代理引用而非 `bool&`，行为与普通容器不同。

书中提到其未来不确定，可能被 `std::bitset` 与动态位集替代（Boost 已有
`boost::dynamic_bitset`）。

## 构建与运行

```bash
cmake --build build --target ch04_vector_bool_example
./build/chapter04_data_structures/ch04_vector_bool_example
```

## 关键观察

- `sizeof(vector<bool>)` 与 `vector<char>` 相同（都存指针+大小，无内联存储）；
- `bits[0]` 返回代理对象，`decltype(bits[0])` 不是 `bool&`（static_assert）；
- `count` 内部逐位块处理，很快；
- `flip()` 反转所有位。

## 注意事项

- 若需要"真正的 bool 数组"（可取 `bool&`），用 `vector<char>`/`deque<bool>`；
- 作为位图使用时 `vector<bool>` 很高效（见 `aos_vs_soa` 中 playing 数组）。
