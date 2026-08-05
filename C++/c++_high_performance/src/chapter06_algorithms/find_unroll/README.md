# find_unroll

`std::find` 的隐藏优化：4 路展开 + 与零比较。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 158-159 页：libstdc++ 的 `std::find_if` 把循环 4 路展开，使
`it != last` 的比较次数减少 3/4；并让 trip_count 递减与零比较
（x86 上用 `test` 而非 `cmp`，略快）。

书中实测（10M 元素）：`find_slow` 3420µs vs `find_fast` 3402µs，仅快 0.5%。
**结论：收益很小，但用 STL 免费获得。**

## 构建与运行

```bash
cmake --build build --target ch06_find_unroll_benchmark ch06_find_unroll_tests
./build/chapter06_algorithms/ch06_find_unroll_tests
./build/chapter06_algorithms/ch06_find_unroll_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，10M int 查找存在值）

| 实现 | mean | 相对 |
|---|---|---|
| `find_slow`（朴素循环） | ~130 ns | 1.0x |
| `find_fast`（4 路展开） | ~87 ns | 0.67x |
| `std::find`（libstdc++） | ~86 ns | 0.66x |

本环境展开版比朴素版快约 1.5 倍（书中仅 0.5%），差异取决于编译器是否
已自动展开朴素循环、CPU 与数据规模。`std::find` 与手写展开版相当——
验证 libstdc++ 确实采用该技巧。

## 注意

- 不要为了"与零比较"这种微优化手写循环而牺牲可读性（书中明确建议）；
- 尾段 `switch` 处理 <4 的余数（tests 覆盖 0..3 长度边界）。
