# loop_vs_count

手写循环 vs `std::count`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

Chapter 1 与 Chapter 6 都讨论了"手写 for 循环 vs STL 算法"。
`std::count` 是模板，内联后与手写循环等价，但 STL 实现通常有更好的向量化/展开。

## 文件

| 文件 | 说明 |
|---|---|
| `baseline.cpp` | 手写 `for` + `if` 计数 |
| `optimized.cpp` | `std::count` |
| `benchmark.cpp` | 4M 个 `int`，统计 `5` 出现次数 |
| `tests.cpp` | 等价性（含 100 个 needle 的随机大输入对比） |

## 构建与运行

```bash
cmake --build build --target ch01_lvc_benchmark ch01_lvc_tests
./build/chapter01_zero_cost/ch01_lvc_tests
./build/chapter01_zero_cost/ch01_lvc_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX）

| 实现 | mean | checksum |
|---|---|---|
| 手写 for 循环 | ~575 µs | 10825704 |
| `std::count` | ~462 µs | 10825704 |

本环境观察：`std::count` 约快 20%，checksum 一致。差异原因待确认——可能来自
标准库循环的展开方式（与直接 `values[i]` 相比，迭代器版本便于编译器做更宽的处理），
也与 4M 数据量下每次命中约 10% 的分支可预测性有关。

注意：这是"本次输入规模和当前编译器"的观察，不是"STL 一定更快"的结论。
在汇编层面（见 `zero_cost_asm`）两者生成几乎相同的向量化循环，差异通常来自
循环结构细节而非抽象本身。

## 观察点

- 若手写循环版本与 `std::count` 汇编一致，说明抽象零成本；
- 若有差异，用 `-O3 -S` 对比 `.L39` 风格的内层循环体。
