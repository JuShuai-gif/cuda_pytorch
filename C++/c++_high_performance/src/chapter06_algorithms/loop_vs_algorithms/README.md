# loop_vs_algorithms

手写循环 vs STL 算法。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 153-159 页：STL 算法的优势——性能（隐藏优化）、安全（边界情况）、
未来可并行化（Chapter 11）、文档完善、可读性（意图由名字表达）。

本实验对比 6 种常见操作的手写循环与 STL 算法版本。

## 构建与运行

```bash
cmake --build build --target ch06_loop_vs_algos_benchmark ch06_loop_vs_algos_tests
./build/chapter06_algorithms/ch06_loop_vs_algos_tests
./build/chapter06_algorithms/ch06_loop_vs_algos_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，4M int）

| 操作 | 手写 | std 算法 | 相对 |
|---|---|---|---|
| count | ~482 µs | ~421 µs | 1.14x |
| accumulate | ~268 µs | ~259 µs | 1.03x |
| transform | ~2.02 ms | ~1.16 ms | 1.75x |
| copy_if | ~10.7 ms | ~10.9 ms | ~1.0x |

- transform 用 STL 快 1.75x（可能因 reserve+预分配）；
- copy_if 相当（两者都遍历+条件 push_back）；
- 差距随数据规模与编译器变化，**不构成"STL 一定更快"的结论**；
- 一致性：所有 checksum 匹配（tests 验证等价性）。

## 结论

- 即使性能相当，STL 算法在可读性、安全性与未来并行化上更优；
- 书中 find 的展开优化（`find_unroll`）是"免费拿到隐藏优化"的例子。
