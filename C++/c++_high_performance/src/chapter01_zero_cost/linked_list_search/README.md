# linked_list_search

C 风格链表搜索 vs STL 算法搜索。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 25-26 页给出了同一问题的两种写法：

- C 风格：`struct string_elem_t` 手写链表 + `for` 循环 + `strcmp`
- C++ 风格：`std::list<std::string>` + `std::count(...)`

原书观点：两个版本"翻译成大致相同的机器码"，但 C++ 版本隐藏了指针、循环和比较细节，
同时（关键）**抽象不增加运行时成本**。

## 文件

| 文件 | 说明 |
|---|---|
| `baseline.hpp/cpp` | C 风格链表 + 手工循环 + `strcmp` |
| `optimized.hpp/cpp` | `std::vector`/`std::list` + `std::count` |
| `benchmark.cpp` | 3 种实现对比（1M 本书，200 个书名池） |
| `tests.cpp` | 正确性等价性 |

## 构建与运行

```bash
cmake --build build --target ch01_lls_benchmark ch01_lls_tests
./build/chapter01_zero_cost/ch01_lls_tests
./build/chapter01_zero_cost/ch01_lls_benchmark
```

## 结果解释（2026-08-05，GCC 13.3 Release，i9-14900HX）

| 实现 | mean | 说明 |
|---|---|---|
| C 风格链表 + 手工循环 | ~2.9 ms | 每个节点独立分配，遍历时逐节点 cache miss |
| `std::list` + `std::count` | ~3.3 ms | 同样是链式存储，与 C 版本同量级 |
| `std::vector` + `std::count` | ~1.5 ms | 连续存储，缓存友好，约快 2 倍 |

结论（限定本次环境）：

- `std::count` 抽象本身没有带来额外成本：C 风格与 `std::list` 版本性能相当。
- 差距来自**存储布局**（链表 vs 连续内存），而非语言抽象。
- 原书用 `std::list` 作示例；本项目额外加入 `std::vector` 以展示连续存储优势，
  与 Chapter 4 的 `contiguous_vs_pointer` 呼应。

## 观察点

- `std::count` 是模板，`-O3` 下完全内联，不产生函数调用。
- 遍历链表时的缓存不命中（cache miss）主导耗时，与算法本身无关。
