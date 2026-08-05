# partial_sorting

`sort` / `partial_sort` / `nth_element`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 160-162 页：很多时候只需部分排序结果（中位数、前 m 个）。

| 算法 | 复杂度 | 用途 |
|---|---|---|
| `std::sort` | O(n log n) | 全排序 |
| `std::partial_sort` | O(n log m) | 前 m 个有序 |
| `std::nth_element` | O(n) | 第 n 个元素就位（前 n 无序） |

书中实测（10M 元素，m=1M）：找中位数快 12.4x，部分区间 4.6-8.7x。

## 构建与运行

```bash
cmake --build build --target ch06_partial_sorting_benchmark ch06_partial_sorting_tests
./build/chapter06_algorithms/ch06_partial_sorting_tests
./build/chapter06_algorithms/ch06_partial_sorting_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，2M 元素，取前 100k）

| 算法 | 相对 |
|---|---|
| `std::sort` | 1.0x |
| `std::partial_sort` | ~3.1x 快 |
| `std::nth_element` | ~8.7x 快 |

## 关键点

- `nth_element` 不保证前段有序，只保证第 n 个元素就位；
- 找中位数：`nth_element(begin, begin+n/2, end)`；
- tests 验证三种算法的语义正确性。
