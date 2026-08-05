# associative_containers

有序 vs 无序关联容器。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 112-114 页：

- **有序**（`set`/`map`/`multiset`/`multimap`）：平衡树（红黑树），
  要求 `<`，增删查 O(log n)；
- **无序**（`unordered_set`/`map` 等）：哈希表，要求 `==` 与哈希函数，
  增删查平均 O(1)。

## 构建与运行

```bash
cmake --build build --target ch04_associative_benchmark ch04_associative_tests
./build/chapter04_data_structures/ch04_associative_tests
./build/chapter04_data_structures/ch04_associative_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，100 万键查找）

| 容器 | mean | 相对 |
|---|---|---|
| `std::map`（红黑树） | ~15.0 ms | 8.9x |
| `std::unordered_map`（哈希表） | ~1.7 ms | 1.0x |

哈希表快约 9 倍。书中的提醒（PDF 114 页）：理论 O(1) vs O(log n)，
实际差异在小容器上不明显，数据量大才显著。

## 结论（限定本环境）

- 需要有序遍历、范围查询 → `map`/`set`；
- 只需要按键查找、数据量大 → `unordered_map`；
- 注意无序容器的哈希质量与 load_factor（见 `hash_policy`）。
