# 07_container —— 数据结构与容器

对应笔记：`note/06_内存与缓存优化.md`（9.6/9.7）
对应 PDF：第 95-105 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `07_baseline` | `push_back` 不 reserve（反复重分配）+ `std::list`（逐元素分配） |
| `07_optimized` | `reserve()` 预留容量 + 复用容量 |
| `07_benchmark` | 遍历（vector/list/deque）、查找（map/unordered_map）、new/delete vs 内存池 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 07_baseline 07_optimized 07_benchmark -j

./build/07_container/07_benchmark
./build/07_container/07_baseline
./build/07_container/07_optimized
```

## 预期结果与解读

- `traverse_vector` 明显快于 `traverse_list`：list 逐元素分配、内存不连续、还有指针依赖链（PDF 第 96-97 页）。
- `map_lookup`（红黑树，指针追逐）比 `umap_lookup`（哈希，连续桶 + 链表）在中等规模下可能更慢；但数据规模小时 `map` 可能更快（PDF 第 98-99 页）。
- `new_delete_loop` 明显慢于 `pool_alloc`：堆管理开销 vs 一次大块分配（PDF 第 97 页）。

## 注意事项

- `std::vector` 增长策略约 50%（PDF 第 98 页），`reserve` 可消除重分配。
- 容器选择要看**实际数据规模与应用场景**（PDF 第 98-99 页决策清单），本实验结论不能外推到所有规模。
