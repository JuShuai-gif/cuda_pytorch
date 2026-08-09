# high_performance_container

综合实践：开放寻址哈希集合。

> 综合 Chapter 4（数据结构、cache locality）+ Chapter 5（迭代器）+ Chapter 7
> （内存分配）的教学实现。

## 功能

- **开放寻址 + 线性探测**：元素存于扁平 `vector`，无链表节点、无逐节点堆分配；
- **负载因子上限 0.7**，超限自动翻倍扩容重哈希（所有旧元素重插）；
- 重复插入返回 `false`，`contains` O(1) 均摊；
- `collect()` 线性收集全部已占用槽（教学性遍历）。

## 构建与运行

```bash
cmake -S projects -B build-projects
cmake --build build-projects --target hash_set_example hash_set_tests \
    hash_set_benchmark -j

./build-projects/high_performance_container/hash_set_example
./build-projects/high_performance_container/hash_set_tests
./build-projects/high_performance_container/hash_set_benchmark
```

## 关键验证

- tests：去重/contains/多次扩容后元素全部可查/大批量（5 万元素）；
- benchmark（本机 GCC 13.3）：查询与 `std::unordered_set`（链式）相当
  （ratio 0.9-1.1x）——本实现 `vector<string>` 存储字符串本身有拷贝，
  与链式集差异不明显；生产可用 `string_view`/`std::pmr` 进一步优化。

## 注意

- 开放寻址的删除需 tombstones 或再哈希（此处未实现，仅插入/查询）；
- `std::vector<bool>` 节省位但读写慢，生产可换 `std::vector<unsigned char>`
  或位集；
- 迭代顺序无意义（哈希槽位序）；需有序遍历请用 `std::set`。
