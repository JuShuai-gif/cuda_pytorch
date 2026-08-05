# sequence_containers

序列容器的遍历与插入。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 108-111 页：

| 容器 | 内存布局 | 遍历 | 插入 |
|---|---|---|---|
| `std::vector` | 连续 | 快（空间局部性） | 尾 O(1) 均摊，中间 O(n) |
| `std::array` | 连续（栈） | 快 | 固定大小 |
| `std::deque` | 分块 | 中 | 头尾 O(1) |
| `std::list` | 双链节点 | 慢（每节点缓存缺失） | 有迭代器时 O(1) |
| `std::forward_list` | 单链节点 | 慢 | 更省内存（单指针） |

## 构建与运行

```bash
cmake --build build --target ch04_sequence_benchmark \
      ch04_sequence_insert_benchmark ch04_sequence_tests
./build/chapter04_data_structures/ch04_sequence_tests
./build/chapter04_data_structures/ch04_sequence_benchmark
./build/chapter04_data_structures/ch04_sequence_insert_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，200 万元素遍历）

| 容器 | mean | 相对 |
|---|---|---|
| `std::vector` | ~182 µs | 1.0x |
| `std::deque` | ~486 µs | 2.7x |
| `std::forward_list` | ~3.6 ms | 20x |
| `std::list` | ~3.7 ms | 21x |

链表慢约 20 倍：每个节点单独堆分配，遍历时逐节点缓存缺失。

插入 benchmark：单次中间插入 vector+deque+list 合计 ~60ms（vector 的 O(n)
搬移主导），尾部 push_back 三者合计 ~3.8ms。

## 结论（限定本环境）

- 默认选择 `std::vector`：连续内存遍历最快；
- 需要头尾插入用 `deque`；需要稳定迭代器引用且插入频繁用 `list`，
  但遍历会付出缓存代价；
- 选择前先问：元素规模、访问模式、是否需要排序（PDF 108 页）。

## 现代补充

> 现代补充：C++17 前 `std::string` 不保证连续内存；C++17 起保证连续
> （PDF 112 页），可安全传给 C API。
