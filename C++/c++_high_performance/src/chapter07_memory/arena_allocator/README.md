# arena_allocator

Arena 与自定义 STL 分配器。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 199-208 页：

- **Arena**：固定大小的连续缓冲 + 分配策略（bump allocator）。对齐到
  `max_align_t`，容量固定，放不下时回退 `::operator new`，只回收栈顶块
  （如栈语义），`reset()` 一次性回收全部；
- **ShortAlloc**：有状态的 STL 分配器，引用一个 Arena；容器
  （`std::set`/`std::vector`）用它从栈缓冲取内存，避免堆分配；
- 书中引用了 Howard Hinnant 的 `short_alloc`（栈上小型容器优化）。

## 构建与运行

```bash
cmake --build build --target ch07_arena_example ch07_arena_benchmark ch07_arena_tests
./build/chapter07_memory/ch07_arena_tests
./build/chapter07_memory/ch07_arena_example
./build/chapter07_memory/ch07_arena_benchmark
```

## 结果解释

example（libstdc++）：10 个 int 进 `set<int, ShortAlloc<int,512>>` →
arena 使用 480 字节、**堆分配 0 次**。

Benchmark（4 万元素插入 std::set，本环境 GCC 13.3 Release）：

| 分配器 | 相对 |
|---|---|
| 全局堆 `std::allocator` | 1.0x |
| 栈 arena `ShortAlloc` | **~3.7x 快** |

## 安全性说明（本实现已验证）

- 对齐：`align_up` 到 `alignof(max_align_t)`，tests 验证；
- 容量：超出回退 `::operator new`（不越界）；
- 生命周期：非拷贝/非移动（`= delete`），reset 后旧指针失效需自行管理；
- 异常：分配失败走 `operator new` 的 `bad_alloc` 路径；
- 非栈顶回收：no-op（如同书中的实现）。

## 结论（限定本环境）

- 自定义分配器适合"大量小对象、短生命周期、单线程"场景；
- 收益依赖数据规模与分配模式；
- 书中强调：通用分配器已很好，先分析模式再定制（PDF 200 页）；
- 现代补充：C++17 `std::pmr` 提供标准化的内存资源抽象（见 note/07）。

## 注意

- 大 arena 放栈上可能溢出（本实验 8MB arena 曾崩溃，改为 2MB）；
- arena 在栈上时其生命周期须长于使用它的容器。
