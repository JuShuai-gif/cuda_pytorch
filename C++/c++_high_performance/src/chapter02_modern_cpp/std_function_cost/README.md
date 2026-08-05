# std_function_cost

`std::function` 的堆分配与内联观察。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 59 页列出 `std::function` 三个性能问题，本实验验证前两个：

1. **不能内联**：调用是间接调用（`call *%rax`）；
2. **堆分配捕获变量**："some implementations of std::function do not
   heap-allocate if the size of the captured variable is less than a specific
   threshold"——即 Small Buffer Optimization（SBO）。

本实验通过覆盖全局 `operator new` 计数，观察构造 `std::function` 时是否分配堆内存。

## 构建与运行

```bash
cmake --build build --target ch02_function_cost_example ch02_function_cost_benchmark
./build/chapter02_modern_cpp/ch02_function_cost_example
./build/chapter02_modern_cpp/ch02_function_cost_benchmark
```

## 结果解释（libstdc++，GCC 13）

`sizeof(std::function<void()>)` = 32 字节，SBO 缓冲 16 字节。

| 捕获 | 大小 | 分配 |
|---|---|---|
| 无捕获 | 1 字节 | `new: +0`（无分配） |
| 小捕获（int） | 4 字节 | `new: +0`（SBO 内，无分配） |
| 大捕获（64 字节） | 64 字节 | `new: +1`（超出 SBO，堆分配） |

汇编观察：`std::function` 调用生成 `call *%rax`（间接调用），
而直接 lambda 调用是直接 `call` 或完全内联。

## 结论

- SBO 依赖实现（libstdc++ 16 字节；其他库可能不同）；
- 大量捕获对象的 `std::function` 有堆分配成本；
- 小函数 + 高频调用 + 捕获大对象是 `std::function` 最不利的组合。

> 原理补充：`operator new` 覆盖使用 `std::malloc`（保证 `alignof(max_align_t)`
> 对齐），计数只在测量窗口开启，不影响程序其余部分。
