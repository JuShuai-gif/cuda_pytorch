# stack_vs_heap

栈与堆的差异。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 177-181 页：

- **栈**：连续内存块、固定上限（默认约 8MB）、每线程独立、永不碎片化、
  分配只移动栈指针（极快）；栈在 x86-64 上向低地址增长；
- **堆**：全局共享、可任意分配/释放、易碎片化、需处理并发。

栈满（深递归或大自动变量）→ 程序崩溃（stack overflow）。

## 构建与运行

```bash
cmake --build build --target ch07_stack_vs_heap_example
./build/chapter07_memory/ch07_stack_vs_heap_example
ulimit -s   # 查看默认栈大小（KB）
```

## 关键点

- 打印栈上局部变量地址观察栈增长方向；
- 堆分配地址通常递增；
- 栈大小可通过 `ulimit -s` 查看/修改。

> 注意：`-O2` 下编译器可能重排/合并栈槽，地址方向仅供观察。
