# false_sharing

伪共享（false sharing）演示与消除：缓存行对齐。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 315 页：

- 两线程各自累加"自己的"计数器，看似无共享；
- 若两计数器在同一**缓存行**，任一写入都使另一核的缓存行失效 → 互相拖累；
- 用 `alignas(std::hardware_destructive_interference_size)` 把每个计数器
  对齐到独立缓存行，消除伪共享。

## 构建与运行

```bash
cmake --build build --target ch10_false_sharing_example \
    ch10_false_sharing_tests ch10_false_sharing_benchmark -j

./build/chapter10_concurrency/ch10_false_sharing_example
./build/chapter10_concurrency/ch10_false_sharing_tests
./build/chapter10_concurrency/ch10_false_sharing_benchmark
```

## 关键点

- `std::hardware_destructive_interference_size`（C++17，`<new>`）给出本机
  缓存行大小；
- benchmark 用两线程并发自增对比 padded/unpadded：本机（GCC 13.3 /
  i7-13700K）实测 unpadded/padded 约 **9.4x**（循环内需 `compiler_barrier`
  强制每次增量写内存，否则编译器把递增提升到寄存器、伪共享不产生）；
- 两种实现的最终结果必须一致（tests 断言）。

## 注意

- padding 用空间换速度：只有**多核并发写相邻数据**时才值得；
- 单线程或只读共享无需 padding；
- 效果依赖 CPU 架构，不同机器差异明显。
