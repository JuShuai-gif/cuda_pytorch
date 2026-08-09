# par_transform

手写并行 `std::transform()`：朴素分块 vs 分治。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 319-326 页：

- **朴素版** `par_transform_naive`：按 `hardware_concurrency()` 均分，
  每块一个 `std::async`；
- **分治版** `par_transform`：递归对半拆到 chunk 阈值，一半交 async、
  一半本线程递归处理；
- 变成本函数（成本随值增长）下，朴素版受最慢块限制，分治版用大量小
  任务让调度器动态平衡。

## 构建与运行

```bash
cmake --build build --target ch11_par_transform_example \
    ch11_par_transform_benchmark ch11_par_transform_tests -j

./build/chapter11_parallel_stl/ch11_par_transform_example
./build/chapter11_parallel_stl/ch11_par_transform_tests
./build/chapter11_parallel_stl/ch11_par_transform_benchmark
```

## 关键点

- `hardware_concurrency()` 可能返回 0，需 clamp 到 1；
- chunk 太小任务开销反噬（书中 chunk=10 时仅 0.55x）；
- tests 验证并行结果与串行逐位一致；
- benchmark（本机 GCC 13.3 / i7-13700K，24 线程，成本随值增长的变换）：
  naive ≈ 10.6x，分治 chunk=10000 ≈ 17.2x（书中 8 核时 3.8-5.9x）。

## 注意

- 手写并行复杂度高、可维护性差，生产应优先用 C++17 执行策略
  （见 execution_policies 实验）。
