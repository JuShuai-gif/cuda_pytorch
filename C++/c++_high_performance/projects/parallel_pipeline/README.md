# parallel_pipeline

综合实践：并行 map → filter → reduce 流水线。

> 综合 Chapter 3（测量）+ Chapter 11（执行策略）的教学实现。

## 功能

- 一条流水线三个并行阶段：
  1. `transform(par, ...)`：逐元素映射（每元素独立）；
  2. `copy_if(par, ...)` + `erase`：按条件过滤并压实；
  3. `reduce(par, ...)`：无序归约（要求可交换 + 可结合）；
- 每一阶段 = 一个执行策略调用，代码短、可读性高（Ch11 主旨）。

## 构建与运行

```bash
cmake -S projects -B build-projects
cmake --build build-projects --target parallel_pipeline_example \
    parallel_pipeline_tests parallel_pipeline_benchmark -j

./build-projects/parallel_pipeline/parallel_pipeline_example
./build-projects/parallel_pipeline/parallel_pipeline_tests
./build-projects/parallel_pipeline/parallel_pipeline_benchmark
```

## 关键验证

- tests：map+filter+reduce 各组合（全保留/全过滤/乘法归约）与串行结果一致；
- benchmark（本机 24 线程）：并行 vs 串行约 1.7x（map 为轻量循环，受内存
  带宽限制；计算密集映射时加速更明显）；
- example：平方后取奇数和，与串行核对。

## 注意

- 阶段间有 `vector` 拷贝与压实，若数据量巨大需考虑内存（Ch4/7）；
- 归约算子必须满足交换律 + 结合律，否则结果不定（Ch11 `std::reduce`）；
- `copy_if(par)` 内部对共享写位置有优化，但谓词仍不得抛异常/加锁。
