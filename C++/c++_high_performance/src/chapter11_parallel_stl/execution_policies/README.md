# execution_policies

C++17 并行 STL 执行策略：seq / par / par_unseq。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 333-337 页：

- 头文件 `<execution>`，命名空间 `std::execution`；
- `seq`：串行（小数据量阈值判断用）；
- `par`：并行；异常在主线程重新抛出（算法中途停止，位置未定义）；
- `par_unseq`：并行 + 允许 SIMD 向量化；谓词**不得抛异常、不得加锁**
  （同线程交错执行会死锁）；
- `std::reduce` 无序归约（要求交换+结合律）；`std::transform_reduce`
  组合 transform 与 reduce。

## 构建与运行

```bash
cmake --build build --target ch11_execution_policies_example \
    ch11_execution_policies_tests -j
./build/chapter11_parallel_stl/ch11_execution_policies_example
./build/chapter11_parallel_stl/ch11_execution_policies_tests
```

## 关键点

- GCC 13 的 `par`/`par_unseq` 后端需要链接 TBB（`-ltbb`）；
- tests 验证 seq/par/par_unseq 与 accumulate 结果一致；
- **异常传播差异**：书中（GCC 7）`par` 谓词抛异常会在调用线程重抛；
  GCC 13 libstdc++ 实测任何 `<execution>` transform 抛异常（含 seq）都直接
  `terminate`。example 用普通 `std::transform` 演示异常正常传播，
  并行谓词应避免抛异常。

## 注意

- `par_unseq` 的谓词约束严格：抛异常是 UB、加锁可能死锁；
- 数据量小时并行比串行慢，可用 `seq` 做阈值切换（书中 `find_largest`）。
