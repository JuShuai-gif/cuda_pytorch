# data_race

数据竞争演示：无保护 vs 互斥锁 vs 原子变量。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 284-285、293-294、300-301 页：

- `++counter` 是"读-改-写"多指令序列，两线程交错时丢失更新（如 42+1+1 只到 43）；
- 数据竞争是**未定义行为**，编译器不警告；
- **mutex**：`lock_guard` 保护临界区，保证同一时刻仅一线程进入；
- **atomic**：`++atomic` 等价 `fetch_add(1)`，增量本身不可分割。

## 构建与运行

```bash
cmake --build build --target ch10_data_race_example ch10_data_race_tests -j
./build/chapter10_concurrency/ch10_data_race_example
./build/chapter10_concurrency/ch10_data_race_tests
```

## 关键点

- mutex 和 atomic 版本恒等于 `n_times * 线程数`，tests 断言恒成立；
- 无保护版本是 UB，结果**可能碰巧正确**——tests 只报告是否偏离，不断言；
- TSan 构建可通过（`-DENABLE_THREAD_SANITIZER=ON`），但本机 Ubuntu 24.04
  运行时报 `unexpected memory mapping`（内核 ASLR 位数过高所致，需
  `sysctl vm.mmap_rnd_bits=28` 或换 clang）；此为本环境限制，非代码问题。

## 注意

- `atomic<int>` 的 `is_lock_free()` 通常为 true（≤ 机器字长）；
- 临界区越短越好：锁/原子会禁用编译器与硬件优化（争用代价）。
