# Chapter 14 — 并行算法进阶

手动实现工业级并行算法，超越 C++17 并行 STL。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_pipeline.cpp` | Pipeline 模式 | 有界队列、阶段解耦、瓶颈分析 |
| `02_parallel_reduce.cpp` | 并行归约 | chunk-based、浮点结合律陷阱 |
| `03_parallel_scan.cpp` | 并行前缀和 | 两阶段 scan、最大值前缀 |
| `04_batch_processing.cpp` | 批量处理 | 批量入队、batch atomic、缓存友好 |
| `05_work_stealing_deep.cpp` | 工作窃取 | WS deque、负载均衡、vs 共享队列 |
| `06_data_task_parallelism.cpp` | 数据vs任务并行 | 对比 demo、混合模式、barrier 同步 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
./ch14_01_pipeline
./ch14_02_parallel_reduce
./ch14_03_parallel_scan
./ch14_04_batch_processing
./ch14_05_work_stealing_deep
./ch14_06_data_task_parallelism
```

## 学习建议

1. 先从数据并行和任务并行的区别入手
2. Pipeline 是生产者-消费者的进阶版，理解阶段间解耦
3. 并行 reduce/scan 是 HPC 的基础组件
4. 批量处理是最简单高效的优化手段
5. Work Stealing 是所有现代调度器的核心
