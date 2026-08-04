# Chapter 16 — 高性能计算 HPC

从 CPU 多线程到 GPU 大规模并行的知识桥梁。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_openmp_hello.cpp` | OpenMP 入门 | parallel/for/critical/atomic/sections |
| `02_openmp_parallel_reduce.cpp` | OpenMP 归约 | reduction、嵌套并行、nowait |
| `03_openmp_schedule.cpp` | 调度策略 | static/dynamic/guided 对比、负载均衡 |
| `04_gpu_concepts.cpp` | GPU 概念 | SIMT 模型、内存层次、Stream、异构计算 |

## 编译运行

```bash
# 确保安装了 OpenMP
sudo apt install libomp-dev  # Ubuntu/Debian

mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)

# OpenMP 相关
./ch16_01_openmp_hello
./ch16_02_openmp_parallel_reduce
./ch16_03_openmp_schedule

# GPU 概念演示 (纯 CPU)
./ch16_04_gpu_concepts

# 手动设置 OpenMP 线程数
OMP_NUM_THREADS=4 ./ch16_01_openmp_hello
```

## 学习建议

1. OpenMP 是数据并行的快速入口，5 分钟即可上手
2. schedule 策略在不均匀负载下影响巨大
3. GPU 概念先用 CPU 代码理解，再实践 CUDA
4. 实际 CUDA 开发需要 NVIDIA GPU + CUDA Toolkit
5. 异构计算 (CPU+GPU) 是现代 HPC 的标准范式
