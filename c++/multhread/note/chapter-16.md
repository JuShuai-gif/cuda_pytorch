# 第16章：高性能计算（HPC）

> HPC 是 C++ 并发知识的自然延伸。OpenMP 提供声明式并行，CUDA 将并行扩展到数千核心的 GPU。本章建立从 CPU 多线程到 GPU 大规模并行的桥梁。

---

## 16.1 OpenMP 基础

### 原理

OpenMP（Open Multi-Processing）是一套**编译器指令 + 运行时库**的并行编程模型：

- 通过 `#pragma omp` 编译器指令声明并行区域
- 编译器自动处理线程创建、任务分配、同步
- 支持 Fortran、C、C++

```cpp
#include <omp.h>
#include <iostream>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::cout << "Hello from thread " << tid << "\n";
    }
    return 0;
}
```

### 核心指令

| 指令 | 功能 |
|------|------|
| `#pragma omp parallel` | 创建线程组 |
| `#pragma omp for` | 将 for 循环迭代分配给线程 |
| `#pragma omp parallel for` | 上两者的组合 |
| `#pragma omp reduction(+:var)` | 并行归约（安全合并局部结果） |
| `#pragma omp critical` | 临界区（互斥） |
| `#pragma omp barrier` | 线程屏障 |
| `#pragma omp single` | 仅一个线程执行 |
| `#pragma omp master` | 仅主线程执行 |

### 编译

```bash
# GCC
g++ -fopenmp program.cpp -o program

# Clang (需要安装 libomp)
clang++ -fopenmp program.cpp -o program

# CMake
find_package(OpenMP REQUIRED)
target_link_libraries(my_target PRIVATE OpenMP::OpenMP_CXX)
```

---

## 16.2 schedule 调度策略

OpenMP 的 `schedule` 子句控制循环迭代如何分配给线程：

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `static` | 编译时平均分配 | 每迭代耗时均匀 |
| `dynamic` | 运行时动态分配（chunk 大小可调） | 迭代耗时不均 |
| `guided` | 逐渐减小的动态分配 | 未知负载分布 |
| `auto` | 编译器/运行时决定 | 通用 |
| `runtime` | 通过 OMP_SCHEDULE 环境变量指定 | 灵活调参 |

```cpp
// 静态分配，每线程 100 个迭代
#pragma omp parallel for schedule(static, 100)
for (int i = 0; i < 10000; ++i) { ... }

// 动态分配，每次获取 1 个迭代
#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < 10000; ++i) { ... }
```

---

## 16.3 reduction 归约

```cpp
int sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; ++i) {
    sum += data[i];
}
// sum 自动包含所有线程的局部部分和
```

支持的归约操作符：`+`, `-`, `*`, `&&`, `||`, `&`, `|`, `^`, `min`, `max`

### 生活类比

老师把一叠考卷分成 4 份，4 个助教各自批改并算出自己那叠的平均分（局部归约），最后老师把所有平均分再平均——这就是 OpenMP reduction 的工作方式。

---

## 16.4 OpenMP vs C++ Threads vs TBB

| 特性 | OpenMP | C++ Threads | TBB |
|------|--------|-------------|-----|
| 抽象层级 | 高（编译器指令） | 低（系统线程） | 中（任务级） |
| 学习曲线 | 低 | 中 | 中 |
| 灵活性 | 低（循环并行为主） | 高（任意并行模式） | 高 |
| 性能 | 好 | 取决于实现 | 很好 |
| CPU 绑定 | 隐式 | 显式 | 隐式 |
| 跨平台 | 一般（需编译器支持） | 很好（标准库） | 很好 |

---

## 16.5 CUDA 基础理论

### GPU 架构

```
CPU (Host)                     GPU (Device)
┌──────────┐                  ┌──────────────────┐
│ 少量强核  │  ← PCIe/NVLink → │ 数千弱核 (SM/CUDA) │
│ 大缓存   │                  │ 小缓存            │
│ 分支预测 │                  │ 无分支预测        │
│ 低延迟   │                  │ 高吞吐量          │
└──────────┘                  └──────────────────┘
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Kernel** | 在 GPU 上执行的函数，`__global__` 标记 |
| **Thread** | GPU 的最小执行单元 |
| **Block** | 线程组（共享 shared memory），最多 1024 线程 |
| **Grid** | Block 的集合（一次 kernel 启动的全部 block） |
| **Warp** | 32 个线程的调度单位（NVIDIA） |
| **Shared Memory** | Block 内共享的片上高速内存（~100x 快于 global memory） |
| **Global Memory** | GPU 的主内存（HBM），大容量高带宽 |

### CUDA Stream

**Stream** 是一系列按顺序在 GPU 上执行的操作（kernel 启动、内存拷贝）。多个 stream 可**并发执行**，实现：

- 计算与数据传输重叠
- 多个 kernel 并发

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel<<<grid, block, 0, stream1>>>(...);  // stream1 上执行
cudaMemcpyAsync(..., stream2);              // stream2 上传输
```

### Unified Memory

```cpp
// 自动在 CPU/GPU 间迁移数据
cudaMallocManaged(&ptr, size);
// CPU 和 GPU 都可以直接访问 ptr
```

优势：简化编程（无需显式 cudaMemcpy）
代价：可能引入隐式的页面错误和迁移开销

### Pinned Memory

```cpp
// 页锁定内存：加速 CPU↔GPU 传输
cudaMallocHost(&ptr, size);
// 传输带宽可比 pageable memory 高 2-3x
```

---

## 16.6 CPU/GPU 异构并行

### 协同模式

```
阶段 1 (CPU):  数据准备、预处理
阶段 2 (GPU):  大规模数值计算
阶段 3 (CPU):  结果汇总、后处理
阶段 4 (GPU):  下一批计算 (同时 CPU 处理上一批结果)
```

关键技术：
- **流水线**：CPU 和 GPU 同时工作在不同批次上
- **异步传输**：cudaMemcpyAsync + stream
- **双缓冲**：GPU 计算一块数据时，CPU 准备下一块

---

## 16.7 知识体系交叉引用

| 本章主题 | 相关章节 |
|----------|----------|
| OpenMP parallel for | 第10章 并行算法、第14章 数据并行 |
| OpenMP reduction | 第14章 并行归约 |
| CUDA stream | 第14章 Pipeline 模式 |
| 异构并行 | 第8章 并发代码设计 |

---

## 16.8 本章小结

1. **OpenMP** 是 CPU 并行的快速路径——几行 pragma 就能利用所有核心
2. **CUDA** 将并行规模从数十核扩展到数千核
3. **Stream** 和 **Unified Memory** 是 GPU 编程的高级抽象
4. CPU+GPU **异构并行**是 HPC 的主流范式
5. 选择合适的工具：数据并行循环 → OpenMP，不规则任务 → C++ Threads/TBB，大规模数值计算 → CUDA
