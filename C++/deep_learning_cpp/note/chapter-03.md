# 第 3 章：CUDA GPU 加速深度学习

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 3 章，第 77–105 页。本章位于 Part 2（"Building and Training Neural Networks in C++"）之前，是全书从 CPU 过渡到 GPU 加速的桥梁章节。

---

## 目录

1. [章节概述](#章节概述)
2. [核心概念：异构并行计算](#核心概念异构并行计算)
3. [文件索引](#文件索引)
4. [代码演进：从 CPU 到多 Block GPU](#代码演进从-cpu-到多-block-gpu)
5. [编译与运行](#编译与运行)
6. [技术速查](#技术速查)
7. [PDF 完整内容对照](#pdf-完整内容对照)
8. [注意事项](#注意事项)

---

## 章节概述

CUDA（Compute Unified Device Architecture）是 NVIDIA 的并行计算平台和编程模型，通过暴露数千个轻量级线程，在数值模拟、深度学习和图像处理等场景中实现远超 CPU 的性能。本章循序渐进地将一个简单的 CPU 数组加法逐步改造为充分利用 GPU 并行能力的 CUDA 实现。

### 预处理六大价值（迁移自第 2 章框架）

| 价值               | 在本章的体现                                                  |
| ------------------ | ------------------------------------------------------------- |
| 性能飞跃           | 多 Block 网格启动将数组加法加速数十倍                         |
| 并行思维转变       | 从单线程 `for` 循环 → 每个线程只算自己那一份数据              |
| 内存管理意识       | 统一内存 vs 显式拷贝的选择影响性能和开发效率                  |
| 调试能力           | 三步错误检查法覆盖 API 调用、启动参数、异步执行               |
| 硬件理解           | Thread → Warp → Block → Grid 层次结构是写出高效 kernel 的基础 |
| 工具链掌握         | `nvcc` 编译、`nvprof` 剖析、Nsight 深度分析                   |

### CPU vs GPU 对比（跑车 vs 自行车队）

| 特性       | CPU（延迟最优）              | GPU（吞吐量最优）                  |
| ---------- | ---------------------------- | ---------------------------------- |
| 设计目标   | 少量复杂线程，大缓存，低延迟 | 大量简单线程，高吞吐量             |
| 线程数量   | 几个到几十个                 | 数千到数百万                       |
| 线程切换   | OS 调度，开销大              | 硬件管理，开销极小                 |
| 内存模型   | 大缓存、统一地址空间         | 独立全局 + 共享 + 寄存器三级       |
| 适合任务   | 串行逻辑、分支密集、IO 密集  | 矩阵乘法、卷积、图像滤波、蒙特卡洛 |
| 赢的场景   | 分支重、同步频繁、kernel 启动开销 > 收益 | 规则的大规模数据并行，coalesced 内存访问 |
| 类比       | 跑车（单乘客快速到达）       | 自行车队（大量包裹并发配送）       |

---

## 核心概念：异构并行计算

### 什么是异构并行计算

异构并行计算 = CPU（主机）编排控制逻辑、IO、批处理 + GPU（设备）执行数据并行 kernel。好的设计需要：

- **最小化主机-设备间 PCIe/NVLink 数据传输**：保持热数据驻留在 GPU 上
- **使用固定内存（pinned memory）**：加快传输速度
- **用 CUDA stream 和 event 重叠传输与计算**：隐藏延迟
- **统一内存** 可加速开发，追求极致性能时改用**显式拷贝 + prefetch + advice**
- **融合 kernel**：减少启动开销和全局内存访问
- **用 Nsight Systems/Compute 剖析**：平衡利用、发现带宽瓶颈、决定哪些工作留在 CPU

### CPU 基线：数组加法（C++ 纯 CPU 版本）

```cpp
#include <iostream>
#include <algorithm>
#include <cmath>

// 在 CPU 上逐元素累加两个数组
void vectorAddCPU(int length, float* inputA, float* inputB) {
    for (int idx = 0; idx < length; ++idx) {
        inputB[idx] += inputA[idx];
    }
}

int main() {
    const int SIZE = 1 << 20;  // 1,048,576 元素
    float* vecA = new float[SIZE];
    float* vecB = new float[SIZE];

    // 填充样本值
    for (int i = 0; i < SIZE; ++i) {
        vecA[i] = 0.5f;
        vecB[i] = 2.5f;
    }

    // 执行向量加法
    vectorAddCPU(SIZE, vecA, vecB);

    // 验证正确性
    float largestDeviation = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        float expected = 3.0f;
        largestDeviation = std::max(largestDeviation, std::fabs(vecB[i] - expected));
    }
    std::cout << "Max error: " << largestDeviation << std::endl;

    delete[] vecA;
    delete[] vecB;
    return 0;
}
```

```bash
# 编译运行
$ g++ vector_add_cpu.cpp -o vector_add_cpu
$ ./vector_add_cpu
Max error: 0.000000
```

> **指针说明（PDF 特别强调）：** `float *x` 表示 `x` 是一个指向 `float` 的指针——它存储的是内存地址而非值本身。在 CPU 上用 `new float[N]` 或在 CUDA 中用 `cudaMallocManaged(&x, N*sizeof(float))` 分配后，`x` 指向分配的内存，通过 `x[0]`（等价于 `*(x+0)`）访问。

---

## 文件索引

> **已实现。** 以下 8 个 .cu/.cpp 文件完整覆盖 PDF 第 3 章的全部知识点。

### 一、环境搭建与验证 — PDF 第 78–87 页

| 文件                  | PDF 页 | 涵盖知识点                                            | 依赖         |
| --------------------- | ------ | ----------------------------------------------------- | ------------ |
| `00_cuda_hello.cu`    | 78–83  | `nvidia-smi` 验证、`nvcc --version` 检查、设备查询示例 | CUDA Toolkit |
| `00_colab_setup.cu`   | 83–87  | Google Colab 中启用 T4 GPU、安装 `nvcc4jupyter` 扩展   | Colab 环境   |

### 二、CUDA 编程模型与向量加法（代码演进四步走）— PDF 第 87–100 页

| 文件                        | PDF 页  | 涵盖知识点                                                                                     | 依赖         | 关键性能变化              |
| --------------------------- | ------- | ---------------------------------------------------------------------------------------------- | ------------ | ------------------------- |
| `01_vector_add_cpu.cpp`     | 87–89   | CPU 基线：`for` 循环逐元素加法、`new`/`delete` 手动内存管理、结果验证                          | STL          | 基线（参照物）            |
| `02_vector_add_cuda.cu`     | 89–91   | `__global__` 核函数、`cudaMallocManaged` 统一内存、`<<<1, 1>>>` 单线程启动、`cudaDeviceSynchronize` | CUDA Toolkit | 单线程 GPU（无加速）      |
| `03_vector_add_parallel.cu` | 93–96   | `<<<1, 256>>>` 多线程、grid-stride 循环、`threadIdx.x`/`blockDim.x` 内置变量、避免 data race    | CUDA Toolkit | 1 个 Block × 256 线程     |
| `04_vector_add_grid.cu`     | 96–100  | `<<<numBlocks, 256>>>` 多 Block、全局线程索引公式 `blockIdx.x*blockDim.x + threadIdx.x`、`nvprof` 分析 | CUDA Toolkit | N 个 Block × 256 线程     |

### 三、错误处理 — PDF 第 100–103 页

| 文件                          | PDF 页    | 涵盖知识点                                                                                             | 依赖         |
| ----------------------------- | --------- | ------------------------------------------------------------------------------------------------------ | ------------ |
| `05_cuda_error_handling.cu`   | 100–103   | 三步错误检查：① API 返回值（`cudaError_t`）→ ② `cudaGetLastError()` 启动错误 → ③ `cudaDeviceSynchronize` 异步错误、`checkCuda` 宏封装 | CUDA Toolkit |

### 四、多维度启动（dim3）— PDF 第 102–103 页

| 文件                | PDF 页    | 涵盖知识点                                                             | 依赖         |
| ------------------- | --------- | ---------------------------------------------------------------------- | ------------ |
| `06_cuda_dim3.cu`   | 102–103   | `dim3` 2D/3D 网格与线程块、最大维度限制表（表 3.1）、图像张量索引场景   | CUDA Toolkit |

### 五、性能剖析与优化（进阶，PDF 第 91–100 页提及）

| 文件                          | PDF 页    | 涵盖知识点                                                             | 依赖            |
| ----------------------------- | --------- | ---------------------------------------------------------------------- | --------------- |
| `07_cuda_profiling.cu`        | 91–92     | `nvprof` 基础剖析、理解 "24 CPU page faults"、Nsight Systems/Compute   | CUDA + Nsight   |
| `08_cuda_optimization.cu`     | 91–100    | coalesced 全局内存访问、共享内存 tiling、CUDA stream 传输-计算重叠     | CUDA Toolkit    |

---

## 代码演进：从 CPU 到多 Block GPU

本章的核心教学线是**同一个数组加法问题的四次迭代**，每一版都在前版基础上引入新机制：

### 第 1 步：CPU 基线

```cpp
// 纯 CPU 实现：单线程顺序遍历所有元素
void vectorAddCPU(int length, float* inputA, float* inputB) {
    for (int idx = 0; idx < length; ++idx)
        inputB[idx] += inputA[idx];
}
```
- **关键点：** 此版本作为性能参照基准（yardstick），记录 CPU 版本的运行时间供后续对比。

### 第 2 步：CUDA 单线程核函数

```cpp
// 第一个 CUDA kernel：仍只有一个线程在工作
__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x;       // 这个线程在 block 内的编号（=0）
    int stride = blockDim.x;       // block 中的线程总数（=1）
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];        // 等价于 CPU 的 for 循环！
}

int main(void) {
    int N = 1 << 20;               // 1,048,576 元素
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));  // CPU + GPU 均可访问
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);        // <<<1 个 block, 1 个线程>>>
    cudaDeviceSynchronize();       // 主机等待 GPU 完成

    // 验证结果：所有 y[i] 应该等于 3.0f
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}
```
- **关键概念：**
  - `__global__` 标记一个函数为 GPU 核函数（kernel），由主机调用，在设备执行，返回值必须为 `void`
  - `<<<gridDim, blockDim>>>` 是 CUDA 独有的启动配置语法（三重尖括号）
  - `cudaMallocManaged` 分配统一内存——CPU 和 GPU 使用同一指针访问，编译器和运行时自动处理底层数据迁移
  - `cudaDeviceSynchronize` 阻塞主机，等待 GPU 完成所有排队工作（kernel 启动是异步的）
- **编译：** `nvcc -arch=sm_70 -o add_cuda add.cu -run`
- **剖析：** `nvprof ./add_cuda` —— 大部分时间花在 `cudaMallocManaged` 和同步上

### 第 3 步：多线程并行

```cpp
// 用 <<<1, 256>>> 启动：1 个 block，256 个线程
// 每个线程用 grid-stride 循环覆盖 N 个元素

int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;  // 向上取整
add<<<numBlocks, blockSize>>>(N, x, y);

// Kernel 内部：
__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x;       // 线程在 block 内的位置 [0, blockDim.x)
    int stride = blockDim.x;       // 每 block 线程总数
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];        // 每个线程跨步处理自己的那份数据
}
```
- **关键概念：**
  - **Data race 问题：** 如果多个线程同时读写同一个 `y[i]`，结果不确定（取决于谁先执行）。grid-stride 循环通过给每个线程分配唯一的起始 offset (`threadIdx.x`) 和跨步 (`blockDim.x`) 避免了竞争
  - `threadIdx.x`：线程在其 block 内的索引（0 到 blockDim.x-1）
  - `blockDim.x`：每个 block 的线程数（模板 `<<<grid, block>>>` 的第二个参数）
  - **为什么 Block 大小选 256？** NVIDIA 硬件以 32 线程为 warp 调度，block 大小应为 32 的倍数以最大化占有率
- **性能：** `nvprof` 显示大幅加速，但利用率仍不高——只有 1 个 block

### 第 4 步：多 Block 扩展

```cpp
// 全局线程索引公式——每个线程得到唯一的数据区间
__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // 全局唯一线程 ID
    int stride = blockDim.x * gridDim.x;                // 所有线程的总数
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

// 启动配置
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```
- **关键概念：**
  - `blockIdx.x`：当前 block 在 grid 中的索引
  - `gridDim.x`：grid 中 block 的总数
  - `blockIdx.x * blockDim.x + threadIdx.x`：**全局线程索引**是 CUDA 编程中最基本的模式
  - 例如 `gridDim.x=4096, blockDim.x=256`，`blockIdx.x=2, threadIdx.x=3` → 全局索引 = 2×256+3 = 515
  - 用 `numBlocks = (N + blockSize - 1) / blockSize`（向上取整除法）确保覆盖所有元素
- **性能：** `nvprof` 汇总显示 GPU 时间进一步缩短。图中报告了 **24 CPU page faults**——统一内存在首次访问时触发的页错误——这是统一内存的典型代价

---

## 编译与运行

### 环境要求

```bash
# 硬件
NVIDIA GPU（计算能力 ≥ 7.0，如 T4 / V100 / A10 / A100 / H100）
VRAM ≥ 8 GB  # 推荐用于后续训练章节

# 软件
NVIDIA 驱动（与 CUDA Toolkit 版本匹配）
CUDA Toolkit 12.x       → /usr/local/cuda
nvcc（CUDA 编译器）     → /usr/local/cuda/bin/nvcc
C++17 编译器（GCC 11+ / Clang 14+ / MSVC 2022）
CMake 3.22+
```

### 快速验证

```bash
# 1. 检查驱动与 GPU
nvidia-smi
# 应显示 GPU 型号、驱动版本、CUDA 版本

# 2. 检查 CUDA 编译器
nvcc --version
# 应显示 CUDA compilation tools, release X.X

# 3. 编译并运行设备查询
nvcc -o deviceQuery /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery.cpp
./deviceQuery
# 应列出所有 GPU 及其计算能力
```

### 编译命令

```bash
# 单文件 CUDA 编译
nvcc -arch=sm_75 -o output file.cu        # T4 (Turing, CC 7.5)
nvcc -arch=sm_80 -o output file.cu        # A100 (Ampere, CC 8.0)
nvcc -arch=sm_90 -o output file.cu        # H100 (Hopper, CC 9.0)

# 编译并立即运行
nvcc -arch=sm_75 -o output file.cu -run

# 剖析运行
nvprof ./output                            # 基础剖析
nvprof --print-gpu-trace ./output          # 完整时间线

# 查询 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 运行示例（CMake 构建后）

```bash
# 环境搭建 & 验证
./build/chapter03/cuda_hello

# 向量加法：CPU 基线
./build/chapter03/vector_add_cpu

# 向量加法：CUDA 逐步加速
./build/chapter03/vector_add_cuda        # 单线程 GPU
./build/chapter03/vector_add_parallel    # 1 Block × 256 线程
./build/chapter03/vector_add_grid        # N Block × 256 线程

# 剖析
nvprof ./build/chapter03/vector_add_grid

# 错误处理
./build/chapter03/cuda_error_handling

# 多维度
./build/chapter03/cuda_dim3
```

### Google Colab 快速入口

```bash
# 1. 菜单：Runtime → Change runtime type → 选择 T4 GPU → Save
# 2. 验证 GPU 可用
!nvidia-smi

# 3. 安装 nvcc4jupyter 扩展
!pip install nvcc4jupyter
%load_ext nvcc4jupyter

# 4. 写 CUDA 代码（方式一：写入 .cu 文件）
%%writefile kernel.cu
#include <iostream>
__global__ void hello() {
    printf("Hello from GPU! block %d, thread %d\n", blockIdx.x, threadIdx.x);
}
int main() {
    hello<<<3, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
!nvcc kernel.cu -o kernel
!./kernel

# 5. 写 CUDA 代码（方式二：直接在 cell 中写 CUDA）
%%cuda
// CUDA code here...
```

---

## 技术速查

### CUDA 核函数编写

| 组件              | 语法 / 模式                                                        | 说明                                            |
| ----------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| 核函数声明        | `__global__ void func(args)`                                       | 在 GPU 上执行，由主机（或 CC≥3.5 时由设备）调用 |
| 启动配置          | `func<<<gridDim, blockDim, [shMem], [stream]>>>(args)`            | gridDim 和 blockDim 可以是 int 或 dim3          |
| 线程索引（1D）    | `int i = blockIdx.x * blockDim.x + threadIdx.x`                    | 全局唯一线程 ID，CUDA 编程最基本的模式          |
| Grid-stride 循环  | `for (int i = index; i < n; i += blockDim.x * gridDim.x)`          | 任意线程数/块数都能覆盖全部 N 个元素            |
| 设备同步          | `cudaDeviceSynchronize()`                                          | 阻塞主机直到 GPU 完成 default stream 中的工作    |
| 内存分配（统一）  | `cudaMallocManaged(&ptr, size)`                                    | CPU 和 GPU 共享指针，运行时管理数据迁移         |
| 内存释放          | `cudaFree(ptr)`                                                    | 释放统一内存或设备内存                          |

### 线程层次结构

| 层级     | 说明                                                      | 限制                            |
| -------- | --------------------------------------------------------- | ------------------------------- |
| Thread   | 最小执行单元，拥有独立的局部寄存器和局部内存              | --                              |
| Warp     | 32 个线程为一组，以 SIMT（单指令多线程）锁步方式执行      | 同一 warp 内分支发散会降低效率  |
| Block    | 1-1024 个线程为一组，可通过 `__shared__` 共享内存和 `__syncthreads()` 同步 | 总线程数 ≤ 1024                 |
| Grid     | 一个或多个 Block 的集合                                   | gridDim.x ≤ 2³¹-1               |
| SM       | Streaming Multiprocessor——GPU 上执行 block 的物理单元     | 例如 A100 有 108 个 SM          |

### dim3 最大维度限制（表 3.1）

| 维度     | Grid 最大值    | Block 最大值 |
| -------- | -------------- | ------------ |
| x        | 2³¹ - 1        | 1024         |
| y        | 65535          | 1024         |
| z        | 65535          | 64           |

```cpp
// 2D 启动示例：适合图像处理（高 × 宽）
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(
    (width  + 15) / 16,
    (height + 15) / 16,
    1
);
someKernel<<<number_of_blocks, threads_per_block>>>();
```

### 内存类型与作用域

| 类型       | 作用域    | 生命周期 | 速度              | 典型用途                         |
| ---------- | --------- | -------- | ----------------- | -------------------------------- |
| 寄存器     | 每线程    | 线程     | 最快（1 周期）    | 局部变量、临时计算结果           |
| 局部内存   | 每线程    | 线程     | 慢（寄存器溢出时） | 大数组或编译器无法放入寄存器的变量 |
| 共享内存   | 每 Block  | Block    | 快（~1-32 周期）  | Block 内线程协作、数据 tiling    |
| 全局内存   | 全设备    | 程序     | 慢（~400-600 周期） | 输入/输出大数组、设备主存        |
| 统一内存   | 主机+设备 | 程序     | 按需迁移          | 简化开发，首次访问触发页错误     |
| 固定内存   | 主机      | 程序     | 传输快            | 加速 `cudaMemcpy`，不可分配太多  |

### 错误处理三步法

CUDA 中的错误可能出现在三个地方，需要逐层检查：

| 步骤 | 检查时机                       | 代码示例                                   | 捕获的错误                         |
| ---- | ------------------------------ | ------------------------------------------ | ---------------------------------- |
| ①    | 每次 API 调用返回后            | `err = cudaMalloc(...); if (err != cudaSuccess) ...` | 内存分配失败、无效参数等           |
| ②    | `<<<>>>` 启动后立刻            | `cudaGetLastError()` 或 `cudaPeekAtLastError()` | grid/block 维度非法、资源不足     |
| ③    | 同步（`cudaDeviceSynchronize`）后 | `err = cudaDeviceSynchronize(); if (err != cudaSuccess) ...` | kernel 执行崩溃、非法内存访问     |

```cpp
// 生产级错误处理宏（PDF 第 102 页提供）
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {
    // 用法示例
    checkCuda(cudaMallocManaged(&a, size));

    kernel<<<grid, block>>>(args);
    checkCuda(cudaGetLastError());          // 步骤②：检查启动错误

    checkCuda(cudaDeviceSynchronize());     // 步骤③：检查异步执行错误
}
```

### 性能剖析命令

| 命令                                        | 用途                                        |
| ------------------------------------------- | ------------------------------------------- |
| `nvprof ./executable`                       | 基础剖析：各 kernel 耗时、内存传输摘要      |
| `nvprof --print-gpu-trace ./executable`     | 完整 GPU 时间线——每次 kernel 启动和 memcpy  |
| `ncu -o report ./executable`                | Nsight Compute：SM 占有率、吞吐量、延迟     |
| `nsys profile ./executable`                 | Nsight Systems：CPU-GPU 交互图、stream 重叠  |

### 性能优化速查

| 优化策略                  | 关键做法                                              | 预期收益                     |
| ------------------------- | ----------------------------------------------------- | ---------------------------- |
| Block 大小选 32 的倍数     | `blockSize = 256` 或 `512`（与 warp=32 对齐）          | 避免 warp 浪费，提高 SM 占用 |
| 避免 data race             | 每个线程唯一索引 / `atomicAdd` / 规约算法              | 正确性保证                   |
| Coalesced 全局内存访问     | 同一 warp 内连续线程访问连续地址                       | 内存带宽利用率提升 3-10x     |
| 共享内存 tiling            | 将频繁访问的数据块载入 `__shared__` 后复用             | 减少全局内存往返             |
| Stream 并发                | `cudaMemcpyAsync` + 多个 stream 重叠传输与计算         | 隐藏 PCIe 延迟               |
| Kernel 融合                | 将多个小 kernel 合并为一个，减少启动开销               | 减少 kernel launch 开销      |
| 避开分支发散               | 确保同一 warp 内的线程走相同分支路径                   | 提升 warp 执行效率           |

---

## PDF 完整内容对照

PDF 第 77–105 页（对应 PDF 文件第 110–142 页附近）的逐页纲要，标注各节对应的知识点和规划的实现文件：

| PDF 页 | 书本页     | 内容                                                                | 实现文件 / 说明                                  |
| ------ | ---------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| 110    | 77         | GPU 加速 vs CPU 应用对比——跑车（CPU）vs 自行车队（GPU）类比          | 概念章节                                         |
| --     | 77–78      | 异构并行计算概念：host 编排 + device 执行、PCIe/NVLink               | --                                               |
| --     | 78–79      | 安装 NVIDIA 驱动（Windows/Linux/macOS 三步走）                       | --                                               |
| --     | 79–82      | 下载安装 CUDA Toolkit、PATH/LD_LIBRARY_PATH 配置、验证 `nvcc --version` | `00_cuda_hello.cu`                               |
| --     | 82–83      | 额外依赖：C++ 编译器、IDE、cuBLAS/cuFFT/cuDNN 库、云 GPU 环境        | --                                               |
| --     | 83–87      | Google Colab 中的 CUDA：启用 T4 GPU、安装 `nvcc4jupyter`、编译 .cu   | `00_colab_setup.cu`                              |
| --     | 87–89      | CPU 基线实现：`vectorAddCPU()` 逐元素加法、性能基准                  | `01_vector_add_cpu.cpp`                          |
| --     | 89–91      | **将 CPU 代码转为 CUDA**：`__global__ void add()`、`cudaMallocManaged`、`<<<1, 1>>>` 启动 | `02_vector_add_cuda.cu`                          |
| --     | 91–92      | `nvprof` 性能剖析入门、理解剖析摘要输出                               | `07_cuda_profiling.cu`                           |
| --     | 92–93      | CUDA 线程层次结构：Thread → Block → Grid + 内存空间（图 3.9）        | 概念章节                                         |
| --     | 93–96      | **多线程并行**：`<<<1, 256>>>`、grid-stride 循环、`threadIdx.x`/`blockDim.x`、data race 避免 | `03_vector_add_parallel.cu`                      |
| --     | 96–100     | **多 Block 扩展**：`<<<numBlocks, 256>>>`、全局线程索引 `blockIdx.x*blockDim.x+threadIdx.x`、SM 满利用 | `04_vector_add_grid.cu`                          |
| --     | 100–101    | 错误检查①：CUDA API 返回值（`cudaError_t`）                           | `05_cuda_error_handling.cu`（第一部分）          |
| --     | 101        | 错误检查②：`cudaGetLastError()` 捕获 `<<<>>>` 启动错误               | `05_cuda_error_handling.cu`（第二部分）          |
| --     | 101–102    | 错误检查③：`cudaDeviceSynchronize()` 检查异步执行错误                | `05_cuda_error_handling.cu`（第三部分）          |
| --     | 102        | `checkCuda()` 错误处理宏封装                                          | `05_cuda_error_handling.cu`（宏部分）            |
| --     | 102–103    | `dim3` 2D/3D 线程块和网格——`dim3(16,16,1)`、最大维度限制表（表 3.1） | `06_cuda_dim3.cu`                                |
| --     | 103–104    | 章节小结：从 CPU 基线到多 Block GPGPU 的完整学习路径                  | --                                               |
| --     | 104        | 课后问题 3 道                                                         | --                                               |
| --     | 104–105    | 拓展阅读：CUDA C 编程指南、最佳实践、Nsight、cuBLAS/cuFFT/cuDNN 等    | --                                               |
| --     | 105        | 参考答案                                                              | --                                               |
| 139–142| 107–113    | Part 2 引言（"Building and Training Neural Networks in C++"） + 第 4 章开头（"Building a Basic Neural Network in C++"） | 跳转至 Chapter 4                                 |

---

## 注意事项

### 硬件与驱动兼容性

| 平台     | 要点                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------- |
| Linux    | 驱动 + CUDA Toolkit 版本必须匹配；`nvidia-smi` 右上角显示的 CUDA 版本是驱动支持的最高版本             |
| Windows  | 需要 MSVC 2019/2022 编译器；安装后确认 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin` 在 PATH |
| WSL2     | 在 Windows 侧安装 NVIDIA 驱动，WSL 内使用 `nvidia-smi` 验证；需要 WSL2 内核 ≥ 5.10.16.3              |
| macOS    | Apple Silicon 无 CUDA 支持（使用 MPS 替代或云 GPU）；旧版 Intel Mac 部分支持但已逐步淘汰             |
| Colab    | 免费 T4 GPU（CC 7.5）；会话断开后文件丢失，建议挂载 Google Drive；nvcc4jupyter 对 LSP 有干扰警告       |

### 编译参数速查

```bash
nvcc -arch=sm_70    # Volta (V100)
nvcc -arch=sm_75    # Turing (T4, RTX 2080 Ti)  
nvcc -arch=sm_80    # Ampere (A100, A10)
nvcc -arch=sm_86    # Ampere (RTX 3090, A40)
nvcc -arch=sm_90    # Hopper (H100)
nvcc -run           # 编译后自动运行（仅演示用，不要用于生产脚本）
nvcc -lineinfo      # 包含行号信息，方便用 cuda-gdb 调试
nvcc -G             # 生成 device debug 信息（-G 会大幅降低性能，仅调试时使用）
```

可通过 `nvidia-smi --query-gpu=compute_cap --format=csv` 或 [NVIDIA GPU 计算能力列表](https://developer.nvidia.com/cuda-gpus) 查询 GPU 对应的 `-arch` 值。

### 常见问题排查

| 症状                                   | 原因                                                           | 解决方法                                                       |
| -------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| `nvcc: command not found`              | CUDA 未安装或 PATH 未配置                                      | `export PATH=/usr/local/cuda/bin:$PATH`                        |
| `no kernel image is available`         | `-arch` 参数与实际 GPU 计算能力不匹配                          | 用 `nvidia-smi --query-gpu=compute_cap` 确认后更正 `-arch`      |
| `cudaMallocManaged` 返回错误           | GPU 计算能力 < 6.0 不支持统一内存                              | 改用 `cudaMalloc` + 显式 `cudaMemcpy`                          |
| 24 CPU page faults（统一内存）         | 正常现象——统一内存在首次 GPU 访问时的页迁移                      | 如果成为瓶颈，改用显式管理 + `cudaMemPrefetchAsync`            |
| GPU 版本比 CPU 还慢                     | 只有 1 个 block/线程、大量页错误、kernel 太小被启动开销淹没    | 增加 block 数、数据量、使用显式内存管理                        |
| Colab 中 `nvcc` 找不到                 | Colab 默认无 CUDA 编译扩展                                      | `!pip install nvcc4jupyter` + `%load_ext nvcc4jupyter`         |
| `<<<1, -1>>>` 不报编译错误             | Kernel 启动参数在运行时校验                                     | 使用 `cudaGetLastError()` 捕获；负数不是合法线程数              |

### PDF 中提及但未独立实现的进阶主题

以下知识点在 PDF 第 3 章有提及或简介，但对应的深入实现安排在后续章节或需要额外库支持：

| 知识点                         | PDF 页 | 说明                                                                         |
| ------------------------------ | ------ | ---------------------------------------------------------------------------- |
| 固定内存（Pinned Memory）      | 78     | `cudaMallocHost` 分配，加速主机→设备传输；过度使用会耗尽系统内存             |
| Nsight Systems / Nsight Compute | 92     | NVIDIA 官方 GPU 剖析器（替代已弃用的 nvprof）；第 3 章仅展示 nvprof 入门      |
| cuBLAS                          | 82     | GPU 加速 BLAS（GEMM 等）；比手写 kernel 快 5-20x，LibTorch 底层依赖           |
| cuFFT                           | 82     | GPU 加速 FFT；替代手写 DFT，第 2 章 `audio_video.cpp` 提及                    |
| cuDNN                           | 82     | 深度学习基元库（卷积、池化、激活）；LibTorch/TensorFlow 底层依赖              |
| 共享内存 tiling                 | 91–100 | 将全局内存数据分块缓存到 `__shared__` 以减少重复读取；进阶优化主题            |
| CUDA Stream 并发                | 78     | 多个 stream 同时执行 kernel + 数据传输实现 overlap；进阶优化主题              |
| Kernel 融合                     | 78     | 将多个小 kernel 合并为一个以减少启动开销                                     |
| Roofline 模型 + Amdahl 定律     | 87,105 | 用于估算理论上限加速比和瓶颈分析                                              |

### 补充说明

- **nvprof 已弃用：** 从 CUDA 10.0 开始，`nvprof` 逐步被 Nsight Systems (`nsys`) 和 Nsight Compute (`ncu`) 替代。新版 CUDA Toolkit 中 `nvprof` 可能不可用，应优先使用 `nsys profile` 和 `ncu`。
- **统一内存（Unified Memory）的限制：** 在 CC ≥ 6.0（Pascal+）的 GPU 上支持按需页迁移；旧 GPU 上统一内存采用全量拷贝，性能很差。
- **动态并行：** `__global__` kernel 在 CC ≥ 3.5 且编译时加 `-rdc=true` 时可以递归启动其他 kernel（本章未涉及）。
- **cudaDeviceSynchronize vs cudaStreamSynchronize：** 前者等待 default stream（stream 0）上的所有工作；多 stream 场景下使用后者以保持并发性。

---

## 拓展阅读

- **NVIDIA CUDA C Programming Guide**（核心参考）: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices Guide**（占有率、coalescing、streams、UM 提示）: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Nsight Systems**（端到端 CPU-GPU 交互剖析）: https://developer.nvidia.com/nsight-systems
- **Nsight Compute**（单个 kernel 深度分析）: https://developer.nvidia.com/nsight-compute
- **Mark Harris - How to Optimize Data Transfers in CUDA**（固定内存、重叠、stream）: NVIDIA Developer Blog
- **Roofline Model**（性能上限分析）: https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/
- **CUDA Samples**（官方示例：grid-stride、规约、矩阵乘等）: https://github.com/NVIDIA/cuda-samples
- **cuBLAS 文档**（何时用库优于手写 kernel）: https://docs.nvidia.com/cuda/cublas/
- **cuDNN 文档**: https://docs.nvidia.com/deeplearning/cudnn/
