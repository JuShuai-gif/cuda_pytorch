# CS149 第 7 讲：GPU 体系结构与 CUDA 思维

**来源**：Stanford CS149，2025 年秋季，第 7 讲

---

## 本讲核心问题

1. GPU 为什么能提供远高于 CPU 的吞吐？
2. CUDA 的 grid / block / thread 层次到底在表达什么？
3. 共享内存为什么是 GPU 优化的核心抓手之一？
4. GPU 的性能为什么高度依赖访存模式和资源占用？

---

## 1. GPU 的设计哲学：把晶体管预算投入吞吐

### 1.1 CPU 与 GPU 的侧重点不同

- CPU：重视低延迟、复杂控制流、单线程表现。
- GPU：重视高吞吐、规则并行、海量数据处理。

### 1.2 为什么 GPU 能堆很多算术单元

因为它把更多资源用在：

- ALU / FMA / Tensor Core
- 更多并发线程上下文
- 更高总带宽的显存接口

而相对减少了 CPU 中那些昂贵的复杂控制逻辑。

### NVIDIA V100 完整规格

| 参数 | 数值 |
|---|---|
| 时钟频率 | 1.245 GHz（可 boost） |
| SM 数量 | 80 |
| 每 SM 子核心数 | 4 |
| 每子核心 fp32 ALU | 16（1 个 32-wide SIMD 需 2 clock） |
| 每子核心 int ALU | 16 |
| 每子核心 fp64 ALU | 8（1 个 32-wide SIMD 需 4 clock） |
| 每 SM 最大 warps | 64 |
| 每子核心 register 容量 | 支持最多 16 warps |
| Shared Memory + L1 | 128 KB |
| 总 fp32 mul-add ALUs | 5,120 |
| 峰值 FP32 | 12.7 TFLOPS |
| L2 Cache | 6 MB |
| HBM2 | 16 GB, 900 GB/sec (4096-bit) |
| 最大并发 threads | 163,840 |

### 1.3 代价是什么

- 更依赖程序暴露大量并行性
- 更怕分支分歧
- 更怕无规律访存
- 更强调延迟隐藏而非缩短单次延迟

### 1.4 GPU 的历史演变：从图形渲染到通用计算

#### 1.4.1 GPU 的图形渲染起源

GPU 最初为 3D 图形渲染管线设计：
- **三角形网格（Primitives）** → **顶点处理（Vertex Processing）** → **片段处理（Fragment Processing）**
- 每个像素/片段执行相同的 shader 程序（纯函数），输入包括纹理、光照方向、法线、UV 坐标等

```glsl
// GLSL shader 示例
void myShader() {
    vec3 normal = normalize(norm);
    vec3 light = normalize(lightDir - position);
    float diffuse = max(0.0, dot(normal, light));
    gl_FragColor = diffuse * texture2D(myTexture, uv);
}
```

#### 1.4.2 GPGPU 的早期 hack (2001-2003)

在 CUDA 出现之前，研究者通过"技巧"将通用计算映射到 GPU：
- 把输出图像设为 512×512
- 渲染两个覆盖整个屏幕的三角形
- 将 shader 函数映射到 512×512 个元素集合上
- 例如 Harris 02、Purcell 02、Bolz 03 等论文的工作

#### 1.4.3 Brook 流编程语言 (2004)

Stanford 图形实验室开发的流编程语言，首次将 GPU 抽象为数据并行处理器：
```
kernel void scale(float amount, float a<>, out float b<>) {
    b = amount * a;
}
```
编译器将流程序翻译为图形命令和 shader 程序。[Buck 2004]

#### 1.4.4 Tesla 架构与 Compute Mode (2007)

接口转变：
- 从 `drawPrimitives(vertex_buffer)` 到 `launch(myKernel, N)`
- GPU 从纯图形处理器进化为通用并行计算平台

---

## 2. CUDA 线程层次：grid、block、thread

### 2.1 为什么需要层次化并行

GPU 上线程数量极多，不能把所有线程看成平铺的一层。CUDA 采用：

- **Grid**：一次 kernel 调用的所有线程块集合
- **Block**：一组可以协作、可共享片上内存、可 block 内同步的线程
- **Thread**：最小逻辑执行实例

### 2.2 block 的意义

一个 block 往往对应：

- 一块数据 tile
- 一个可合作加载共享内存的工作组
- 一个调度到单个 SM 上执行的资源单位

### 2.3 thread 的意义

每个线程通常负责：

- 一个元素
- 一个输出位置
- 一个子任务的一小部分

### 2.4 为什么 block 不能随便跨边界协作

- block 之间没有廉价、即时的硬件级全局同步。
- 设计上就是为了让不同 block 能独立调度和扩展。

### 2.1.1 CUDA 具体语法

- `__global__` 声明 kernel 函数（从 host 调用，在 device 执行）
- `__device__` 声明设备函数（仅可从 device 调用）
- `dim3` 类型用于 2D/3D 线程维度
- Launch 语法：`myKernel<<<numBlocks, threadsPerBlock>>>()`
- 线程全局 ID 计算：`blockIdx.x * blockDim.x + threadIdx.x`
- 当数据大小不是 blockDim 整数倍时需要 guard clause 防越界访问

```cuda
__global__ void matrixAdd(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}
```

### 2.1.2 CUDA 内存模型

- `cudaMalloc(&deviceA, bytes)` — 在设备上分配全局内存
- `cudaMemcpy(deviceA, A, bytes, cudaMemcpyHostToDevice)` — 主机到设备传输
- 主机不能直接通过指针访问设备地址空间
- 三种设备地址空间：per-thread private、per-block shared、per-program global

> 对应源码：`lecture7_part1.cpp`
> 内容：grid / block / thread 层次与索引映射的 CPU 模拟。

---

## 3. Warp 执行与 SIMD/SIMT 语义

### 3.1 Warp 是什么

- GPU 实际上常以 warp 为调度与执行单位。
- 典型大小是 32 个线程。

### 3.2 SIMT 与 SIMD 的关系

- 从编程视角，CUDA 看起来像很多独立线程。
- 从硬件视角，warp 中线程常常以锁步方式执行同一条指令。
- 这被称为 SIMT（Single Instruction, Multiple Threads）。

### 3.3 分支分歧为什么伤性能

如果 warp 内不同线程走不同分支：

- 硬件通常要分批执行不同路径
- 不在当前路径上的线程 lane 会被屏蔽
- 等效并行度下降

这和 CPU SIMD 中 mask 执行的代价是同一类问题。

---

## 4. 共享内存：GPU 上的显式可管理片上缓存

### 4.1 共享内存是什么

- 位于每个 SM 上的片上 SRAM
- block 内线程可共同读写
- 延迟低、带宽高

### 4.2 为什么共享内存很重要

因为 GPU 的全局内存虽然总带宽很高，但单次访问延迟仍然很大。共享内存允许：

- 把重复访问的数据先搬到片上
- 多个线程合作完成一次加载
- 后续在片上多次复用

### 4.3 典型模式：协作加载 + 复用

例如卷积：

- 相邻输出会共享一部分输入窗口
- 若每个线程各自从全局内存重复读取，流量巨大
- 若先把 tile 加载进共享内存，再重复使用，成本显著下降

### 4.0.1 1D 卷积：全局内存 vs 共享内存对比

**版本 1（仅全局内存）**：128 个线程，每个做 3 次全局加载 = 384 次 load

**版本 2（共享内存 + 协作加载）**：
- 128 个线程 + 额外的 2 个边界元素 → 130 次 load 替代 384 次
- load count 减少约 3x
- 使用 `__syncthreads()` barrier 确保所有线程完成加载后开始计算

### 4.0.2 CUDA 同步原语

- `__syncthreads()`：block 内 barrier
- 原子操作：`atomicAdd`、`atomicMin`、`atomicMax` 等（可用于全局和共享内存）
- Kernel 返回时提供隐式的 host/device barrier

> 对应源码：`lecture7_part2.cpp`
> 内容：1D 卷积中朴素全局访存与共享内存缓存版本对比。

---

## 5. GPU 内存层次

常见层次包括：

- **Global Memory**：大、慢、所有线程可见
- **Shared Memory**：块级共享、片上、快
- **Registers**：线程私有、最快
- 以及只读缓存、常量缓存、纹理路径等特定机制

### 5.1 不同层次的核心权衡

| 层次 | 可见范围 | 容量 | 速度 | 典型用途 |
|---|---|---:|---|---|
| Register | 单线程 | 很小 | 最快 | 私有临时值、累加器 |
| Shared Memory | 单 block | 小 | 很快 | tile 缓冲、线程协作 |
| Global Memory | 全设备 | 大 | 慢 | 主输入输出张量 |

### 5.2 GPU 优化的本质

大多数高性能 GPU kernel 都在做同一件事：

- 尽量把数据从 global 拉到 shared / register
- 尽量在片上做更多复用
- 尽量减少不规则访问与重复搬运

> 对应源码：`lecture7_part3.cpp`
> 内容：全局内存、共享内存、寄存器、原子操作与 block 调度约束模拟。

---

## 6. Occupancy、资源约束与延迟隐藏

### 6.1 GPU 为什么需要很多线程

因为单个线程会经常等待：

- 全局内存访问
- 长流水计算
- 某些同步点

GPU 通过保留大量可运行 warp，在某个 warp 等待时切换去执行另一个 warp，从而隐藏延迟。

### 6.2 Occupancy 是什么

Occupancy 可以粗略理解为：

- 一个 SM 上同时驻留的活跃线程 / warp 数量相对于最大值的比例。

### 6.3 为什么 occupancy 不是越高越好

虽然更高 occupancy 有助于隐藏延迟，但如果为了追求 occupancy 而：

- 减少寄存器使用导致 spill
- 放弃共享内存复用
- 降低每线程有效工作量

那也可能得不偿失。

### 6.4 真实优化要看的不是单一指标

还要综合：

- 访存模式
- 共享内存占用
- 寄存器压力
- warp 分歧
- 实际瓶颈是带宽、算力还是延迟

### 6.0.1 Thread Block 调度详细原理

CUDA 编译器生成 device binary 包含：
- 程序指令
- 资源需求信息：128 threads per block、B bytes local data per thread、520 bytes shared space per block

GPU 调度器根据这些资源需求将 blocks 映射到 SM。以一个虚构的 2 核 GPU 为例：
- 每个 SM 有固定数量的执行上下文 slot 和共享内存
- Block 调度是 interleaved 的（轮流分配到不同 SM）
- 当资源不足时（如 3×520 > 1.5KB shared memory），新的 block 必须等待

### 6.0.2 为什么 Block 内线程必须同时存在

假设一个 256 线程的 block 运行在仅有 128 线程容量的 SM 上：
- 不能简单先跑 0-127 再跑 128-255
- 因为 `__syncthreads()` 创建了跨线程的依赖关系
- CUDA 语义：block 内线程在逻辑上是并发的

### 6.0.3 Cross-block 同步的危险

两个 thread block 通过全局标志做握手同步是**不安全的**：
```cuda
// 危险做法！
if (blockIdx.x == 0) {
    // ... 计算 ...
    myFlag[0] = 1;
} else if (blockIdx.x == 1) {
    while (myFlag[0] != 1); // 等待 block 0
}
```
block 的执行顺序不确定！单 SM GPU 可能让 block 1 先执行，导致死锁。

### 6.0.4 Persistent Thread 编程风格

完全绕过 GPU 的 thread block 调度器，用 `atomicInc` 自行分配工作。需硬编码 `BLOCKS_PER_CHIP`，所有 block 同时运行，程序员假设所有 CUDA 线程同时存活。这种风格在极端性能优化中使用。

### 6.0.5 CUDA 与 ISPC 的概念映射

| CUDA | ISPC/pthread |
|---|---|
| Thread Block | ISPC task（无依赖，可任意顺序调度）|
| Warp | ≈ ISPC gang（SIMD 执行）|
| Warp 不是编程模型概念 | 是 GPU 硬件实现细节 |

---

## 7. 原子操作与协作更新

### 7.1 为什么需要原子

当多个线程要更新同一共享位置时：

- 普通读改写会产生竞争
- 原子操作保证更新不可被打断

### 7.2 原子的代价

- 序列化热点更新
- 增加一致性 / 仲裁成本
- 在高冲突场景下可能严重降速

### 7.3 工程上的常见替代思路

- 先做 block 内局部归约
- 使用分层直方图
- 用 sort + scan 代替细粒度冲突写

这为后续数据并行模式和 histogram 设计埋下了伏笔。

---

## 8. 从本讲得到的 GPU 编程直觉

1. 先思考如何把问题切成大量相似线程。
2. 再思考哪些数据可在一个 block 内协作复用。
3. 再思考访存是否连续、是否合并。
4. 最后再看 occupancy、同步和资源约束。

也就是说，GPU 优化的第一原则通常是：

- **让内存访问规则化，让片上复用最大化。**

---

## 常见误区

1. **误区：GPU 就是“线程更多的 CPU”。**
   实际执行模型、内存层次和优化重点都完全不同。
2. **误区：只要开很多线程就能快。**
   若访存乱、分歧多、片上复用差，线程再多也喂不饱硬件。
3. **误区：共享内存只是一个小缓存。**
   它更像程序员显式控制的数据复用空间。
4. **误区：occupancy 越高性能越高。**
   高 occupancy 只是手段，不是目标。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture7_part1.cpp` | CUDA 线程层次 | grid / block / thread 索引映射 |
| `lecture7_part2.cpp` | 共享内存卷积 | 协作加载、数据复用、减少全局访存 |
| `lecture7_part3.cpp` | GPU 内存层次与调度 | global/shared/register、原子与资源约束 |

---

## 学完本讲应做到

- 能解释 GPU 与 CPU 的设计目标差异。
- 能看懂 CUDA 线程层次为什么这样设计。
- 能理解共享内存对性能的重要性。
- 能从 warp 分歧、访存模式、occupancy 三个角度分析 GPU kernel。

