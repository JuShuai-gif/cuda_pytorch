# 03｜访存合并与共享内存 Bank Conflict：用原生 CUDA 验证

## 本模块解决的问题

Stage 1 的 `note/kernel/01` 讲了 memory coalescing 和 bank conflict 的理论。本章用**原生 CUDA C++**（不是 PyTorch 封装）把它跑成可复现的数字，并展示边缘端真正的工作方式：

```text
写 .cu → nvcc 编译（ptxas -v 打印 register/shared）→ 跑二进制 → 读 JSON
```

回答：

```text
strided 访问到底比 coalesced 慢多少？为什么不是理论上的 32x？
bank conflict 对共享内存带宽的真实影响？
为什么要走到 CUDA C++ 而不是停在 PyTorch？
```

配套代码：`src/kernel/cuda_core/src/coalescing.cu`、`src/kernel/cuda_core/src/bank_conflict.cu`，构建见 `src/kernel/cuda_core/README.md`。

---

## 1. 为什么边缘端要用 CUDA C++

边缘端（Jetson/Thor、机器人、VLA runtime）的推理代码主流是 C++，原因：

```text
1. TensorRT 的生产 API 是 C++（Python 只是原型层）
2. 自定义 kernel / plugin 只能用 CUDA C++（或 CUTLASS）
3. 低延迟 runtime 要避开 Python 解释器、dispatcher、GIL
4. 机器人集成（ROS/ROS2、传感器、控制器）是 C++ 生态
5. 编译期就能拿到 register/shared 用量，做 occupancy 决策
```

Python 的定位是原型、benchmark、serving 编排（原 master prompt 的「Python prototype → C++ runtime」）。所以从本模块开始，CUDA 层的实验全部用原生 `.cu`。

---

## 2. 编译链再确认：CUDA C++ → PTX → SASS

本模块的每个二进制都走完整编译链：

```text
coalescing.cu
   ↓ nvcc -arch=sm_110 -Xptxas -v
PTX（虚拟 ISA）
   ↓ ptxas（打印 register/shared 用量）
SASS（sm_110 机器码）
   ↓
GPU
```

用三个命令分别看三层：

```bash
cuobjdump --dump-ptx  build/bin/occupancy   # PTX：.visible .entry ... .reg .pred
cuobjdump --dump-sass build/bin/occupancy   # SASS：LDC R1, c[0x0][0x37c]
cuobjdump --dump-resource-usage build/bin/occupancy  # REG:72 STACK:0 ...
```

其中 `-Xptxas -v` 在**编译时**就打印每个 kernel 的 register 数和 shared memory 用量，是判断 occupancy 的第一手材料（本模块 `build.sh` 已默认开启）。

---

## 3. 实验一：Memory Coalescing（访存合并）

### 设计

两个 kernel 都读 16MB 输入、写 16MB 输出，唯一的区别是访问模式：

```cuda
// coalesced：thread i 读 in[i]，一个 warp 的 32 次访问是连续的
__global__ void read_coalesced(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// strided：thread i 读 in[i * 32]，相邻 thread 相隔 32 个 float = 128 字节
__global__ void read_strided(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * STRIDE;           // STRIDE = 32
    if (idx < n) out[idx] = in[idx] * 2.0f;
}
```

### 本机实测（Thor, sm_110，100 次取均值）

```text
coalesced：434 GB/s
strided  ：216 GB/s
slowdown ：2.0x
```

### 为什么不是理论上的 32x？

stride=32 个 float = 128 字节，正好是一个 cache line。所以 strided 访问时，每个 thread 读 4 字节却拉进 128 字节的 cache line，**有效数据利用率 1/32**，理论上带宽应掉到 1/32。

但实测只慢 2x，原因是：

1. **L2 cache**：整个 16MB 输入装得进 32MB L2，第一次读后全部命中 L2，后续 99 次迭代都在 L2 里读，DRAM 带宽不再是瓶颈，瓶颈变成 L2 带宽。
2. **相邻 thread 访问相邻 cache line**：strided 访问仍保留了顺序性，L2 的硬件预取和 sector 机制能部分补偿。

**教训**：内存系统的真实性能取决于"数据是否命中缓存 + 访问是否合并"两个因素，单看"地址是否连续"会得出错误结论。要精确判断合并程度，用 ncu 的 `gld_efficiency` / `L2 sectors`（本机 ncu 需授权，见 `note/profiling/02`）。

### 什么时候 slowdown 会接近 32x

- 数据集**超出 L2**（本机 >32MB），每次都要回 DRAM。
- **随机 gather**（thread i 读 in[perm[i]]），破坏顺序性，预取失效。

---

## 4. 实验二：Bank Conflict（共享内存冲突）

### 设计

shared memory 分 32 个 bank（每 bank 4 字节）。三种访问模式，循环 256 次累加：

```cuda
// none：thread i 读 s[(tid + k) % N]，32 个 thread 打 32 个不同 bank
acc += s[(tid + k) % N];

// 2-way：thread i 读 s[(tid*2 + k) % N]
//   thread 0 和 16 都命中 bank 0（地址 s[0] vs s[32]），每 bank 2 路冲突
acc += s[((tid * 2) + k) % N];

// 32-way：thread i 读 s[(tid*32 + k) % N]
//   所有 thread 都命中 bank 0（不同地址），32 路冲突
acc += s[((tid * 32) + k) % N];
```

### 本机实测（Thor, sm_110，200 次取均值）

```text
none  ：0.0072 ms
2-way ：0.0081 ms  （1.13x）
32-way：0.0472 ms  （6.6x）
```

### 读法

1. **32-way 慢 6.6x**：所有 32 个 thread 争抢同一个 bank，访问被串行化，效果显著。
2. **2-way 只慢 1.13x**：2 路冲突理论上慢 2x，但这里冲突开销被循环里的算术指令部分掩盖——共享内存不是这个 kernel 的唯一瓶颈。这说明"bank conflict 的影响要放到具体 kernel 的指令混合里看"，不是所有 conflict 都致命。

### 生产意义

- **转置、reduction、matmul 的 shared tile** 若布局不当会产生严重 conflict，吃掉共享内存本应提供的高带宽。
- 常用规避：pad 一行（如 `s[TILE][TILE+1]`）让相邻行的 bank 错开。
- 广播（所有 thread 读同一地址）不算 conflict，可放心用。

---

## 5. 与 Stage 1 PyTorch 版的对照

| 实验 | PyTorch 版（Stage 1） | C++ 版（本模块） |
|---|---|---|
| pinned vs pageable | `pin_memory()` | `cudaHostAlloc` / `cudaMallocHost` |
| async H2D | `copy_(non_blocking=True)` | `cudaMemcpyAsync` |
| stream | `torch.cuda.Stream` | `cudaStreamCreate` / `<<<..., stream>>>` |
| graph | `torch.cuda.CUDAGraph` | `cudaStreamBeginCapture` + `cudaGraphLaunch` |

C++ 版的价值：**控制粒度到 API 级**，能看到 PyTorch 替你隐藏的细节（比如 `cudaMemcpy` 是同步、`cudaMemcpyAsync` 才异步、capture 必须独占一条 stream），这正是写 TensorRT plugin / 自定义 runtime 时每天要面对的东西。

---

## 6. 本模块闭环小结

```text
问题：访存模式和共享内存布局对带宽的影响到底多大
      ↓
原理：coalescing（按段传输）+ bank conflict（32 bank 串行化）
      ↓
实现：原生 CUDA C++（coalescing.cu / bank_conflict.cu）
      ↓
证据：ptxas -v 的 register/shared + 实测带宽 + cuobjdump 看 SASS
      ↓
结论：strided 慢 2x（L2 缓解）、32-way conflict 慢 6.6x
```

下一模块：`note/kernel/04_occupancy_register_pressure.md`，回答"register 用量如何决定 occupancy，`__launch_bounds__` 如何反过来控制 register"。
