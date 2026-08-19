# 04｜Occupancy 与 Register Pressure：用 `__launch_bounds__` 控制寄存器

## 本模块解决的问题

Stage 1 的 `note/kernel/01` 讲了 occupancy 和 register pressure 的理论。本章用原生 CUDA C++ 把它变成可查询的数字，回答：

```text
register/thread 怎么决定 occupancy？
__launch_bounds__ 到底在"限制"什么，又是怎么反过来影响寄存器的？
为什么"限制线程数"反而会让编译器用更多寄存器、降低 occupancy？
CUDA Graph / stream 在 C++ 层又是怎么工作的？
```

配套代码：`src/kernel/cuda_core/src/occupancy.cu`（register/occupancy）、`src/kernel/cuda_core/src/async_copy.cu`、`stream_overlap.cu`、`graph_launch.cu`。

---

## 1. 寄存器文件是 occupancy 的硬上限

每个 SM 的寄存器文件是固定的（本机 Thor `regsPerBlock = 65536`）。一个 kernel 占多少 register，直接决定一个 SM 能同时驻留多少 thread：

```text
max_threads_per_sm = regsPerBlock / registers_per_thread（向下取整到 block 粒度）
occupancy         = 实际驻留 warp / 理论最大 warp
```

所以 register 与 occupancy 是**反比**：register/thread ↑ → threads/SM ↓ → occupancy ↓。

```text
40 regs/thread  →  65536/40 = 1638 → 1536 threads（打满）→ 100% occupancy
72 regs/thread  →  65536/72 = 910  → 768 threads（3 blocks）→ 50% occupancy
```

---

## 2. `__launch_bounds__` 的两个参数

```cuda
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS) kernel(...);
```

- `MAX_THREADS`：每 block 最多多少 thread。
- `MIN_BLOCKS`：每个 SM 至少驻留多少 block（可选）。

它**不是**直接"限制寄存器"，而是告诉编译器"必须满足这个线程/block 约束"，编译器据此**推算允许的寄存器上限**：

```text
允许的 regs/thread = regsPerBlock / (MAX_THREADS * MIN_BLOCKS)
```

- `__launch_bounds__(256, 6)` → 上限 ≈ 65536/(256*6) ≈ 42 regs/thread → 编译器被迫省寄存器（必要时 spill 到 local memory）。
- `__launch_bounds__(256, 1)` → 上限 ≈ 65536/256 = 256 regs/thread → 编译器可以放手用更多寄存器。

---

## 3. 实验：三个 kernel 的寄存器与 occupancy

同一个 64-float 累加 kernel，三种 launch_bounds：

```cuda
__global__ void kernel_default(...)                     // 无约束
__global__ void __launch_bounds__(256, 6) kernel_high_occupancy(...)
__global__ void __launch_bounds__(256, 1) kernel_low_occupancy(...)
```

用 occupancy API 查询（`cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`），本机实测：

```text
high_occupancy : 40 regs/thread, local=0, 6 blocks/SM, 1536 threads, occupancy 100%
default        : 40 regs/thread, local=0, 6 blocks/SM, 1536 threads, occupancy 100%
low_occupancy  : 72 regs/thread, local=0, 3 blocks/SM,  768 threads, occupancy 50%
```

`cuobjdump --dump-resource-usage` 交叉验证：

```text
Function _Z21kernel_high_occupancyPKfPf: REG:40 STACK:0 LOCAL:0
Function _Z14kernel_defaultPKfPf:       REG:40 STACK:0 LOCAL:0
Function _Z20kernel_low_occupancyPKfPf: REG:72 STACK:0 LOCAL:0
```

### 读法（一个关键的反直觉点）

1. **high_occupancy 和 default 相同（40 regs）**：这个 kernel 的 64-float 数组编译器用 40 个寄存器（通过寄存器重用 + 指令调度）就装得下，所以 `__launch_bounds__(256,6)` 的 42-reg 上限没有触发 spill，结果和 default 一致。

2. **low_occupancy 用了 72 regs**：放开约束后，编译器选择"多用寄存器、展开更激进、减少指令依赖"，代价是 occupancy 从 100% 掉到 50%。

3. **结论**：`__launch_bounds__` 是**反向控制**工具——你约束线程数，编译器响应式调整寄存器。想要高 occupancy，就给它更紧的 block 约束（可能引入 spill）；想要编译器自由优化，就放松约束（可能掉 occupancy）。

4. **occupancy 是手段不是目的**：本例里 `local=0` 说明三个版本都没 spill。真实优化中，高 occupancy（少寄存器）可能因 spill 到 local memory 反而更慢，低 occupancy（多寄存器）可能因更少访存更快。**要用 ncu 的 stall 指标判断哪个 win**，而不是盲追 occupancy。

---

## 4. 实验补充：C++ 层的异步、stream、graph

这三个实验在 C++ 层复刻了 Stage 1 的 PyTorch 版，用真正的 CUDA API。

### 4.1 async_copy：pinned vs pageable + 同步 vs 异步

```text
pageable_sync：94.6 GB/s
pinned_sync  ：100.9 GB/s
pinned_async ：100.1 GB/s
```

与 PyTorch 版结论一致：统一内存平台上 pinned 略快于 pageable（无 PCIe 跨越，差距小）。C++ 层能看到 `cudaMemcpy`（同步、阻塞 host）与 `cudaMemcpyAsync`（异步、host 立即返回）的 API 差异。

### 4.2 stream_overlap：单 stream vs 多 stream

设计要点：每个 kernel 只用 **5 个 block（20 SM 的 1/4）**，故意吃不饱 GPU。

```text
single_stream：7.31 ms
multi_stream ：2.14 ms
speedup      ：3.42x
```

**读法**：单 stream 时 4 个 kernel 串行，每个只用 5/20 SM，其余 15 个 SM 空闲；多 stream 时 4 个 kernel 并行填满 20 个 SM。这是 multi-stream 收益的**前提条件**——单个 kernel 吃不饱 GPU。反过来说，如果 kernel 已经吃满 GPU（Stage 1 里 512×512 GEMM 那种），多 stream 几乎无收益（实测 ~1.0x）。

### 4.3 graph_launch：normal vs CUDA Graph

64 个 tiny kernel 的链，用 `cudaStreamBeginCapture` + `cudaGraphLaunch`：

```text
normal：wall 0.173 ms / event 0.155 ms
graph ：wall 0.074 ms / event 0.056 ms
wall speedup 2.35x，event speedup 2.75x
```

**读法**：wall 和 event **同降**，说明这个 workload 同时被 launch 开销（host 侧）和 kernel 间隙（device 侧）拖累。graph 重放一次 launch 拉起整张图，两者一起消除。注意 capture 必须独占一条 stream（`cudaStreamBeginCapture` 的语义），这是 PyTorch 版替你处理、C++ 版必须自己写的细节。

---

## 5. C++ 层才看得到的东西

| 层面 | Python 能看到 | C++ 能看到 |
|---|---|---|
| register/shared 用量 | 间接（torch profiler） | `ptxas -v` / `cuobjdump` 直接打印 |
| launch_bounds 控制 | 无 | `__launch_bounds__` |
| memcpy 同步/异步语义 | `copy_` 的隐式语义 | `cudaMemcpy` vs `cudaMemcpyAsync` 显式 |
| occupancy 查询 | 无 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| SASS 反汇编 | 无 | `cuobjdump --dump-sass` |

这就是"边缘端更常用 C++"的实锤：**kernel 层的所有决策依据（寄存器、occupancy、访存、指令）都是 C++ 层拿到的**，Python 只能看到一个黑盒的 wall time。

---

## 6. 本模块闭环小结

```text
问题：register 怎么决定 occupancy，如何主动控制
      ↓
原理：occupancy = 驻留 warp / 最大 warp，register 与它反比
      ↓
工具：__launch_bounds__（约束线程→反向控制寄存器）+ occupancy API
      ↓
实测：40 regs→100%，72 regs→50%；多 stream 3.42x（吃不饱时）；graph 2.35x
      ↓
结论：occupancy 是手段不是目的，最终用 ncu stall 指标裁决
```

---

## 7. Stage 2 收尾：从 PyTorch 到 CUDA C++ 的能力闭环

至此，Stage 1（PyTorch 基线 + nsys/ncu SOP）和 Stage 2（原生 CUDA C++ 验证）合起来形成：

```text
PyTorch 基线（inference/）
   ↓ 定位问题（wall vs event、batch sweep）
nsys（profiling/）找慢 kernel / 气泡
   ↓
ncu（profiling/）看 memory/compute/occupancy/stall
   ↓
原生 CUDA C++（kernel/cuda_core/）验证访存、bank、occupancy、stream、graph
   ↓
用 before/after 数据证明优化
```

下一模块进入 Stage 4：**Triton Kernel**（Vector Add → Reduction → Softmax → LayerNorm → RMSNorm → GEMM → Attention → Quantization），逐个实现 + 正确性 + benchmark + ncu 分析。
