# 02｜GPU 推理管线：一次 forward 的数据流、同步点与延迟来源

## 本模块解决的问题

"模型在 GPU 上推理一次"不是一个原子操作，而是一条横跨 CPU、内存、PCIe/统一内存、GPU 前端、SM、显存层次的流水线。只有把这条流水线的**每一个 stage、数据的位置、Host↔Device 传输点、同步点**都画出来，才能在"慢"出现时判断它发生在哪一段。

本章回答三组问题：

```text
数据在哪里？        CPU memory / pinned memory / GPU HBM / unified memory
什么时候发生搬运？  H2D（输入进 GPU）、D2H（结果回 CPU）
同步点在哪里？      cudaDeviceSynchronize、.item()、copy 默认语义
每一段延迟多大？    launch / memcpy / kernel / sync 各占多少
```

配套代码：`src/inference/benchmark_latency.py`（wall vs event 的差就是 host 开销），`src/profiling/profile_target.py`（把管线各 stage 用 NVTX 标出来给 nsys 看）。

---

## 1. 一次推理的完整 stage 分解

以 `y = residual_mlp(x)`（`nn.Linear + LayerNorm + GELU` 堆叠）为例，从"输入在 CPU"到"输出可用"：

```text
1. input 在 host pageable memory
        │
2. 预处理 / normalize（CPU 上跑）
        │
3. H2D：copy 到 GPU memory（pinned 或 pageable，速度差数倍）
        │
4. 每个算子：dispatcher 选 ATen kernel → allocator 分配 → 提交 stream
        │
5. GPU front-end 调度 thread block 到 SM，warp 执行 kernel
        │
6. 输出留在 GPU memory（若下游还要用，就不 D2H）
        │
7. D2H：copy 回 CPU（只有 postprocess / 返回给调用方时才需要）
        │
8. host 读取结果（隐式或显式同步）
```

其中 4-5 是对一个含 N 个算子的小模型循环 N 次。对 batch=1 的小模型，**4（launch）往往比 5（执行）更贵**，这就是 launch-bound 的来源。

---

## 2. CPU 做什么，GPU 做什么

严格分工：

| 环节 | 谁在做 | 具体动作 |
|---|---|---|
| Python 前向调用 | CPU | 构建计算图/分派 |
| ATen dispatcher | CPU | 按 dtype/shape/device 选 kernel 实现 |
| allocator | CPU | 从 caching allocator 取/还显存块 |
| kernel launch | CPU | 把 kernel + 参数提交到 CUDA stream |
| kernel 执行 | GPU | SM 上实际计算 |
| memcpy | DMA/GPU | H2D/D2H 搬运（CPU 只发起） |

**关键认知**：CPU 是"指挥"，GPU 是"执行"。CPU 每一步都要花时间，GPU 却可能已经在跑上一个 kernel。两者是异步流水，中间靠 stream 排序、event 同步。

---

## 3. 数据在哪里（内存模型）

```text
host pageable memory   普通 malloc 出来的 CPU 内存，可被换页
host pinned memory     锁页内存，DMA 可直接访问，copy 更快（cudaHostAlloc / pin_memory）
device memory          GPU 全局显存（HBM / 统一内存）
unified memory         同一物理内存被 CPU+GPU 共享（Jetson/Thor 默认模型）
```

**本机（Jetson/Thor）是统一内存**：CPU 和 GPU 共享同一块物理 DRAM。这意味着：

- H2D 不是跨 PCIe 的"真搬运"，更多是**页迁移 / 一致性处理**，所以 pinned vs pageable 的差距远小于离散 GPU。
- 但 pinned 仍有意义：它保证 DMA 引擎可以绕过 CPU 直接访问，减少页表走查和逐页固定开销。

实测（`src/kernel/cuda_async/benchmark.py`，Thor 统一内存，64MB payload）：

```text
pageable H2D  ~83 GB/s
pinned  H2D   ~95-103 GB/s
```

对比离散 GPU（PCIe 上 pageable 可能只有 pinned 的 1/3~1/2），这个差距小得多——这正是统一内存平台的特征，笔记里必须如实记录而不是照搬 A100 的结论。

---

## 4. 什么时候发生 Host ↔ Device 传输

| 时机 | 方向 | 谁触发 |
|---|---|---|
| 输入进入 GPU | H2D | 显式 `x.to(device)` / `copy_` |
| 预处理在 CPU | 无 | 数据本来就在 CPU |
| 结果回 CPU | D2H | `.cpu()` / `.item()` / `.numpy()` |
| 日志/指标 | D2H | 每个 batch 取 `loss.item()` 之类 |

**常见性能事故**：在训练/推理循环里每步 `loss.item()`（隐式 D2H + 同步），把本来可以 overlap 的异步流水硬生生切断。见 `note/kernel/02_cuda_async_and_stream.md`。

---

## 5. 同步点在哪里

CUDA 是异步的，真正的"同步"只发生在少数显式/隐式点：

```text
显式：
  torch.cuda.synchronize()       全设备 barrier
  stream.synchronize()           单 stream barrier
  torch.cuda.Event + wait_event  跨 stream 依赖

隐式（最容易踩）：
  .item() / .cpu() / .numpy()    触发 D2H + 同步
  print(tensor)                  （调试时偷偷同步）
  torch.cuda.Stream 默认语义     同一 stream 内天然串行
```

每个同步点都是一个"气泡"：GPU 停下来等 CPU，或 CPU 停下来等 GPU。nsys timeline 里这些气泡就是 `cudaStreamSynchronize` / `cudaDeviceSynchronize` 的长条。

---

## 6. 每一段可能产生多少延迟（量级感）

下面是**量级**，不是精确值，用来建立直觉：

```text
kernel launch             ~2-8 us / 次（CPU 侧，大量 tiny kernel 会累加成大头）
H2D（PCIe 离散 GPU）      ~微秒到毫秒，取决于字节数和 pinned
H2D（统一内存）           明显更小，页迁移成本
一次小 kernel 执行        ~1-10 us
一次 GEMM 执行            ~几十 us 到 ms，取决于 shape
一次 .item() 同步         ~5-50 us（D2H + barrier）
allocator 首次分配        可能触发 cudaMalloc（慢，可到 ms）
```

**batch=1 推理的典型画像**：launch 开销 + 几十个 tiny kernel，总 wall 时间被 CPU 侧的 launch 支配，而不是被 GPU 计算支配。对应实测：本机 4 层小模型 wall 237us vs event 229us，host 开销约 8us；换到更小的 kernel 链（`cuda_graph` 实验）这个比例会剧烈放大。

---

## 7. 可能消耗多少显存

推理的显存由三部分构成：

```text
1. 权重 weights        参数量 * dtype bytes
2. 激活 activations    中间张量（batch * seq * hidden * layers * dtype）
3. workspace / 临时      cuBLAS/cuDNN workspace、allocator 碎片、KV cache（LLM 阶段）
```

对小模型，权重主导；对大 batch 或长序列，激活主导；对 LLM decode，KV cache 主导。精确数字用 `torch.cuda.memory_allocated()` 和 `torch.cuda.max_memory_allocated()` 测，不要心算。注意 caching allocator 的 `reserved` 通常大于 `allocated`（碎片和预留），两者含义不同。

---

## 8. 用 timeline 证明"哪一段最慢"

单靠 wall time 只能告诉你"总共 237us"，不能告诉你这 237us 花在哪。要让管线可观测：

1. 在代码里给每个 stage 打 NVTX range（见 `src/profiling/profile_target.py` 的 `h2d` / `block_N` / `postprocess`）。
2. `nsys profile --trace=cuda,nvtx,osrt` 采集。
3. 在 nsys-ui 里看：CPU 线程在忙什么、GPU 在忙什么、中间的空隙在哪。

```text
CPU:  dispatch | launch | dispatch | launch | .item() wait........ | next
GPU:           [kernel1][kernel2][kernel3]                          [kernel4]
               ↑ 无空隙 = compute 主导        ↑ 长空隙 = CPU 喂不饱
```

如果 GPU 行上 kernel 之间没有空隙，说明是 compute/device 主导；如果 GPU 有大量空隙而 CPU 行一直忙，说明是 CPU/launch 主导。这就是"CPU 是否喂得饱 GPU"的直接证据。详细 SOP 见 `note/profiling/01_nsys_inference_profiling.md`。

---

## 9. 工业故障与定位

| 现象 | 第一证据 | 定位动作 |
|---|---|---|
| 延迟高但 GPU-Util 低 | nsys：GPU 空隙多 | 看 CPU 行是否满载 → CPU-bound / launch-bound |
| 延迟高且 GPU-Util 100% | ncu：compute/DRAM 瓶颈 | 判断 compute-bound vs memory-bound |
| 延迟周期性尖刺 | p99/max 远高于 mean | 找 allocator、GC、clock 抖动、thermal |
| 显存持续上涨 | `max_memory_allocated` 增长 | 找缓存泄漏 / 无界 cache / KV cache 未回收 |

---

## 10. 本模块闭环小结

```text
问题：一次推理的时间去哪了
      ↓
原理：CPU 指挥、GPU 执行，异步流水 + 同步气泡
      ↓
管线：preprocess → H2D → N×(launch+kernel) → D2H → postprocess
      ↓
测量：wall vs event、NVTX 分 stage、memory_allocated
      ↓
证据：nsys timeline 看空隙归属
      ↓
优化：针对 host 开销（fusion/graph）或 device 瓶颈（kernel/精度）分别下手
```

下一模块：`03_latency_and_throughput.md`，定量回答"batch 怎么改变 latency 和 throughput，二者如何权衡"。
