# 02｜CUDA 异步、Stream、Event 与 CUDA Graph

## 本模块解决的问题

GPU 推理的很多延迟不在 GPU 里，而在"CPU 如何把工作交给 GPU"这个环节。本章回答：

```text
kernel launch 返回了，工作真的完成了吗？
synchronous H2D 和 asynchronous H2D 差在哪？
pinned memory 为什么更快？
single stream 和 multi stream 什么时候有区别？
CUDA Graph 为什么能把 launch-bound 的延迟砍掉一大截？
```

配套代码：`src/kernel/cuda_async/`（pinned / non_blocking / stream 实验）、`src/kernel/cuda_graph/`（normal vs graph）。

---

## 1. 异步执行模型

CPU 调用一个 CUDA 操作（kernel launch 或 `copy_`）时，通常只是把工作**入队**到某条 stream，然后立即返回：

```text
CPU:  launch A | launch B | launch C | ...（飞快入队，不等 GPU）
GPU:            [A][B][C]            （按 stream 顺序执行）
```

所以：

- **launch 返回 ≠ 完成**。用 `time.time()` 夹住 `model(x)` 往往只测到入队时间。
- **同一 stream 内顺序保证**：A 一定先于 B 完成。
- **不同 stream 之间无顺序保证**，除非用 event 显式建立依赖。

这就是 `note/inference/01` 里 `wall` 和 `event` 两个量不一致的根源。

---

## 2. Synchronous vs Asynchronous H2D

| 方式 | 行为 | 影响 |
|---|---|---|
| synchronous copy | CPU 发起后等待 copy 完成才返回 | CPU 被阻塞，无法准备下一批数据 |
| `non_blocking=True` | copy 入队后 CPU 立即返回 | CPU 可以继续跑，H2D 与计算可重叠 |

```python
# 阻塞：CPU 卡在这里直到搬完
y.copy_(x)
# 非阻塞：入队即返回，CPU 继续
y.copy_(x, non_blocking=True)
```

**什么时候有区别**：只有当"后面还有 CPU 工作可以和 copy 重叠"时，non_blocking 才有收益。如果 CPU 马上要 `synchronize()` 或 `.item()`，非阻塞也没有意义。nsys 里能看到 H2D 是否和 compute 重叠（见 `note/profiling/01`）。

---

## 3. Pinned vs Pageable Memory

普通 CPU 内存是 pageable 的，操作系统可能随时把页换出去，DMA 引擎不能直接安全访问。因此 GPU 复制 pageable 内存时，要先做一次"页固定"（page pinning）或分页搬运，慢。

pinned memory（锁页内存，`pin_memory()` / `cudaHostAlloc`）保证页不会被换出，DMA 可以直接访问：

```text
pageable H2D：需要额外页处理，慢，且往往占 CPU
pinned  H2D：DMA 直接搬，快
```

**离散 GPU**（PCIe）上这个差距可达数倍；**统一内存平台（本机 Thor）** 差距小很多（实测约 83 vs 95-103 GB/s，见 `note/inference/02`）。用 pinned 的原则：**输入张量固定、反复搬运的场景才值得 pin**，且 pin 太多内存会耗尽可换页内存、拖慢整个系统。

---

## 4. CUDA Stream 与 Multi-Stream

stream 是"在 GPU 上串行执行的一队工作"。默认 stream 之外可以创建多条 stream，让**相互独立**的工作并行执行：

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    a = torch.mm(x1, w)   # 在 s1 上
with torch.cuda.stream(s2):
    b = torch.mm(x2, w)   # 在 s2 上，可与上面并行
```

**什么时候多 stream 有效**：

```text
有效：多个独立 kernel，且单个 kernel 吃不饱 GPU（有空闲 SM）
无效：kernel 已经把 GPU 占满（无资源可重叠）
无效：工作之间有数据依赖（必须串行）
```

本机实测（4 个独立 512×512 GEMM，每 stream 8 次）：single stream 2.11ms vs multi stream 1.98ms，收益约 6%——因为 GEMM 已经比较吃资源，重叠空间有限。真正的重叠收益出现在"kernel 很小 + CPU 有间隙"的 launch-bound 场景，而不是 compute-bound 的 GEMM。

### 多 stream 的代价

- 需要显式管理 event 依赖，否则数据竞争（两个 stream 同时读写同一块）。
- PyTorch caching allocator 会为跨 stream 的 tensor 加额外同步，抵消部分收益。
- 代码复杂度上升，先确认单 stream 已优化到瓶颈再说。

---

## 5. CUDA Event：测量与依赖

event 有两个用途：

1. **计时**：`start.record(); ...; end.record(); elapsed = start.elapsed_time(end)` 得到纯 GPU 时间。
2. **跨 stream 依赖**：`s2.wait_event(e)` 让 s2 等 s1 上的某个 event。

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
model(x)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)   # device 时间，排除 host 开销
```

这是 `src/common/measure.py::cuda_event_latency` 的实现基础，也是 benchmark 里唯一"合法"的 GPU 计时方式之一（另一种是 host 同步边界）。

---

## 6. CUDA Graph：把多次 launch 折叠成一次

**问题**：batch=1 的小模型，forward 里有几十上百个算子，每个算子一次 launch，每次 launch 有 ~2-8us 的 CPU 开销。当 GPU 执行每个 kernel 只要 1-5us 时，CPU 侧的 launch 开销就成了主导——GPU 大部分时间在等 CPU 喂下一个 kernel（launch-bound）。

**原理**：CUDA Graph 把一整串 kernel 的 launch 参数、内存地址、依赖关系**录制成一张图**，之后用一次 launch 重放整张图，把 N 次 launch 开销压缩成 ~1 次。

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):   # 录制：这期间的 op 不真正执行，只记录
    model(x)
g.replay()                  # 重放：一次 launch 跑完整张图
```

**实测（本机 Thor，64 个 tiny op 的链）**：

```text
normal launch：wall 647us  / event 639us
CUDA Graph   ：wall 128us  / event 113us   → 约 5x 下降
```

**读法**：wall 从 647 掉到 128，说明砍掉的是 CPU launch 开销（launch-bound 的实锤）；event 从 639 掉到 113，是因为重放时 GPU 能连续执行、消除了 kernel 之间的气泡。两者同降 = 这个 workload 同时被 launch 和气泡拖累。

### CUDA Graph 的限制（为什么不是银弹）

```text
静态性：图里的输入/输出地址是固定的（静态 buffer），shape 变了要重新录制
内存：图内不允许改变内存分配（cudaMalloc / 新的 allocator 块会失败）
动态 shape：VLM/LLM 变长输入需要 shape padding + 多张图
录制成本：首次录制有一次性开销
```

所以工业上的用法是：**固定 shape 的推理**（尤其 batch=1 机器人实时推理）用 graph；**动态 shape / 变长**场景用 graph 要配合 padding 或 shape 桶。

---

## 7. 常见故障与定位

| 现象 | 第一证据 | 根因 | 定位 |
|---|---|---|---|
| wall ≫ event | benchmark 两者差距大 | launch-bound | nsys 看 kernel 数量/间隙 |
| H2D 慢 | copy 占大比例 | pageable 未 pin | 换 pinned / non_blocking |
| multi-stream 无收益 | 时间几乎不变 | 无空闲 SM / 有依赖 | 确认 kernel 是否吃满 GPU |
| graph 录制失败 | `RuntimeError: ... during capture` | 图内动态分配 | 预分配静态 buffer |
| 跨 stream 结果错 | 数值不稳定 | 缺 event 依赖 | 加 `wait_event` |

---

## 8. 本模块闭环小结

```text
问题：CPU 交给 GPU 的环节慢、有气泡
      ↓
原理：异步入队 + stream 顺序 + 同步点产生气泡
      ↓
实验：pinned/non_blocking/stream（cuda_async）、normal vs graph（cuda_graph）
      ↓
证据：nsys timeline 看 launch gap、H2D 重叠、同步气泡
      ↓
优化：pinned + async H2D、fusion、CUDA Graph
```

下一模块：`note/profiling/01_nsys_inference_profiling.md`，建立"用 nsys 找慢 kernel / 找气泡"的标准 SOP。
