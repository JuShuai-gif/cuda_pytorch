# Profiler 性能分析源码解析

> 源码路径: `/home/ghr/code/pytorch/torch/autograd/profiler.py` (1808 行), `torch/profiler/` (新版)
> C++ 后端: `torch/csrc/autograd/profiler_kineto.cpp` — Kineto 集成, `torch/csrc/profiler/` — 通用 profiler 基础设施

## 0. 一句话总览

PyTorch Profiler 基于 CUPTI（CUDA Profiling Tools Interface）采集 GPU kernel 级别的执行信息，通过 chrome://tracing 可视化。新版 (`torch.profiler.profile`) 使用 Kineto 库，支持 trace、stack trace、memory profiling、FLOPs 估算。

---

## 一、新旧 Profiler 对比

| 特性 | Legacy `torch.autograd.profiler` | New `torch.profiler.profile` |
|------|------|------|
| Trace 输出 | 简单文本表 | `chrome://tracing` JSON |
| GPU kernel 级 | 支持 | 支持 + CUPTI 详细 trace |
| Memory profiling | 无 | 支持（alloc/dealloc trace） |
| Stack trace | 无 | 支持（显示 Python/C++ 调用栈） |
| FLOPs | 无 | 通过 `with_stack` + shapes 估算 |
| 可视化 | 终端打印 | TensorBoard 插件 |

---

## 二、`torch.profiler.profile` 核心 API

### 2.1 基本用法

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for _ in range(10):
        output = model(input)
        prof.step()

# 输出: ./log/ 下的 chrome trace JSON, 用 chrome://tracing 打开
```

### 2.2 `schedule` 参数

```
wait=2, warmup=2, active=4, repeat=2

周期 1: [wait 2次] [warmup 2次] [active 4次（采集）]
周期 2: [wait 2次] [warmup 2次] [active 4次（采集）]
结束

总计: 2 * (2+2+4) = 16 次 prof.step() 调用
采集部分: 2 * 4 = 8 次
```

**为什么需要 wait/warmup**: 前几次 iter 包含 CUDA 初始化、缓存预热等，数据不具代表性。warmup 后再采集。

### 2.3 `record_shapes` 和 `with_stack`

- `record_shapes=True`: 记录每个 op 的输入输出 shape → 可用于估算 FLOPs
- `with_stack=True`: 记录 Python/C++ 调用栈 → 知道每个 op 是谁调用的
- `profile_memory=True`: 记录 CUDA 内存分配/释放事件 → 分析显存使用

---

## 三、Profiler 的 C++ 基础设施

### 3.1 Dispatch Key 拦截

Profiler 通过在 dispatch 层插入 `Python` dispatch key 来拦截所有算子调用。当 profiler 活跃时，每个算子的执行会经过 profiler 的 callback。

### 3.2 CUPTI 集成 (`torch/csrc/profiler/`)

CUPTI (CUDA Profiling Tools Interface) 是 NVIDIA 提供的底层 API，可以：
- 拦截 GPU kernel launch（`cuptiActivityRegisterCallbacks`）
- 记录 kernel 开始/结束时间（纳秒级）
- 记录显存拷贝、同步等 CUDA runtime 事件

PyTorch 的 `profiler_kineto.cpp` 将 CUPTI 事件与 Python 的 profiling events 关联。

### 3.3 Event 数据结构

```cpp
// profiler/events.h
struct ExtraFields {
    std::vector<std::vector<int64_t>> inputs_;   // 输入 shapes
    std::vector<std::vector<int64_t>> outputs_;  // 输出 shapes
    uint64_t flops_;       // 估算的 FLOPs
    std::string kernel_name_;  // CUDA kernel 名 (如 "volta_sgemm_128x128_nn")
};
```

---

## 四、常见性能分析模式

### 4.1 识别瓶颈

用 chrome://tracing 打开 trace JSON：

```
GPU Stream:
  [===matmul===]         [==conv==]    [=add=]
  <- kernel gap ->       <- gap ->

CPU:
  [op launch] [op launch] [op launch]

分析:
  - GPU kernel 之间有大的 gap → CPU launch 跟不上（CPU-bound）
  - GPU kernel 持续满载但 CPU 空闲 → GPU 是瓶颈
  - kernel 之间有小 gap → kernel launch overhead（考虑 CUDA Graph）
```

### 4.2 分析显存

```
Memory timeline:
  [malloc(1GB)  ] [free(0.5GB)] [malloc(2GB)]

分析:
  - 查找 peak memory 位置
  - 是否有 alloc-free 对（可能的内存泄漏）
  - peak 是否接近 GPU 总内存
```

### 4.3 分析 FLOPs

```
op                  FLOPs    % total
aten::linear       2048M      45%
aten::matmul       1536M      34%
aten::add           512M      11%
aten::softmax       256M       6%
...
```

找到 FLOPs 占比最高的 op，考虑优化（如 kernel fusion, Flash Attention 替代 standard attention）。

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| Legacy profiler | `torch/autograd/profiler.py` | — |
| `torch.profiler.profile` | `torch/profiler/profiler.py` | — |
| `schedule` | `torch/profiler/profiler.py` | — |
| `tensorboard_trace_handler` | `torch/profiler/profiler.py` | — |
| Kineto 集成 (C++) | `torch/csrc/autograd/profiler_kineto.cpp` | — |
| Event 数据结构 | `torch/csrc/profiler/events.h` | — |
| CUPTI 回调 | `torch/csrc/profiler/cupti_strings.cpp` | — |
| Python/TensorBoard 工具 | `torch/utils/tensorboard/` | — |

---

## 六、可借鉴的工程技巧

1. **采样而非全量采集 (schedule)**: 用 `wait/warmup/active` 控制采集周期，避免 trace 文件过大和性能影响。

2. **分层 profiling**: CUPTI → GPU kernel, Python trace → operator call, stack trace → 调用来源。三层信息交叉关联。

3. **Shape + FLOPs 估算**: 不实际计算 FLOPs，而是从 op type + input/output shapes 表中查表估算。

4. **Event 关联**: 将 CUPTI GPU events 和 Python CPU events 通过 correlation ID 关联，实现端到端的时间线。

5. **Chrome Trace 格式**: 使用业界标准的 JSON trace 格式，可与 chrome://tracing / Perfetto / TensorBoard 通用。
