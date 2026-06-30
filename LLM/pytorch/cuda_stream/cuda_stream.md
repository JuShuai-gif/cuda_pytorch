# CUDA Stream, Event & CUDA Graph 源码分析

> Python 端: `/home/ghr/code/pytorch/torch/cuda/streams.py` (271 行), `/home/ghr/code/pytorch/torch/cuda/graphs.py` (543 行)
> C++ Stream: `/home/ghr/code/pytorch/torch/csrc/cuda/Stream.cpp`, `c10/cuda/CUDAStream.h`
> C++ Event: `/home/ghr/code/pytorch/torch/csrc/cuda/Event.cpp`
> C++ Graph: `/home/ghr/code/pytorch/aten/src/ATen/cuda/CUDAGraph.h` + `.cpp`

## 0. 一句话总览

CUDA Stream = 提交 GPU 工作的**命令队列**（每个 stream 内的工作串行执行）；Event = 跨 stream 的**同步点**；CUDA Graph = 把一系列 kernel launch 录制成**可重放的静态执行图**，消除 CPU launch overhead。

---

## 一、Stream 的创建与池化

### 1.1 Stream 池 (`c10/cuda/CUDAStream.h:10-50`)

PyTorch **不是每次创建新 stream**，而是维护两个优先级池（低优先级 + 高优先级），每个设备 32 个 stream，轮询复用：

```cpp
// CUDAStream.h:33
constexpr int kStreamsPerPoolBits = 5;       // 2^5 = 32
constexpr int kStreamsPerPool = 1 << 5;       // 32
```

调用链:

```
torch.cuda.Stream()           streams.py:37
  -> THCPStream_pynew()       Stream.cpp:16
    -> at::cuda::getStreamFromPool(priority, device)
      -> CUDAStreamPool 取出池中下一个 stream
        -> cuStreamCreate()
```

### 1.2 `current_stream()` (`__init__.py:1253-1268`)

```python
def current_stream(device=None):
    streamdata = torch._C._cuda_getCurrentStream(_get_device_index(device))
    return Stream(stream_id=streamdata[0], device_index=streamdata[1], ...)
```

`torch._C._cuda_getCurrentStream` → `THCPModule_getCurrentStream_wrap` (`Module.cpp:171`) → `at::cuda::getCurrentCUDAStream()`。

### 1.3 `synchronize()` (`__init__.py:1227-1237`)

```python
def synchronize(device=None):
    _lazy_init()
    with torch.cuda.device(device):
        return torch._C._cuda_synchronize()   # -> cudaDeviceSynchronize()
```

全局同步，等待设备上所有 stream 的所有工作完成。

---

## 二、Event 的创建与同步

### 2.1 Event 创建 (`streams.py:159-271`, `Event.cpp:18`)

```python
event = torch.cuda.Event(
    enable_timing=True,    # 是否计时
    blocking=False,        # 同步时是否 blocking wait
    interprocess=False,    # 是否跨进程共享
)
```

C++ 端: `THCPEvent_pynew` (`Event.cpp:18`) 构造 `at::cuda::CUDAEvent(flags)`。

### 2.2 record + wait 工作流

```python
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    x = torch.randn(1024, device="cuda") * 2

event = stream.record_event()  # 在 stream 中记录一个时间点

# 主 stream 等待 event 完成后才能继续
torch.cuda.current_stream().wait_event(event)
```

C++ 对应:
- `stream.record_event()` → `cudaEventRecord(event_ptr, stream_ptr)`
- `stream.wait_event(ev)` → `cudaStreamWaitEvent(stream_ptr, event_ptr, 0)`

### 2.3 elapsed_time (`streams.py:234`)

```python
t_ms = start_event.elapsed_time(end_event)
```

用于测量两个 CUDA event 之间的 GPU 时间（精度 ~0.5μs）。

---

## 三、CUDA Graph — 消除 CPU launch overhead

### 3.1 原理

CUDA Graph 允许你将一系列 GPU 操作（kernel launches、memcpy 等）**录制**成一个图，然后**一次 launch** 重放整个图。消除了逐 kernel 的 CPU→GPU launch 延迟（每发 kernel ~5-10μs）。

### 3.2 Python API (`graphs.py:78-543`)

```python
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):  # 开始录制
    y = x * 2 + 1
    z = y.relu()

# 录制结束，graph 已完成

graph.replay()  # 一次 launch 重放整个图
```

### 3.3 核心实现 (`CUDAGraph.cpp:101-322`)

**录制 (`capture_begin`)**:
1. 设置私有内存池（或复用已有池）
2. `cudaStreamBeginCapture(stream, capture_mode)` — CUDA API 开始录制
3. 保存 RNG 状态（确保 replay 时数值一致）
4. 向全局 map 注册此 graph 的 capture ID

**结束 (`capture_end`)**:
1. `cudaStreamEndCapture(stream, &graph_)` — 获得 `cudaGraph_t`
2. 调用 `cudaGraphInstantiateWithFlags()` 创建 `cudaGraphExec_t`（可执行实例）
3. 销毁模板 graph（`cudaGraphDestroy`，可选保留）

**重放 (`replay`)**:
1. 恢复 RNG 状态
2. `cudaGraphLaunch(graph_exec_, currentStream)` — 单次 launch

### 3.4 内存池

CUDA Graph 要求输入 tensor 的地址**在 replay 之间不变**（因为 graph 记录了固定地址）。PyTorch 通过**私有内存池**解决：录制前切换到专属 allocator，replay 时复用同一块内存。

```python
# 两个 graph 可以共享内存池以减少碎片
graph2 = torch.cuda.CUDAGraph()
graph2.capture_begin(pool=graph.pool())  # 共享内存池
```

### 3.5 限制

- CUDA Graph 中不能出现 CPU 同步操作（`.item()`, `.to("cpu")`, `synchronize()` 等）
- 不能动态分配内存（录制时所有 tensor shape 必须固定）
- 不能调用依赖动态 shape 或 data-dependent control flow 的操作

---

## 四、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `Stream` 类 | `torch/cuda/streams.py` | 17 |
| `ExternalStream` | `torch/cuda/streams.py` | 136 |
| `Event` 类 | `torch/cuda/streams.py` | 159 |
| `CUDAGraph` 类 | `torch/cuda/graphs.py` | 78 |
| `graph()` 上下文管理器 | `torch/cuda/graphs.py` | 420 |
| `synchronize()` | `torch/cuda/__init__.py` | 1227 |
| `current_stream()` | `torch/cuda/__init__.py` | 1253 |
| Stream 池常量 | `c10/cuda/CUDAStream.h` | 33 |
| `CUDAStream` 类 | `c10/cuda/CUDAStream.h` | 60 |
| `capture_begin` (C++) | `aten/src/ATen/cuda/CUDAGraph.cpp` | 101 |
| `capture_end` (C++) | `aten/src/ATen/cuda/CUDAGraph.cpp` | 182 |
| `replay` (C++) | `aten/src/ATen/cuda/CUDAGraph.cpp` | 268 |
| `instantiate` (C++) | `aten/src/ATen/cuda/CUDAGraph.cpp` | 238 |
| Python 绑定 (Graph) | `torch/csrc/cuda/Graph.cpp` | 21 |
| Python 绑定 (Stream) | `torch/csrc/cuda/Stream.cpp` | 176 |
| Python 绑定 (Event) | `torch/csrc/cuda/Event.cpp` | 223 |
| `is_current_stream_capturing` | `torch/cuda/graphs.py` | 57 |

---

## 五、可借鉴的工程技巧

1. **对象池化 (Stream pool)**: 32 个 stream 轮询复用，避免创建/销毁开销。类比：数据库连接池、线程池。

2. **录制-重放 (CUDA Graph)**: 把重复的指令序列录制成模板，消除每次的解析/调度开销。类比：JIT 编译、正则表达式编译。

3. **私有内存池**: CUDA Graph 要求地址稳定，通过专属 allocator 隔离避免与其他操作冲突。

4. **RNG 状态保存/恢复**: `capture_begin` 保存生成器状态，`replay` 恢复，确保随机数一致。类比：checkpoint/resume 中的 RNG 状态序列化。

5. **Event 做精确 GPU 计时**: `elapsed_time` 测量 GPU 执行时间（不含 CPU launch overhead），用于 kernel 性能分析。

---

## 六、实战常见坑点

### 1. 主 stream 阻塞了 CUDAGraph 的 replay
**现象**: CUDAGraph replay 的 kernel 时间很长, 不知道为什么在等什么。
**原因**: CUDA Graph 被录制在非默认 stream 上, replay 时也必须在**同一 stream** 上。如果 CUDAGraph 内部有 `wait_event` 等在默认 stream 上, 而默认 stream 有未完成的工作 → 死等。
**解决**:
```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    with torch.cuda.graph(graph):
        ...  # 录制在 s 上
# replay 也在 s 上
with torch.cuda.stream(s):
    graph.replay()
s.synchronize()  # 等待 s 完成
```

### 2. CUDAGraph 内 tensor 地址被复用导致崩溃
**现象**: replay 时出现 "CUDA error: illegal memory access"。
**原因**: 图的输入 tensor 在两次 replay 之间被释放+重新分配, 新 tensor 复用了旧的地址 → graph 读到旧地址的垃圾数据。
**解决**:
```python
# 固定输入 tensor 的地址（使用同一内存池）
x = torch.zeros(B, C, H, W, device="cuda")
for step in range(100):
    x.copy_(data[step])  # in-place 替换内容，不改变地址
    graph.replay()
```

### 3. 多个 stream 同时写同一个 tensor
**现象**: 结果随机出错——有时对有时不对。
**原因**: 两个 CUDA stream 同时写同一个 tensor 的不同位置 → 数据竞争。
**排查**:
```python
# CUDA_LAUNCH_BLOCKING=1 让所有 kernel 同步执行 → 如果问题消失 → 是并发 bug
```
**解决**: 用 event 保证跨 stream 的顺序; 不同 stream 操作不同 tensor。

### 4. `torch.cuda.synchronize()` 过度使用
**现象**: 代码跑得慢, profiler 显示大量时间在 CPU 等待。
**原因**: `synchronize()` 是全局同步 → 等待所有 stream 的所有工作 → GPU 利用率骤降。
**最佳实践**: 只在 profiling / checkpoint / debug 时用; 热路径用 stream-level sync (`stream.synchronize()`) 或 event-based sync。

### 5. CUDA Graph capture 失败 — "capturing would exceed max memory"
**现象**: 录制大模型时报 CUDA 内存不足。
**原因**: capture 期间分配的内存被锁定到 CUDA Graph 的私有池 → 不可释放。
**解决**:
```python
# 方案 A: 共享内存池减少碎片
pool = torch.cuda.graph_pool_handle()
graph1 = torch.cuda.CUDAGraph()
graph2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph1, pool=pool):
    ...
with torch.cuda.graph(graph2, pool=pool):
    ...
# 方案 B: 减小录制范围内 op 的数量
```

