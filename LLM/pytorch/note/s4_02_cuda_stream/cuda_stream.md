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



---

# [合并自 47_cuda_graph]

# CUDA Graph 源码分析

> Python 端: `torch/cuda/graphs.py` (543 行)
> C++ 端: `aten/src/ATen/cuda/CUDAGraph.h` + `CUDAGraph.cpp`
> Python 绑定: `torch/csrc/cuda/Graph.cpp`

## 0. 一句话

CUDA Graph 把一系列 GPU kernel launch **录制**成一个静态图，**一次 launch 重放**整个图，消除 CPU→GPU 逐 kernel launch 开销（每发 ~5-10μs）。

## 1. 为什么需要

PyTorch eager mode 每执行一个 op 都要：
1. Python 解释器执行 op
2. ATen dispatch → 选 kernel
3. CUDA runtime launch kernel (~5-10μs)

对于小 kernel（计算时间 < launch 时间），launch overhead 占大头。CUDA Graph 录制后重放时，一次 `cudaGraphLaunch()` 替代 N 次 `cuLaunchKernel()`。

## 2. Python API

```python
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):          # capture_begin → capture_end
    y = x * 2 + 1
    z = y.relu()

graph.replay()                          # 一次 launch 重放全部 op
```

### 2.1 `torch.cuda.graph()` 上下文管理器 (`graphs.py:420-498`)

```python
class graph:
    def __init__(self, ...):
        self.graph = graph

    def __enter__(self):
        self.graph.capture_begin(pool=self.pool, ...)
        return self.graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.graph.capture_end()
        return False
```

### 2.2 `capture_begin` (`graphs.py:124-256`)

```python
def capture_begin(self, pool=None, capture_mode="global"):
    # 1. 如果没指定 pool，创建私有内存池
    # 2. cudaStreamBeginCapture(stream, capture_mode)
    # 3. 保存 RNG 状态
    # 4. 注册 capture_id → stream 映射
```

**内存池逻辑** (`graphs.py:173-219`):
```python
if pool is None:
    self.pool_created = True
    self.pool = torch.cuda.caching_allocator_alloc()  # 私有池
else:
    self.pool_created = False
    self.pool = pool  # 复用已有池
```

### 2.3 `capture_end` (`graphs.py:272-290`)

```python
def capture_end(self):
    # 1. cudaStreamEndCapture(stream, &graph_)
    # 2. cudaGraphInstantiateWithFlags()  → cudaGraphExec_t
    # 3. 销毁模板 graph
```

### 2.4 `replay` (`graphs.py:310-330`)

```python
def replay(self):
    # 1. 恢复 RNG 状态
    # 2. cudaGraphLaunch(graph_exec_, current_stream)
```

### 2.5 `reset` (`graphs.py:340-360`)

```python
def reset(self):
    # 1. cudaGraphExecDestroy(graph_exec_)
    # 2. 释放私有内存池（如果自己创建的）
    # 3. 重置状态
```

## 3. C++ 实现 (`CUDAGraph.cpp`)

### 3.1 核心数据结构 (`CUDAGraph.h:35-65`)

```cpp
struct CUDAGraph {
    cudaGraph_t graph_;          // 模板图
    cudaGraphExec_t graph_exec_; // 可执行实例
    cudaStream_t stream_;        // 录制时的 stream
    int device_;                 // GPU 设备
    bool has_graph_exec_;        // 是否已实例化
    bool is_capturing_;          // 是否在录制中
    at::Generator generator_;    // RNG 状态
    // ... memory pool handle ...
};
```

### 3.2 `capture_begin` (`CUDAGraph.cpp:101-180`)

```cpp
void CUDAGraph::capture_begin(...) {
    // 1. 设置 CUDA allocator 使用私有内存池
    // 2. cudaStreamBeginCapture(stream, capture_mode)
    //     - capture_mode: global / thread_local / relaxed
    // 3. 保存当前 RNG 状态（用于 replay 时恢复）
    // 4. 注册全局 capture ID
}
```

### 3.3 `capture_end` (`CUDAGraph.cpp:182-240`)

```cpp
void CUDAGraph::capture_end() {
    // 1. cudaStreamEndCapture(stream, &graph_)  // 获得 cudaGraph_t
    // 2. cudaGraphInstantiateWithFlags(&graph_exec_, graph_, flags, ...)
    //    flags: 可重放、可更新
    // 3. has_graph_exec_ = true
}
```

### 3.4 `replay` (`CUDAGraph.cpp:268-300`)

```cpp
void CUDAGraph::replay() {
    // 1. 恢复 RNG 状态
    // 2. cudaGraphLaunch(graph_exec_, getCurrentCUDAStream())
    // 3. 如果有 dependent graph，依次 replay（图间依赖）
}
```

## 4. 关键设计

### 4.1 私有内存池

CUDA Graph 录制时，所有 tensor 分配都来自私有内存池，保证 replay 时 tensor 地址不变。

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    y = x * 2
with torch.cuda.graph(g2, pool=g1.pool()):  # 共享池
    z = x * 3
```

注意: 不能同时 capture 两个 graph（全局只有一个 capture 上下文）。

### 4.2 RNG 状态保存/恢复

录制时保存 Philox RNG 状态，replay 时恢复，保证重放结果的数值一致性。

### 4.3 `graph_pool_handle()` (`graphs.py:530-540`)

一个方便的工厂函数，创建一个空的内存池 handle（不绑定 graph），让多个 graph 共享同一池。

## 5. 源码位置速查

| 概念 | 文件 | 行号 |
|------|------|------|
| `CUDAGraph` 类 | `torch/cuda/graphs.py` | 78 |
| `capture_begin` | `torch/cuda/graphs.py` | 173 |
| `capture_end` | `torch/cuda/graphs.py` | 272 |
| `replay` | `torch/cuda/graphs.py` | 310 |
| `graph()` 上下文 | `torch/cuda/graphs.py` | 420 |
| `graph_pool_handle` | `torch/cuda/graphs.py` | 530 |
| `capture_begin` (C++) | `aten/.../CUDAGraph.cpp` | 101 |
| `capture_end` (C++) | `aten/.../CUDAGraph.cpp` | 182 |
| `replay` (C++) | `aten/.../CUDAGraph.cpp` | 268 |
| Python 绑定 | `torch/csrc/cuda/Graph.cpp` | 21 |

## 6. 可借鉴的工程技巧

1. **录制-重放**: 把重复指令序列录制成模板，消除每次的解析/调度开销 → JIT 编译、正则表达式编译
2. **私有内存池**: 地址稳定，避免与其他操作冲突 → 任何需要内存地址稳定的场景
3. **RNG 状态恢复**: save/restore 确保数值一致性 → checkpoint 序列化