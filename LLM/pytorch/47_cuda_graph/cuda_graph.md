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