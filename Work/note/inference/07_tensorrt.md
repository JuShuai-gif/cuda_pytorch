# 07｜TensorRT：PyTorch → ONNX → Engine 的完整链路（C++）

## 本模块解决的问题

TensorRT 是 NVIDIA 的推理优化引擎，它做三件事：**算子融合（layer fusion）、精度选择（FP32/FP16/INT8）、针对特定 shape 的 kernel 自动调优（tactic selection）**。本章走通 PyTorch → ONNX → TensorRT engine 的完整链路，并回答：

```text
TensorRT 相比 torch eager / torch.compile 快在哪？
Builder / Network / Engine / ExecutionContext 各自是什么？
Optimization Profile（min/opt/max shape）是什么，为什么需要？
dynamic shape 为什么可能降低性能？
FP16 的精度损失和收益各是多少？
```

配套代码：`src/inference/tensorrt/`（C++ 全流程 + Python 导出）。

---

## 1. 为什么这条链路用 C++

回到边缘端实战：TensorRT 的生产 API 是 **C++**。Python 的 `tensorrt` 包只是 C++ API 的薄封装，而且——本机的关键约束——**TensorRT 的 pip wheel 不支持 Tegra/Jetson**：

```text
RuntimeError: TensorRT does not currently build wheels for Tegra systems
```

所以本机只有两条路：系统 Python 的 apt 版 TensorRT，或 C++。按「边缘端用 C++」的原则，本模块用 C++ 走完整链路，只有导出 ONNX 这一步用 Python（因为 torch 在 flashrt env）。

```text
flashrt (torch) --导出 ONNX--> model.onnx
                                │
                         C++ TensorRT（本模块）
                     ONNX parser → Builder → Engine → ExecutionContext
```

---

## 2. 完整流程与四个核心对象

### 流程

```text
1. export_onnx.py   torch 模型 → model.onnx（dynamic batch/seq）+ 参考 I/O
2. build_engine.cpp ONNX → TensorRT engine（序列化到 .engine 文件）
3. run_engine.cpp   反序列化 engine → 推理 → 正确性对比 → benchmark
```

### 四个核心对象

| 对象 | 作用 | 生命周期 |
|---|---|---|
| `INetworkDefinition` | 模型的计算图（从 ONNX 解析而来） | 构建期 |
| `IBuilder` + `IBuilderConfig` | 把 network 编译成 engine，选 tactic、精度 | 构建期 |
| `ICudaEngine` | 编译好的、针对特定 shape/精度的可执行引擎 | 可序列化保存 |
| `IExecutionContext` | engine 的一次执行句柄，绑定输入输出 buffer | 推理期 |

关键认知：**engine 是"编译产物"**，构建期（tactic selection）很慢（本机 FP16 构建 35s），但推理期很快。所以工业上**离线构建 engine、线上只做反序列化 + 推理**。

---

## 3. Optimization Profile 与 dynamic shape

ONNX 导出时声明 batch 和 seq 是动态维度。TensorRT 构建时需要知道这些维度的**可能范围**，这就是 Optimization Profile：

```cpp
profile->setDimensions(input, kMIN, Dims3{1,  1,  1024});   // min shape
profile->setDimensions(input, kOPT, Dims3{8,  16, 1024});   // opt shape（编译优化的目标）
profile->setDimensions(input, kMAX, Dims3{32, 64, 1024});   // max shape
```

- **min/max**：定义了 engine 能接受的 shape 范围。超出会报错。
- **opt**：TensorRT 针对这个 shape 做最优化的 tactic 选择。

### 为什么 dynamic shape 可能降低性能

TensorRT 的核心优化（tile 大小、kernel 选择、内存布局）是**针对具体 shape 调优**的。static shape 的 engine 可以针对唯一 shape 极致优化；dynamic shape 的 engine 必须对范围内所有 shape 都"过得去"，往往在 opt shape 上最优、其他 shape 次优。所以：

```text
static shape engine  针对唯一 shape，最极致，但 shape 一变就要重建
dynamic shape engine 覆盖范围，灵活，但单个 shape 可能不如 static
```

工业权衡：固定 shape 的推理（机器人 batch=1）用 static engine；变长输入（LLM/VLM）用 dynamic engine + 多个 shape 桶。

### 本机实测：dynamic shape 确实覆盖了范围

同一个 FP16 engine（opt=8×16），batch 从 1 到 32 都能跑：

```text
batch=1  ：latency 0.141ms  throughput 5.4k samples/s
batch=8  ：latency 0.148ms  throughput 46.6k samples/s
batch=32 ：latency 0.230ms  throughput 66.4k samples/s
```

batch 1→32，吞吐提升 12x（更大的 GEMM 吃满 Tensor Core），但单 batch 延迟也上升（0.141→0.230ms）——这是 batch 的经典 tradeoff（见 `note/inference/03`）。

---

## 4. 实测对比：torch vs TensorRT

残差 MLP（4 层，hidden=1024），batch=1，seq=16，本机 Thor/sm_110：

| runtime       | latency (mean) | 相对 eager | engine size |
|---------------|----------------|------------|-------------|
| torch eager   | 0.44 ms        | 1.0x       | -           |
| torch.compile | 0.51 ms        | 0.87x      | -           |
| TRT FP32      | 0.11 ms        | **4.0x**   | 33.9 MB     |
| TRT FP16      | 0.09 ms        | **4.8x**   | 17.1 MB     |

（注：TRT 多次运行 latency 在 0.09-0.14ms 波动，取决于时钟/热状态；量级和趋势稳定。）

### 读法

1. **TensorRT 比 eager 快 4x**：这不是"玄学优化"，而是三层叠加——layer fusion（把 LayerNorm+GELU+Linear 融合成少数 kernel）、tactic selection（针对 1×16×1024 shape 选最优 kernel）、低精度（FP16 用 Tensor Core）。

2. **torch.compile 在这个小模型上反而慢 13%**：torch.compile 的收益需要足够大的图来摊薄编译/调度开销，batch=1 的小模型上 Inductor 的 overhead 抵消了融合收益。这不是 torch.compile "没用"，而是它在这个 scale 下不占优。

3. **FP16 比 FP32 快 1.2x、engine 小一半**：Tensor Core 的 fp16 吞吐是 fp32 的数倍（memory-bound 的小模型上收益被带宽掩盖一部分），且权重减半。

### 精度（正确性）

```text
TRT FP32 max diff 6.7e-4   （几乎无损）
TRT FP16 max diff 4.1e-3   （fp16 尾数损失，可接受）
```

FP16 的精度损失在 `1e-3` 量级，对推理足够；INT8 才是真正需要 calibration 的精度权衡（Stage 8）。

---

## 5. 构建期 vs 推理期

```text
build 时间：FP32 6.0s，FP16 35.2s
推理时间：  ~0.1ms（单次）
```

FP16 构建比 FP32 慢 6 倍，因为 fp16 的 tactic 组合更多（是否用 Tensor Core、是否转 tf32 等）。这印证了「engine 是编译产物」：**构建慢、推理快，所以要离线构建、线上复用**。这也是为什么生产系统有"engine 缓存"和"模型 registry"（Stage 17/24）。

---

## 6. 与前面各 Stage 的关系

TensorRT 是把 Stage 4-6 的优化**自动化**了：

```text
Stage 4 Triton   手写 fused kernel
Stage 6 Fusion   手写 fusion，理解何时赢
Stage 7 TensorRT 自动 layer fusion + 自动 tactic 选择 + 自动精度管理
```

TensorRT 内部就是 CUTLASS/cuBLAS 级别的 fused kernel 库 + 一个针对具体 shape 的自动调优器。理解了 Stage 4-6，就能理解 TensorRT 为什么快、以及它的边界在哪（它覆盖不到的算子，就是 Plugin 出场的地方，Stage 7 的 Plugin 篇）。

---

## 7. 本机 TensorRT 的关键事实

```text
版本：TensorRT 10.13.3.9（JetPack apt）
约束：pip wheel 不支持 Tegra → 用 C++ API 或系统 Python
API：10.x 用 setMemoryPoolLimit（替代 setMaxWorkspaceSize）、
     setTensorAddress + enqueueV3（替代 executeV2/bindings）
坑：  torch.compile 的 Inductor 也走 Triton → 同样要设
     TRITON_PTXAS_BLACKWELL_PATH（见 note/kernel/05）
```

---

## 8. 本模块闭环小结

```text
问题：如何把 PyTorch 模型变成 GPU 上最快、可部署的推理产物
      ↓
流程：PyTorch → ONNX → Builder → Engine → ExecutionContext
      ↓
机制：layer fusion + tactic selection + 精度选择
      ↓
实测：TRT 比 eager 快 4x，FP16 再快 1.2x 且 engine 减半
      ↓
动态：Optimization Profile 覆盖 shape 范围，但静态 shape 更极致
      ↓
下一步：Stage 7 剩余部分 —— TensorRT Plugin（自定义算子接入），
      → Stage 8 量化（FP8/INT8/INT4 + calibration + SmoothQuant/AWQ/GPTQ）
```

要继续就说「继续」。
