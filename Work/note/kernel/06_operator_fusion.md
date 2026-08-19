# 06｜Operator Fusion：什么时候融合能赢，什么时候不能

## 本模块解决的问题

现代推理系统（TensorRT、torch.compile、vLLM 的算子层）高度依赖 fusion。但「融合一定更快」是错的。本章用四个案例把 fusion 的收益和代价拆开，回答：

```text
fusion 到底省了什么？（kernel count、memory traffic）
为什么有的 case 快 7x，有的 case 反而慢 0.55x？
什么时候该融合，什么时候不该？
工业界正确的融合姿势是什么？
```

配套代码：`src/kernel/fusion/`（`fused_ops.py` 四个 fused Triton kernel + `benchmark.py`）。

---

## 1. fusion 的理论收益

一次 eager 执行 `residual + rmsnorm` 会物化一串中间张量：

```text
y = x + r            → kernel 1（读 x,r，写 y）
y2 = y.pow(2)        → kernel 2（读 y，写 y2）
m  = y2.mean(-1)     → kernel 3（读 y2，写 m）
s  = rsqrt(m + eps)  → kernel 4（读 m，写 s）
o1 = y * s           → kernel 5（读 y,s，写 o1）
o  = o1 * w          → kernel 6（读 o1,w，写 o）
```

每一步都是「global memory 读 → 算 → global memory 写」的往返。中间张量 y、y2、m、s、o1 都是**只为了传给下一步才存在**的。

融合后，这些中间值只活在 register/SRAM，global memory 只碰两次（读入 x,r,w，写出 o）：

```text
读 x,r,w → SRAM 里算完整个 rmsnorm → 写 o
```

所以 fusion 的两个收益是：

```text
kernel count   ↓  （N 个 kernel → 1 个）
memory traffic ↓  （N 次往返 → 1 次往返）
```

---

## 2. 实测：四个案例

统一 benchmark（fp16，CUDA-event device 时间，traffic 为分析估算）：

| case | kernels | traffic | latency | 结论 |
|---|---|---|---|---|
| bias_relu | 2→1 | 10.5→6.3MB | 55.0→99.6us | **慢 0.55x** |
| residual_rmsnorm | 7→1 | 83.9→25.2MB | 336→48.6us | **快 6.92x** |
| gemm_bias | 2→1 | 10.5→6.3MB | 59.1→92.2us | **慢 0.64x** |
| dequant_gemm | 4→1 | 17.8→5.2MB | 79.7→134.4us | **慢 0.59x** |

### 读法

**第一层：kernel count 和 traffic 全部下降。** 这是 fusion 的确定性收益，四个 case 无一例外（2→1、7→1、2→1、4→1；traffic 分别省 40%、70%、40%、71%）。

**第二层：latency 剧烈分化。** 只有 residual_rmsnorm 快了（6.92x），另外三个反而慢（0.55-0.64x）。为什么？

---

## 3. 为什么含 GEMM 的融合反而慢

bias_relu、gemm_bias、dequant_gemm 的共同点：**里面有一个矩阵乘**。

- unfused 的 `F.linear(x,w,b)`、`a@b` 走的是 **cuBLAS**，它有数十年的 auto-tuning、tile 选择、软件流水线，fp16 下 1024³ GEMM 约 78us。
- 我的 fused kernel 里的 `tl.dot` 是**手写 tiled matmul**（Stage 4 的默认配置），约 90-100us。

所以 fusion 省下的 memory traffic（比如 gemm_bias 省 4MB ≈ 几十 us 的带宽），**不足以弥补手写 GEMM 与 cuBLAS 之间 ~20us 的性能差距**。

```text
fusion 净收益 = 省的 traffic 收益 - (手写 kernel 与优化库的性能差距)
```

当融合的算子里有 GEMM，而你的 GEMM 打不过 cuBLAS，净收益就是负的。

### 对比：residual_rmsnorm 为什么赢

它不含 GEMM，全是 elementwise/reduction。eager 版本物化 5 个中间张量（83.9MB traffic），fused 只搬 25.2MB。**没有 cuBLAS 来抵消**，fusion 的 traffic 收益就完整兑现成 latency（6.92x）。

---

## 4. 工业界的正确融合姿势

结论不是「不要融合 GEMM」，而是「融合 GEMM 时，GEMM 部分必须达到优化库水平」。三条路：

### 路线 1：cuBLAS / CUTLASS 的 fused epilogue

cuBLAS 和 CUTLASS 本身就支持在 GEMM 的 epilogue 里融合 bias、ReLU、量化等。TensorRT 的 layer fusion 就是这么做的——**它融合的是 cuBLAS/CUTLASS 内核，不是手写 Triton**。

### 路线 2：torch.compile 自动融合

```python
compiled = torch.compile(lambda x, w, b: torch.relu(F.linear(x, w, b)))
```

torch.compile 会把 elementwise 算子融合，但 **GEMM 仍回落到 cuBLAS/CUTLASS**（Inductor 生成的代码里 gemm 用 CUTLASS 模板）。这就是「既融合了 elementwise、又没丢掉 GEMM 性能」的正确姿势。

### 路线 3：Triton 用于没有 GEMM 的算子

Triton 的主场是 **cuBLAS 覆盖不到的组合算子**：
- RMSNorm / LayerNorm / RoPE（无 GEMM）
- attention（flash attention，cuBLAS 不做）
- 量化 GEMM（cuBLAS 没有 int8 weight + fp16 act 的原生 kernel）

**规则**：标准 GEMM 交给 cuBLAS/CUTLASS；cuBLAS 覆盖不到的组合算子用 Triton fusion。

---

## 5. 量化 GEMM 的额外说明

dequant_gemm（0.59x 慢）还有一个更微妙的问题。unfused 是 `a @ (wq.float()*ws).to(fp16)`，它虽然物化了 dequant 后的 weight，但 GEMM 是 fp16 cuBLAS。

fused 的 `tl.dot` 在 SRAM 里 dequant，省了 weight 物化，但：
1. 手写 GEMM 不如 cuBLAS（同上）
2. int8 的 `tl.load` 每次 K tile 都要 dequant，重复了 K/BLOCK_K 次 dequant 计算

真正的工业级量化 GEMM（如 FP8/INT4 的 TensorRT 内核）会在 **tensor core 的 mma 指令层面**直接吃低精度输入，而不是先 dequant 到 fp16 再 dot。这留到 Stage 8 量化专题。

---

## 6. 判据总结

| 融合对象 | 能否用 Triton 手写 | 正确做法 |
|---|---|---|
| elementwise + reduction（residual+rmsnorm） | 能，收益大 | Triton fused |
| elementwise + GEMM（bias+relu, gemm+bias） | 能写，但 GEMM 打不过 | cuBLAS/CUTLASS epilogue 或 torch.compile |
| dequant + GEMM | 能写，但低精度 mma 才最优 | TensorRT/专用量化内核 |
| attention（QK^T + softmax + PV） | 能，且 cuBLAS 不做 | Triton flash attention |

---

## 7. 本模块闭环小结

```text
问题：现代推理系统为什么依赖 fusion，fusion 一定快吗
      ↓
原理：fusion 省 kernel count + memory traffic，但净收益要扣除手写 GEMM 的性能差距
      ↓
实测：4 个 case，kernel/traffic 全降，latency 只有无 GEMM 的 residual_rmsnorm 赢
      ↓
结论：含 GEMM 的融合必须走 cuBLAS/CUTLASS epilogue 或 torch.compile
      ↓
下一步：Stage 7 TensorRT（工业级 layer fusion 的完整实现）
```

---

## 8. Stage 4-6 收尾：Triton + Fusion 能力闭环

至此 GPU 算子层的能力链完整了：

```text
PyTorch baseline（inference/）
   ↓
手写 CUDA C++（cuda_core/）—— 最高控制力，验证访存/occupancy/stream/graph
   ↓
Triton（triton/）—— 快速写 fused 算子，cuBLAS 覆盖不到的主场
   ↓
Fusion（fusion/）—— 理解什么时候融合赢、什么时候交给 cuBLAS
   ↓
TensorRT（下一 Stage）—— 工业级自动 layer fusion + 量化 + 动态 shape
```

下一模块进入 **Stage 7 TensorRT**：PyTorch → ONNX → TensorRT 完整流程，Benchmark PyTorch Eager / torch.compile / ONNX Runtime / TensorRT FP32/FP16/INT8，比较 latency、throughput、显存、engine size、accuracy。要继续就说「继续」。
