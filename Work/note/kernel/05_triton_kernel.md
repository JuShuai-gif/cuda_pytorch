# 05｜Triton Kernel：从 Vector Add 到 GEMM 的算子阶梯

## 本模块解决的问题

手写 CUDA C++（Stage 2 的 `cuda_core`）能拿到最高控制力，但每个 kernel 都要处理 index 计算、mask、block/warp 划分、shared memory、寄存器权衡，开发和调试成本极高。Triton 用「块级编程模型」把这份繁琐交给编译器，同时保留比 PyTorch eager 更高的算子融合自由度。

本章回答：

```text
Triton 相比手写 CUDA 少了什么、多了什么？
program_id / tl.arange / mask / tl.load / tl.store 各是什么？
BLOCK_SIZE / num_warps / num_stages 怎么影响性能？
为什么 rmsnorm 快 2x，gemm 却打不过 cuBLAS？
```

配套代码：`src/kernel/triton/`（`operators/` 六个算子 + `benchmark.py` + `sweep.py`）。

---

## 1. 一个关键环境坑（本机 sm_110）

Triton 3.6 自带的 `ptxas-blackwell` 是 CUDA 12.9 编译的，不认识本机 Thor 的 `sm_110a`，会直接报：

```text
ptxas-blackwell fatal: Value 'sm_110a' is not defined for option 'gpu-name'
```

解法：把 ptxas 指向系统 CUDA 13 的 ptxas（支持 sm_110a）：

```bash
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas
```

本包在 `kernel/triton/__init__.py` 里 `os.environ.setdefault` 了它。**这是边缘端（新架构 SoC）部署的典型问题**：工具链版本跟不上硬件架构，需要显式指定编译器。

---

## 2. Triton 编程模型（五分钟版）

一个 kernel 里只有几个概念：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)              # 当前 block 的 index（grid 的第几块）
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # 这个 block 负责的元素下标
    mask = offs < n                     # 尾部边界：n 不是 BLOCK 整数倍时
    x = tl.load(x_ptr + offs, mask=mask)       # 从 global 读（编译器决定放 shared/register）
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask) # 写回
```

对照手写 CUDA：

| 手写 CUDA | Triton | 谁负责 |
|---|---|---|
| `blockIdx.x * blockDim.x + threadIdx.x` | `pid * BLOCK + tl.arange(0, BLOCK)` | Triton 自动映射到 thread |
| 手动 shared memory + `__syncthreads` | `tl.load`/`tl.dot` 的 tile | 编译器自动分配 SRAM |
| 手动 coalescing / bank 规避 | 同上 | 编译器尽力优化 |
| `__launch_bounds__`、register 权衡 | `num_warps`/`BLOCK` 参数 | 编译器调 |

Triton 把「block 级」作为编程单位，thread 级细节交给编译器——这就是它比手写 CUDA 快写、比 PyTorch 灵活的原因。

---

## 3. 算子阶梯与实测结果

统一 benchmark（`benchmark.py`，CUDA-event device 时间，warmup 20 + 100 次）：

### fp16（真实 LLM 场景，tensor core 可用）

| 算子 | Triton | PyTorch | 结论 |
|---|---|---|---|
| vector_add | 225us | 229us | 持平（memory-bound） |
| reduction_sum | 135us | 131us | 持平（fp32 累加器） |
| softmax | 259us | 369us | **Triton 快 1.4x** |
| layernorm | 160us | 209us | **Triton 快 1.3x** |
| rmsnorm | 148us | 307us | **Triton 快 2.1x** |
| gemm | 136us | 93us | Triton 慢（cuBLAS 领地） |

### 关键结论：Triton 不是银弹

结果清楚地分成三类，对应三种 bound：

**第一类：memory-bound（vector_add、reduction）—— 打平**
搬多少字节就花多少时间，Triton 和 PyTorch 都受 DRAM 带宽限制，没有优化空间。元素级加法再怎么写也不会快，因为瓶颈在「把数据从内存搬进搬出」，不在算术。

**第二类：fusion 收益（rmsnorm、layernorm、softmax）—— Triton 赢**
PyTorch 的 `rmsnorm` 参考实现是手写组合 `x * rsqrt(x.pow(2).mean(-1) + eps) * w`，eager 模式会物化 `x.pow(2)`、`mean`、`rsqrt`、两个 `mul` 共 **5 个中间张量、5 次 kernel launch、5 次 global memory 往返**。Triton 一个 kernel 里算完，中间值只活在寄存器/SRAM。layernorm 的参考 `F.layer_norm` 本身已 fused，所以收益小（1.3x）；rmsnorm 没有内置 fused 实现，收益大（2.1x）。

**第三类：cuBLAS 领地（gemm）—— Triton 输**
`torch.matmul` 走的是 cuBLAS，它有数十年积累的 auto-tuning、tile 选择、软件流水线。从零手写一个 tiled matmul 打不过它是正常且诚实的。**但**——triton 的价值在于「cuBLAS 覆盖不到的算子」（如带 mask 的 attention、自定义量化 GEMM、fused layernorm+gemm），而不是在标准 GEMM 上硬碰 cuBLAS。

---

## 4. 各算子的实现要点

### vector_add —— 模板
边界 mask 处理 `n % BLOCK != 0` 的尾部，是后续所有 kernel 的骨架。

### reduction —— 两级归约 + atomic
```python
partial = tl.sum(x, axis=0)      # block 内归约
tl.atomic_add(out_ptr, partial)  # block 间归约
```
**本模块踩过的坑**：fp16 的 `atomic_add` 没有硬件原生支持，会降级成 CAS 循环（慢 16x），且 fp16 标量累加精度崩溃。所以 reduction 固定用 fp32 累加器——这是写量化/reduction kernel 时必须记住的。

### softmax —— 数值稳定的三步
```python
m = tl.max(x, axis=0)      # 减 max 防 exp 溢出
e = tl.exp(x - m)
s = tl.sum(e, axis=0)
out = e / s
```
这里每行一个 program、行宽正好一个 BLOCK，所以是「naive 版」。行宽超过 BLOCK 时要走 online softmax（分段跑 max→exp→sum 再合并），那是 flash-attention 的前置知识，留到 attention 篇。

### layernorm / rmsnorm —— 归约 + 广播 + scale
模式都是「`tl.sum` 沿行归约 → 广播回来 → elementwise scale」。rmsnorm 少一个 mean 中心化，是 LLaMA 系的标准归一化，也是「自定义 fused op」在真实模型上最常看到的第一个 win。

### gemm —— 分块 + `tl.dot`
```python
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, ...)   # 载 A tile 到 SRAM
    b = tl.load(b_ptrs, ...)   # 载 B tile 到 SRAM
    acc += tl.dot(a, b)        # 矩阵乘累加（fp16 -> Tensor Core）
```
**本模块踩过的坑**：fp32 下 `tl.dot` 默认 `input_precision="tf32"`，会丢掉尾数位导致和 `torch.matmul` 结果不一致。显式 `input_precision="ieee"` 才精确（但放弃 tf32 加速）。fp16/bf16 下这个参数被忽略（直接走 tensor core）。

---

## 5. BLOCK / num_warps / num_stages 调参

`benchmark.py` 的 sweep 结果（gemm fp16，1024³，device time）：

```text
BM=64  BN=64  BK=32  nw=4  ->  81.94us   (默认)
BM=64  BN=64  BK=64  nw=4  ->  79.40us
BM=64  BN=128 BK=64  nw=4  ->  78.20us   (最快，接近 cuBLAS 77.8us)
BM=128 BN=128 BK=32  nw=8  -> 102.71us   (更慢!)
```

读法：

1. **BK 从 32 → 64 更快**：K 维循环次数减半，A/B tile 的 load 次数减半，省下的是 global→SRAM 的搬运次数。

2. **num_warps 8 反而慢**：更大的 tile（128×128）配更多 warps，但每个 warp 分到的子 tile 更小、寄存器/共享内存压力增大、occupancy 下降，抵消了 tile 变大的收益。**盲加 warps 不是优化**。

3. **最优配置（78.2us）逼近 cuBLAS（77.8us）**：说明 Triton 的块级模型经过调参能达到 cuBLAS 量级；cuBLAS 的优势就是它**自动**做了这个 sweep 而且做得更细。

### 三个参数各自控制什么

| 参数 | 控制什么 | 调大收益 | 调大代价 |
|---|---|---|---|
| `BLOCK` | 每 program 处理的元素量 | 减少 launch 数、提高数据复用 | 寄存器/SRAM 压力↑、occupancy↓ |
| `num_warps` | 每 program 的 warp 数（并行度） | 更多并行 | 每 warp 分到的 tile 变小、同步开销↑ |
| `num_stages` | K 循环的软件流水线级数 | load 与 dot 重叠、隐藏访存延迟 | SRAM 用量↑（每 stage 一份 tile buffer） |

`num_stages` 只在带循环的 kernel（gemm 的 K 循环）有意义：它让「下一块 A/B 的 load」和「当前块的 dot」重叠，把访存延迟藏进计算里。代价是每个 stage 要占一份 SRAM。

---

## 6. 为什么某个 shape 快、某个 shape 慢

回到 master prompt 的要求：禁止只报「运行时间变短」，要解释为什么。以 gemm 为例：

```text
shape 影响：
  M/N 不是 BLOCK 整数倍 → 尾部 tile 有 mask 浪费（算力空转）
  K 太小            → K 循环次数少，load 延迟占比高，掩盖不了访存
  K 很大            → 数据复用充分，compute 主导，接近峰值

dtype 影响：
  fp32 + ieee → 不用 tensor core → compute 弱
  fp16        → tensor core → compute 强，但精度要看场景
  bf16        → tensor core + 更大动态范围，精度略低于 fp16

bound 判断：
  DRAM 接近峰值 + SM 空闲   → memory-bound（elementwise/reduction）
  tensor core 高           → compute-bound（大 gemm）
  两者都低 + occupancy 不足 → 调 BLOCK/num_warps/num_stages
```

判断 bound 需要 ncu 的 `SpeedOfLight`（DRAM vs SM/Tensor Core），本机 ncu 需授权（`note/profiling/02`），未授权时用 `sweep.py` 的参数敏感性推断。

---

## 7. Triton 在工业推理中的真实位置

```text
cuBLAS/cuDNN 覆盖 → 用现成（不用手写）
组合算子未覆盖 → Triton fused kernel（如 fused RMSNorm、RoPE、量化 GEMM、masked attention）
极致优化 / 特殊架构 → 手写 CUDA C++（Stage 2 的 cuda_core）
```

vLLM、SGLang、FlashAttention 的算子层大量用 Triton，因为「fused + 可移植 + 开发快」的平衡点恰好落在 Triton。这也是岗位 A（GPU Inference）核心技能。

---

## 8. 本模块闭环小结

```text
问题：PyTorch 覆盖不到的算子怎么在 GPU 上高效实现
      ↓
原理：块级编程模型 + 编译器处理 thread/SRAM/寄存器
      ↓
实现：vector_add → reduction → softmax → layernorm → rmsnorm → gemm
      ↓
验证：正确性（vs torch）+ benchmark（event/wall）+ sweep（BLOCK/num_warps）
      ↓
结论：memory-bound 打平、fusion 赢、cuBLAS 领地输
```

下一模块：继续 Triton 的 **attention 相关算子**（flash-attention 前置：online softmax → causal mask → attention）与**量化 kernel**（dequant + gemm），然后进入 Stage 6 Operator Fusion（用 Triton 做 Bias+Activation、Residual+RMSNorm、Dequant+GEMM、QKV fusion 的系统对比）。
