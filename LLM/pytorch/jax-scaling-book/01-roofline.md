**问题 1 [INT8 矩阵乘法]：** 假设我们想以 INT8 精度（每个参数 1 字节）而非 bfloat16（每个参数 2 字节）做矩阵乘法 $X[B,D] \cdot Y[D,F] \to Z[B,F]$，因为 TPU/GPU 在较低精度下做矩阵乘法更快。

- 需要从内存加载多少字节？需要写回内存多少字节？
- 总共执行多少 OPs？
- 算术强度是多少？
- $T_{\text{math}}$ 和 $T_{\text{comms}}$ 的 Roofline 估算是多少？整个操作的合理上下界是多少？

假设 HBM 带宽为 `8.2e11 bytes/s`，INT8 峰值 OPs/s 为 `3.94e14`（约为 bfloat16 的 2 倍）。

1.
- 需要加载的字节数 = BD + DF
- 需要写回的字节数 = BF

2. 执行的操作数 = 2BDF

3. 算术强度 = 2BDF / (BD + DF + BF)

假设 B（批次大小）相对于 D 和 F 较小，则近似 = 2BDF / DF = 2B

当 2B > (3.94e14 / 8.21e11) → B > 240 时进入计算密集型区域。
即当批次大小超过 240 时。

4.
- Tmath = 2BDF / 3.94e14
- Tcomm = (BD + DF + BF) / 8.2e11

下界是 max(Tmath, Tcomm)
上界是 Tmath + Tcomm

**问题 2 [INT8 + BF16 矩阵乘法]：** 实践中我们常对权重和激活值采用不同的量化策略，即用非常低的精度存储权重，但保持激活值（和计算）在较高精度。假设权重使用 INT8 量化，但激活值（和计算）保留 bfloat16。在多大批次大小时进入计算密集型区域？假设 bfloat16 的峰值 FLOPs/s 为 1.97e14。

提示：这具体指 bf16$[B, D]$ × int8$[D, F]$ → bf16$[B, F]$，其中 $B$ 是批次大小。

算术强度 = 2BDF / (2BD + DF + 2BF)
同样近似下 = 2B > 240 即 B > 120

**问题 3：** 在问题 2 的设置下，绘制 $F=D=4096$ 和 $F=D=1024$ 时峰值 FLOPs/s 与批次大小 $B$ 的 Roofline 图。使用精确的字节加载量，不做近似。

```
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

**问题 4：** 如果我们要执行 INT8$[B,D]$ ⋅ INT8$[B,D,F]$ → INT8$[B,F]$，即每个批次元素有不同矩阵，此操作的算术强度是多少？

算术强度 = 2BBDF / (BD + BDF + BF)

*待办 - 这个计算有误，需要进一步理解*
