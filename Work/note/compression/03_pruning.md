# 03｜模型剪枝：参数减少 ≠ latency 下降

## 本模块解决的问题

剪枝是最直观的模型压缩手段：把"不重要"的参数归零。但工业上最大的误区是**把稀疏度当成加速比**。本章回答：

```text
为什么 99% 的参数剪掉，推理 latency 几乎不变？
理论 FLOPs reduction 和真实 hardware speedup 差在哪？
什么剪枝才能真正加速？2:4 稀疏是什么？
```

配套代码：`src/compression/pruning/`。

---

## 1. 剪枝的分类

| 类型 | 剪什么 | 稀疏结构 | 能否真实加速 |
|---|---|---|---|
| Unstructured | 单个权重 | 零散分布 | 难（见下） |
| Structured (row/channel) | 整行/整列/channel | 规则 | 能（维度减小） |
| Head pruning | 整个 attention head | 规则 | 能（head 维度减小） |
| Block sparsity (2:4) | 每 4 个元素保留 2 个 | 硬件规则 | 能（Tensor Core 支持） |

Head pruning 本质是 channel pruning 的 attention 特例（剪掉 Q/K/V 的整组输出通道）。

---

## 2. 为什么 unstructured 剪枝不加速

GPU 的 dense matmul（cuBLAS）**不感知零值**。一个 90% 稀疏的权重矩阵，cuBLAS 仍然做完整的矩阵乘——那些 0 照样被 load、照样参与乘加。所以：

```text
稀疏度 90% → FLOPs 减到 10% → 但 dense matmul 还是算 100% → speedup ≈ 1x
```

本机实测（1024³ fp16 matmul，baseline 32.7us）：

```text
unstructured 50%  稀疏  flops 50%  speedup 1.02x
unstructured 90%  稀疏  flops 10%  speedup 1.02x
unstructured 99%  稀疏  flops 1%   speedup 1.01x
```

**FLOPs 减到 1%，speedup 几乎为 0。** 这就是"理论 FLOPs reduction ≠ 真实 hardware speedup"的实锤。

要利用非结构化稀疏，需要**稀疏专用 kernel**（SpMM），但 SpMM 的硬件效率通常远低于 dense GEMM——稀疏矩阵的索引开销、非对齐访存、负载不均衡，让 SpMM 在大多数稀疏度下反而**更慢**于 dense。这就是为什么工业上非结构化剪枝主要用于"减小模型体积/显存"，而不是"加速推理"。

---

## 3. structured 剪枝：真实加速，但打折

剪掉整行（channel），权重矩阵维度真实变小，dense GEMM 直接缩小：

```text
剪掉 50% 行 → 矩阵 (512,1024) → GEMM FLOPs 减半 → 真实加速
```

本机实测：

```text
structured_row 50%  维度减半  speedup 1.34x   （不是 2x）
structured_row 75%  维度 1/4   speedup 1.60x   （不是 4x）
```

**为什么 speedup < FLOPs 减少比例？**

1. **小 GEMM 效率更低**：1024³ 的 GEMM 能喂满 Tensor Core，512×1024 的 GEMM tile 少、pipeline 短，峰值利用率下降。
2. **launch 开销和内存带宽不随 FLOPs 线性减少**：即使 FLOPs 减半，kernel launch、H2D、中间 activation 的访存没有同比例下降。

所以 structured 剪枝的 speedup 通常是 FLOPs reduction 的 **0.6~0.8 倍**，且只在剪掉足够多、且剩余 GEMM 仍够大时才有明显收益。

---

## 4. 2:4 块稀疏：让"剪枝"真正加速的硬件机制

NVIDIA Ampere+（含本机 Thor 的 Blackwell）Tensor Core 支持 **2:4 结构化稀疏**：每 4 个连续元素恰好保留 2 个非零。硬件专门为此设计了稀疏 GEMM 路径：

```text
dense: 每 4 元素存 4 个 → 算 4 个乘加
2:4  : 每 4 元素存 2 个 → 算 2 个乘加（硬件跳过零）→ 理论 2x 加速
```

2:4 的价值：**它是"结构化 + 细粒度"的结合**——稀疏模式是硬件定义的规则（不需要索引），同时剪枝粒度细（能保留 50% 的重要权重）。这是目前唯一能让"剪掉一半参数"稳定兑现 ~2x 加速的方式。

### 本机实测状态（诚实记录）

```text
torch.sparse.to_sparse_semi_structured：转换 API 存在
2:4 matmul：Not Supported（SparseSemiStructuredTensorCUSPARSELT matmul:
           operation is not supported）
```

本机 Jetson Thor 的 cuSPARSELt/CUTLASS 没有为 sm_110 提供 2:4 matmul kernel，所以 2:4 加速**标记 Not Validated**，只讲机制。这是又一次"API 存在 ≠ 硬件可用"（和 FP8 一样）。

---

## 5. 剪枝的工业实践结论

```text
1. 非结构化剪枝 → 减小模型体积/显存，但几乎不加速推理
2. 结构化剪枝（channel/head）→ 真实加速，但 speedup < FLOPs 减少比例
3. 2:4 稀疏 → 唯一"稳定 2x"的剪枝加速，但需要硬件 + 库支持
4. 剪枝通常和量化叠加（剪枝 + int8/int4）
```

真实工业里的剪枝主要用于：
- **减小显存**（权重减半能放下更大的模型 / 更大的 batch）
- **配合量化**做"压缩 + 加速"的组合
- **结构化剪枝**在算力受限的边缘设备上真实减小计算量

但要记住本模块的核心判断：**看到"模型参数减少 50%"时，不要默认"推理快 2x"——先问稀疏结构是什么、硬件能否利用。**

---

## 6. 本模块闭环小结

```text
问题：剪枝能加速推理吗
      ↓
原理：dense GEMM 不感知零值；结构化/2:4 才能被硬件利用
      ↓
实测：unstructured 99% 稀疏 speedup 1x；structured 50% speedup 1.34x
      ↓
结论：FLOPs reduction ≠ speedup，取决于稀疏结构 + 硬件
      ↓
下一步：Stage 10 模型蒸馏（large model → small edge model）
```

要继续就说「继续」。
