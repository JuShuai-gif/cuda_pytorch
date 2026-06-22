# 04_operator_fusion - Operator Fusion

## 工业背景：为什么融合对 LLM 推理至关重要

Operator Fusion 可以说是对推理性能影响最大的单一优化手段。单个 transformer block 在 matmul 之间包含数十个逐元素操作。
每个未融合的 op 都要从全局内存（HBM/DRAM）读取和写入。将它们融合后**消除了中间内存传输**，让数据留在寄存器或 L1/shared memory 中。

### 内存墙问题

现代 GPU 拥有巨大的计算能力但内存带宽有限：

| GPU | 峰值 FP16 TFLOPS | 内存带宽 (TB/s) |
|-----|-----------------|-------------------|
| H100 | 989 | 3.35 |
| A100 | 312 | 2.0 |
| RTX 4090 | 330 | 1.0 |

对于 1.0 TB/s 的带宽以及每个元素约 12 字节读写量的逐元素操作，可实现吞吐量仅为约 830 亿元素/秒。而同一 GPU 上的 matmul 可以达到数百 TFLOPS。

**在 decode 阶段（batch=1），大多数操作受内存带宽限制。** Fusion 直接解决了这一瓶颈。

### 融合节省了什么

考虑一个没有融合的 transformer FFN block 中的冗余内存 I/O：

```
未融合（每个 op 都写入全局内存）：
  h = linear1(x)        # 计算密集型 matmul
  h = h + bias          # 读 h、读 bias、写 h_intermediate（3 次传输）
  h = gelu(h)           # 读 h_intermediate、写 h_output（2 次传输）
  # 逐元素内存总流量：5 次 tensor 访问
  # = 5 * n_elements * sizeof(dtype)

融合后（一个 kernel）：
  h = linear1(x)        # 相同的 matmul
  h = fused_bias_gelu(h, bias)
  # 读 h、读 bias、写 h_output（3 次传输）
  # 总计：3 次 tensor 访问
  # 内存流量减少约 40%
```

在实际 LLM 中，融合 residual + norm + activation 可以在逐元素区域带来 **2-5 倍加速**，因为直接解决了内存瓶颈。

## 生产系统中如何处理融合

| 系统 | 方式 |
|--------|----------|
| **torch.compile / inductor** | 自动 FX 图级别融合、模式匹配、循环融合 |
| **xFormers** | 手写融合 attention + MLP kernel |
| **vLLM** | PagedAttention 与 rotary embeddings 融合、融合 MoE |
| **FlashInfer** | 融合 sampling、top-k、top-p kernel |
| **Triton** | 程序员通过编写自定义 kernel 控制融合 |

torch.compile 使用 torch.fx 追踪计算图并自动应用融合 pass。它可以融合一系列逐元素操作、归约，甚至某些 matmul 变体。

## 本模块的 Kernel

### kernel_add_relu.py
融合 add + ReLU。工业应用：前馈网络（Wx + b，然后 ReLU）。
节省：2 次融合全局内存访问 vs 5 次顺序访问。

### kernel_bias_gelu.py
融合 bias + GELU。工业应用：transformer FFN，紧跟在 linear 层之后。GELU 比 ReLU 更复杂，使得融合价值更高。

### kernel_residual_layernorm.py
融合 residual（x + f(x)）+ LayerNorm。工业应用：每个 transformer block。
为清晰起见进行了简化：无可学习的 gamma/beta。消除了 2 个中间 tensor。

### kernel_rmsnorm.py
RMSNorm kernel。工业应用：LLaMA、Mistral、Gemma 使用 RMSNorm 替代 LayerNorm。
带可学习 weight 参数的完整实现。RMSNorm 更简单（无需计算均值）且更快。

### torch_fx_fusion.py
演示基于 torch.fx 的融合：追踪模型，应用自定义 add+relu 和 bias+gelu 融合 pass，显示融合前后的图。这就是 torch.compile 底层使用的机制。

## 常见陷阱

### 1. 融合边界
并非所有操作都应该融合。将计算密集型 matmul 与内存受限的逐元素操作融合可能因降低 occupancy 而损害性能。只融合**都是内存受限**的操作，或具有明确数据依赖链的操作。

### 2. 数值稳定性差异
融合 kernel 可能产生与顺序执行不同的数值结果，原因包括：
- 不同的累加顺序
- 降低精度的中间值未存储到内存
- FMA vs 分离的乘法和加法

始终以适当的容差测试融合 kernel 与顺序基线的对比。

### 3. 形状不匹配
当融合涉及 broadcasting 的操作时（如形状为 [D] 的 bias 作用于 [B, D]），确保融合 kernel 中的 broadcast 语义正确。在传入 kernel 之前先扩展 bias。

### 4. torch.compile 行为
torch.compile + inductor 可能在不同的 GPU 架构或不同的 triton 版本上产生不同的融合。始终在目标硬件上做 benchmark。

## 运行测试

```bash
pytest 04_operator_fusion/test_operator_fusion.py -v
```

## 运行基准测试

```bash
python 04_operator_fusion/benchmark_operator_fusion.py
```

## 运行 torch.fx 融合演示

```bash
python 04_operator_fusion/torch_fx_fusion.py
```

## 参考文献

- **Operator Fusion**：Jia et al. (2019) - "TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions"
- **PyTorch Inductor**：https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- **Triton Fused Kernels**：https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
