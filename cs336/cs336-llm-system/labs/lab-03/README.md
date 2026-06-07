# Lab 03: Systems — Kernels 与并行计算

## 任务目标

通过本实验，你将：

1. 理解 GPU kernel 编程的基本概念（以 Triton 为例）
2. 动手实现一个 fused RMSNorm kernel
3. 理解 kernel fusion 为什么能提升性能
4. 掌握 DDP (Distributed Data Parallel) 的原理和实现要点

## 实验任务

### Task 1: 理解 Triton Kernel 基础 (20%)

阅读 `background.md` 和提供的 skeleton 代码，理解：

1. Triton 的编程模型（program vs kernel）
2. Block-level 并行 vs thread-level 并行
3. Memory hierarchy：HBM → L2 → SRAM
4. `tl.load` / `tl.store` / `tl.reduce` 等基本操作

### Task 2: 实现 Fused RMSNorm Kernel (40%)

在 `starter.py` 中完成 Triton RMSNorm kernel：

```python
@triton.jit
def rmsnorm_fwd_kernel(...):
    """RMSNorm forward: y = x * w / sqrt(mean(x^2) + eps)"""
    # YOUR CODE HERE
```

要求：
1. 正确性：与 PyTorch 参考实现误差 < 1e-5
2. 性能：比 PyTorch 实现快（通过 operator fusion）
3. 支持任意 hidden dimension（通过 grid/block 切分）

### Task 3: DDP 原理题 (20%)

回答以下问题（代码 + 文字）：

1. DDP 中 AllReduce 梯度发生在哪个时机？为什么在 backward 而不是 forward？
2. Gradient bucketing 是什么？为什么能提升性能？
3. 如果使用 `find_unused_parameters=True` 会有什么性能影响？

### Task 4: Kernel 性能对比 (20%)

编写 benchmark 代码，比较：

1. Naive PyTorch RMSNorm (`torch.rsqrt` + element-wise)
2. PyTorch fused `nn.RMSNorm`（如可用）
3. 你的 Triton kernel

在不同 hidden dimension (1024, 4096, 8192, 16384) 下的延迟。

## 验收标准

- [ ] Triton RMSNorm kernel 输出与 PyTorch 误差 < 1e-5
- [ ] Fused kernel 在 hidden_dim=4096 时比 naive PyTorch 快 > 20%
- [ ] DDP 问题回答正确，包含对 gradient bucketing 的理解
- [ ] Benchmark 脚本能运行并生成对比数据

## 参考资料

- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [PyTorch DDP 文档](https://pytorch.org/docs/stable/notes/ddp.html)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)（理解 kernel fusion 的动机）

## 时间估计

约 4-5 小时
