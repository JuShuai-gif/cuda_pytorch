# Lab 02: Resource Accounting 与 GPU 计算

## 任务目标

通过本实验，你将：

1. 掌握 Transformer 模型的 FLOPs 计算方法
2. 理解 Roofline Model 和 Arithmetic Intensity 分析
3. 计算训练/推理阶段的内存占用与带宽需求
4. 学会使用 PyTorch Profiler 测量实际 GPU 性能

## 实验任务

### Task 1: FLOPs 计算 (30%)

在 `starter.py` 中实现函数，计算给定配置下的 Transformer FLOPs：

1. **Forward pass FLOPs**：QKV 投影、attention scores、FFN 等
2. **Backward pass FLOPs**：约为 forward 的 2 倍
3. 验证你的计算与论文中的数值一致

### Task 2: Memory Accounting (30%)

实现 GPU 显存计算器：

1. **Model weights** (parameters + gradients + optimizer states)
2. **Activations** (intermediate tensors during forward)
3. 计算 total memory 需求，判断是否能在指定 GPU 上训练

### Task 3: Roofline Model 分析 (20%)

1. 计算关键操作的 Arithmetic Intensity (FLOPs / byte)
2. 绘制简化的 Roofline 图
3. 判断各操作是 compute-bound 还是 memory-bound

### Task 4: PyTorch Profiler (20%)

使用 `torch.profiler` 测量一个简单 Transformer 的实际性能：

1. 记录 kernel 执行时间
2. 识别性能瓶颈
3. 分析 GPU utilization

## 验收标准

- [ ] Forward/backward FLOPs 计算与手工推导一致
- [ ] Memory 计算结果在 5% 误差以内
- [ ] Roofline 分析正确识别 compute-bound 和 memory-bound 操作
- [ ] 能够运行 PyTorch Profiler 并解读结果

## 参考资料

- [Transformer FLOPs 推导 (Kaplan et al.)](https://arxiv.org/abs/2001.08361)
- [Roofline Model 详解 (Williams et al., 2009)](https://people.eecs.berkeley.edu/~kubitron/cs252/readings/papers/Roofline-CommACM.pdf)
- [GPU Mode 讲义 - Arithmetic Intensity](https://github.com/gpu-mode/lectures)

## 时间估计

约 3 小时
