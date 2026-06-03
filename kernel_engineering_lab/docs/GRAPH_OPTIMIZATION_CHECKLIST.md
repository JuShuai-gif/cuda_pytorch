# 图优化检查清单

计算图层级优化的系统化检查清单。在构建图优化器或融合 pass 时，逐项检查每个部分。

---

## 1. 逐元素融合模式

### 水平融合

- [ ] **融合相邻的逐元素操作**：如 `x = relu(x); x = dropout(x)` 这样的连续操作应为单个 kernel。
- [ ] **融合逐元素 + 归约预备操作**：`sum(x)` 之前的 `x = x * scale` 可以融合到归约中。
- [ ] **Broadcast + 逐元素**：bias 是 1D、x 是 ND 的 `x + bias` 模式可以融合进来。
- [ ] **类型转换链**：`float32 -> float16 -> float32` 的转换对应该被折叠。

### 垂直融合

- [ ] **生产者-消费者链**：A -> B，两者都是逐元素操作且没有 A 的其他消费者。
- [ ] **Reduce-scatter + 逐元素**：归约输出被逐元素操作消费。
- [ ] **多输出融合**：当单个输入 tensor 馈送多个逐元素操作时，考虑将它们融合为一个具有多个输出的 kernel。

### 需要检查的约束条件

- [ ] **内存预算**：融合 kernel 的总 shared memory + 寄存器内存应在 SM 限制内。
- [ ] **Occupancy 影响**：融合太多操作可能增加寄存器压力并降低 occupancy。使用 Nsight Compute 测试。
- [ ] **Launch 开销收益**：融合节省 N-1 次 kernel launch。如果 N=2 且每个 kernel 是 5us，节省 5us 是微不足道的。
- [ ] **中间 tensor 消除**：确认中间 tensor 不被任何下游操作消费。

---

## 2. 归一化融合模式

### LayerNorm 融合

- [ ] **需要融合到单个 LayerNorm kernel 的子表达式**：
  - `mean = sum(x) / N`
  - `var = sum((x - mean)^2) / N`
  - `x_norm = (x - mean) / sqrt(var + eps)`
  - `output = x_norm * gamma + beta`
  - 在单个 kernel 中完成所有四个操作，只需一次数据遍历。

### RMSNorm 融合

- [ ] **融合 RMSNorm kernel 应包含**：
  - `rms = sqrt(sum(x^2) / N + eps)`
  - `output = x / rms * gamma`
  - 两次数据遍历 -> 一次遍历。

### 融合 Residual + Norm

- [ ] **融合 residual add + LayerNorm**：`x = x + residual; x = layernorm(x)` -> 单个 kernel。节省一次完整的 x 读/写。
- [ ] **Residual 中的可选 dropout**：`x = x + dropout(residual)` 可以融合到同一个 kernel 中。
- [ ] **融合 QKV 投影 + LayerNorm**：常见于 transformer attention：`q, k, v = split(W @ layernorm(x))`。

---

## 3. Dead Code Elimination（DCE）

- [ ] **移除未使用的节点输出**：从图输出反向遍历图，标记可达节点，移除不可达节点。
- [ ] **移除只写 tensor**：如果一个 tensor 只被写入从未被读取，消除它（如果没有副作用，也消除其生产者）。
- [ ] **参数剪枝**：在前向计算图中未使用的模型参数。
- [ ] **梯度 barrier 消除**：在推理模式下，梯度记录 tensor 是死代码。

### 算法
```
1. 从图输出节点开始
2. 沿数据边反向 BFS/DFS
3. 将所有访问的节点标记为 "活跃"
4. 删除所有未标记的节点
5. 重新运行直到不动点（删除一个节点可能使它的输入变为死代码）
```

---

## 4. Constant Folding

- [ ] **折叠标量常量的操作**：`3.0 + 4.0 * 2.0` -> `11.0`，在图编译时完成。
- [ ] **折叠常量输入形状的形状推断**：如果输入形状是静态的，预计算 broadcasting 和 reshape 决策。
- [ ] **折叠零/恒等模式**：
  - `x + 0` -> `x`
  - `x * 1` -> `x`
  - `x * 0` -> `zeros_like(x)`（注意：NaN * 0 = NaN，但如果 x 已知是有限的，这是安全的）
  - `concat([x])` -> `x`
  - `slice(tensor, all axes)` -> `tensor`
- [ ] **折叠转置链**：`transpose(transpose(x, [1, 0]), [1, 0])` -> `x`。

### 安全检查
- [ ] **浮点**：常量折叠可能产生稍微不同的浮点结果。对 ML 工作负载可以接受。
- [ ] **除零**：`1.0 / 0.0` -> `inf` 是正确的，但需要警告。
- [ ] **大型常量**：不要物化太大而无法存储在 IR 中的常量 tensor。

---

## 5. 内存规划

### Buffer 复用

- [ ] **In-place 操作**：当 `y = op(x)` 且 `x` 在之后不再使用时，将 `x` 的 buffer 复用于 `y`。
- [ ] **Buffer 分配算法**：
  1. 计算每个 tensor 的活跃区间（首次使用到最后使用）
  2. 创建干涉图（如果两个 tensor 的区间重叠，则它们干涉）
  3. 为不干涉的 tensor 分配颜色（内存 buffer）
- [ ] **最小 buffer 数量**：图着色启发式算法（贪心在实践中效果很好）。

### 峰值内存减少

- [ ] **重新计算代替存储**：通过重新计算中间 tensor 而非存储它们，以计算换取内存（对训练尤其重要：activation checkpointing）。
- [ ] **Gradient checkpointing**：标记重新计算昂贵的操作保留，重新计算便宜的操作丢弃。
- [ ] **逐层调度**：一次执行并释放一层，而不是物化整个前向图。

### 内存池分配

- [ ] **Bump allocator**：对于已知的、生命周期短的 tensor，使用简单的 bump-pointer 分配器配合空闲列表。
- [ ] **大小类分桶**：按大小类对分配进行分组以便高效复用。
- [ ] **CUDA memory pool**：使用 `torch.cuda.caching_allocator_alloc` / `torch.cuda.caching_allocator_delete`（PyTorch 会自动执行此操作）。

---

## 6. Operator 调度策略

### 拓扑排序变体

- [ ] **BFS/DFS 调度**：标准的尊重依赖关系的拓扑排序。
- [ ] **深度优先内存最小化**：首先处理深层分支，尽早释放内存。
- [ ] **广度优先并行最大化**：暴露独立操作以进行潜在的并发 kernel launch。

### 并发

- [ ] **识别独立子图**：没有数据依赖的操作可以并发运行。
- [ ] **Stream 分配**：将独立子图分配到不同的 CUDA stream。
- [ ] **拷贝-计算重叠**：调度 `cudaMemcpyAsync` 进行数据传输，同时在其他数据上进行计算。

### Batch 聚合

- [ ] **分组小 launch**：如果多个小逐元素操作无法融合，将它们合并为一个操作（通过 JIT 的水平融合）。
- [ ] **CUDA Graph 捕获点**：确定图中形状稳定的区域——捕获一次，重放。

---

## 7. Common Subexpression Elimination（CSE）

- [ ] **识别结构相同的子图**：两个节点使用相同的输入计算相同的函数。
- [ ] **基于哈希的去除重复**：对节点进行哈希（op_type、input_ids、attributes），合并哈希相同的节点。
- [ ] **归约 CSE**：`sum(x) + sum(x)` -> `2 * sum(x)`。两个 sum 从相同输入计算。
- [ ] **矩阵乘法 CSE**：`(x @ W) + (x @ W)` -> `2 * (x @ W)`。

### 作用域考量
- [ ] **局部 CSE**：在基本块或小窗口的操作内。
- [ ] **全局 CSE**：跨整个图（成本更高，收益更大）。
- [ ] **跨模块边界**：例如，两个 attention 层计算相同的投影可能被去重。

---

## 8. 图 Pass 基础设施

### Pass Manager

- [ ] **Pass 注册**：每个 pass 声明其名称、依赖关系和所需分析。
- [ ] **Pass 排序**：由依赖关系 + 手动指定的优先级定义。
- [ ] **不动点迭代**：运行 pass 直到图停止变化（DCE 可能暴露新的融合机会）。
- [ ] **Pass 验证**：在每次 pass 后，验证图的不变量（无悬挂边、有效形状等）。

### 分析原语

- [ ] **Use-def chains**：对于每个 tensor，追踪所有生产者和消费者。
- [ ] **活跃分析**：对于每个 tensor，计算首次使用和最后使用的位置。
- [ ] **形状推断**：在图传播形状；尽早检测形状不匹配。
- [ ] **别名分析**：检测两个 tensor 共享相同内存（view、reshape、slice）。

### 调试基础设施

- [ ] **图可视化**：以 DOT 格式导出图，用于 Graphviz 可视化。
- [ ] **优化前后对比**：计算优化前后的 kernel 数量、内存使用、FLOPs。
- [ ] **数值验证**：运行参考图 vs 优化图并比较输出（`torch.allclose`）。

---

## 9. 集成点

### 与 PyTorch FX

- [ ] **图捕获**：使用 `torch.fx.symbolic_trace()` 进行图捕获。
- [ ] **自定义 pass**：实现 `torch.fx.Interpreter` 用于分析，`torch.fx.Transformer` 用于重写。
- [ ] **Lowering 到 Inductor**：`torch.compile(backend="inductor")` 自动处理许多这些 pass。

### 与 TorchScript

- [ ] **JIT trace/script**：`torch.jit.trace` / `torch.jit.script` 用于图捕获。
- [ ] **自定义融合 pass**：使用 `torch.jit.FusionStrategy` 注册。

### 与 Triton Kernel 生成

- [ ] **基于模板的 codegen**：将融合子图映射到 Triton kernel 模板。
- [ ] **形状特化**：为常见形状生成不同的代码路径。
- [ ] **Auto-tune 分发**：根据问题规模选择最佳 kernel 变体。

---

## 10. 成功指标

在每次优化 pass 前后追踪以下指标：

| 指标 | 描述 | 目标 |
|--------|-------------|--------|
| Kernel 数量 | GPU kernel launch 总次数 | 减少 30-80% |
| 内存流量 | 每次迭代读/写的 GB 数 | 减少 20-60% |
| 峰值内存 | 最大已分配 GPU 内存（GB） | 减少或不变 |
| 端到端延迟 | 每次迭代的墙上时钟时间 | 减少 10-50% |
| 编译时间 | 图优化 + codegen 时间 | 典型模型 < 5 秒 |
| 数值误差 | `allclose(optimized, reference)` | rtol=1e-3, atol=1e-5 |
