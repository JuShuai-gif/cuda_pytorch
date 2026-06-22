# 09_graph_optimization - 图级优化

## 工业背景：ML 编译器架构

图级优化是每个现代 ML 编译器的基础。这些系统不是优化单个 operator kernel，而是将**计算图**作为一个整体进行优化——在生成任何代码之前转换程序的结构。

### 主要 ML 编译器系统

| 系统 | IR | 关键图优化 | 目标 |
|--------|-----|------------------------|--------|
| **XLA (Google)** | HLO (High-Level Optimizer) | 融合、布局优化、代数化简 | TPU, GPU, CPU |
| **TensorRT (NVIDIA)** | TensorRT Network Definition | 层融合、tensor 格式优化、精度校准 | NVIDIA GPU |
| **torch.compile / inductor** | FX Graph + Triton IR | 模式匹配融合、循环融合、CSE、常量折叠 | PyTorch → Triton/OpenAI |
| **TVM (Apache)** | Relay IR + TIR | Operator 融合、布局变换、自动调度 | CPU, GPU, 专用加速器 |
| **MLIR (LLVM)** | MLIR dialects | 多级 IR、渐进式 lowering、dialect 转换 | 通用 |
| **ONNX Runtime** | ONNX Graph | 图分区、节点融合、常量折叠 | 多平台 |

### 价值主张

图优化可以在**零 kernel 工作**的情况下带来 **10-50% 的端到端加速**。
关键洞察：数据移动主导了 kernel 的执行成本。

```
未融合（每个 op 是一个独立的 kernel launch）：
  x = matmul(w, x)         # Kernel 1：计算密集型
  x = x + bias             # Kernel 2：内存密集型（读 x、bias；写 x）
  x = relu(x)              # Kernel 3：内存密集型（读 x；写 x）
  x = x + residual         # Kernel 4：内存密集型（读 x、residual；写 x）
  x = layernorm(x)         # Kernel 5：内存密集型（读 x；写 x、保存统计量）

  总计：5 次 kernel launch，13 次 tensor 从全局内存读/写

融合后（单个融合 kernel）：
  x = fused_residual_layernorm(relu(matmul(w, x) + bias), residual)

  总计：2 次 kernel launch（matmul + 融合），6 次 tensor 读/写
```

### 融合如何减少内存带宽

内存墙是大多数操作的主要瓶颈：

| 操作类型 | 算术强度 (FLOP/byte) | 受限因素 |
|---------------|--------------------------------|-------|
| Matmul（大） | ~100-1000 | 计算 |
| 逐元素（ReLU、add） | ~0.25 | 内存 |
| LayerNorm / RMSNorm | ~10 | 内存 |
| Softmax | ~5 | 内存 |
| 归约 | ~0.5 | 内存 |

融合逐元素操作可以消除多次读/写循环：
- 3 个顺序逐元素 ops：约 6 次读 + 3 次写 = 9 次内存操作
- 融合为 1 个 kernel：约 3 次读 + 1 次写 = 4 次内存操作
- **内存流量减少：55%**

## 图重写基础设施

### 模式匹配 + 融合

TensorRT、XLA 和 torch.inductor 中的核心算法：

1. **图遍历**：按拓扑顺序遍历 DAG
2. **模式匹配**：对每个节点，检查是否匹配已知模式
3. **子图替换**：将匹配的子图替换为融合操作
4. **不动点迭代**：重复直至没有更多模式匹配

常见融合模式：
- Conv + BatchNorm + ReLU（CNN）
- MatMul + Bias + Activation（Transformer）
- Residual + LayerNorm（Transformer）
- Multi-head attention（Q、K、V 投影 + scores + softmax）

### Dead Code Elimination（DCE）

移除从图输出不可达的节点。死代码来源：
- 推理时的梯度计算节点
- 未使用的常量 tensor
- 融合创建新节点后的中间值
- 控制流优化后被废弃的分支

算法：从输出节点反向 BFS/DFS，标记可达节点，移除其余节点。

### Constant Folding

在图优化时计算所有输入均为常量的子表达式：
- 编译时已知的形状
- 来自配置文件的权重
- 超参数（如缩放因子、epsilon）

支持：标量和 tensor 常量的 ADD、SUB、MUL、DIV。级联折叠（如果折叠结果提供给另一个可折叠节点，则再次折叠）。

### Common Subexpression Elimination（CSE）

相同的计算子图 → 计算一次，重用结果：
- 按 (op_type, input_ids, attrs) 对节点进行哈希
- 合并重复的哈希条目
- 将所有消费者重定向到规范实例

CSE 机会的常见来源：
- 多个计算路径上相同的 bias 加法
- 重复的归一化（如相同的 epsilon）
- 权重共享导致的重复 matmul

## Pass 顺序至关重要

优化 pass 的顺序非常关键：

```
不正确的顺序：
  DCE → CSE → CF → Fusion
  问题：DCE 先运行，但 CF 尚未创建新的常量；
        Fusion 错过了 CSE 本可暴露的机会

推荐顺序：
  CF → CSE → Fusion → DCE
  理由：
    1. CF 首先减少常量（使更多 CSE 成为可能）
    2. CSE 去除重复的共享子图（使更多融合成为可能）
    3. Fusion 将模式替换为融合 ops
    4. DCE 清理前面 pass 遗留的所有内容

  可能需要多次迭代（不动点循环）直到收敛。
```

## 模块结构

### ir.py - 计算图 IR
受 XLA HLO 和 ONNX 启发的最小但完整的 IR。关键设计：
- Node ID 为整数（快速哈希、比较）
- 边以 node ID 的输入/输出列表形式存储
- OpType 枚举涵盖激活、算术、归一化和融合 ops
- 带符号维度支持的 TensorShape
- Graph 方法：add_node、topological_sort、clone、to_dot (GraphViz)、validate

### passes.py - 优化 Pass
实现优化管线的四个 pass：
- **dead_code_elimination**：移除不可达节点（从输出反向 BFS）
- **constant_folding**：计算常量表达式（常量的 ADD、SUB、MUL、DIV）
- **common_subexpression_elimination**：去除重复的相同计算
- **pattern_fusion**：融合 ADD+RELU、ADD+GELU、ADD+RMSNORM 模式
- **optimize_graph**：以正确顺序运行所有 pass，使用不动点迭代

### executor.py - 图解释器
用于验证正确性的参考执行器。按拓扑顺序运行节点：
- 使用 torch 实现支持所有 OpType ops
- 实现激活函数（GELU、SiLU、ReLU、Tanh、Sigmoid、Exp、Log）
- 实现归一化（LayerNorm、RMSNorm），使用简化公式
- 实现融合 ops（ADD+RELU、BIAS+GELU、RESIDUAL+RMSNORM）

### graph_demo.py - 演示管线
构建一个真实的 transformer block 图并运行完整优化：
1. 使用 QKV 投影、激活、residual、RMSNorm 构建图
2. 以 DOT 格式打印原始图
3. 应用 optimize_graph
4. 打印优化后的图，显示节点数量减少
5. 执行两个图，验证结果匹配

### test_graph_optimization.py - 测试套件
全面的 pytest 测试，覆盖：
- IR 构建：add_node、topological_sort、clone、validate
- DCE：移除不可达节点、保留可达节点
- CF：折叠常量、级联折叠、跳过非常量
- CSE：去除重复的相同节点、保留不同的 ops
- Fusion：检测并融合模式、减少节点数量
- 端到端：优化、执行、比较输出

### benchmark_graph_optimization.py - 基准测试
测量优化影响：
- 层数扫描（4、8、16、32、64 层）
- CSE 影响（有意冗余的图）
- 融合加速（具有许多可融合模式的图）
- 内存节省估计（中间 tensor 减少）
- Hidden size 扫描（32 到 1024）

## 常见陷阱

### 1. 非规范图形状导致错失融合机会

不同框架构建的图可能以不同方式表示相同的计算：

```
# 两种表示相同操作的方式
# 方式 1（直接）：ADD(x, bias) → RELU
# 方式 2（带缩放）：MUL(ADD(x, bias), scale) → RELU
                      ↑ 额外的 scale 阻止了模式匹配

# 解决方案：在融合之前运行代数化简（此处未实现）。
# 在生产环境（XLA、TensorRT）中，canonicalization pass
# 在模式匹配之前规范化图形状。
```

### 2. Pass 顺序至关重要

以错误顺序运行 pass 可能错失优化机会：
- DCE 在 CSE 之前：未使用的重复节点存活
- CF 在 Fusion 之后：可融合位置中的常量永远不会被折叠
- CSE 在 CF 之前：常量重复节点存活

解决方案：以正确的初始顺序进行不动点迭代。

### 3. CSE 误报

两个节点可能看起来相同但计算不同的东西：
```python
# 节点 A：ADD([1, 2])，输入 1 = constant_3.14，输入 2 = input_x
# 节点 B：ADD([1, 2])，输入 1 = constant_3.14，输入 2 = input_x
# 相同的哈希，但形状可能不同 → CSE 会错误地合并它们

# 缓解措施：在节点签名中包含 shape/dtype
# （我们的 node_signature 为简洁起见有意排除了形状；
#  生产系统会包含它以防止误报）
```

### 4. 融合破坏反向传播

融合前向传播的 ops 时，反向传播所需的中间 activations 会丢失。在训练中，融合仅限于推理路径，或者反向传播单独处理（例如 PyTorch 的 autograd 引擎记录融合反向传播公式）。

### 5. Operator 语义变化

融合操作会产生与顺序执行不同的数值结果：
- 不同的舍入（更少的中间舍入步骤）
- 不同的操作顺序（如融合乘加 vs 分离）
- 降低精度的中间值（值保留在寄存器中）

始终以适当的容差（rtol=1e-4 或更高）验证融合 vs 未融合的结果。

### 6. 图循环

有效的计算图必须是 DAG。循环表明：
- 循环连接（RNN、LSTM）→ 需要特殊处理（scan ops）
- 图构建 bug → 每次 pass 后验证

我们的 `validate()` 方法通过拓扑排序检查循环。

## 运行测试

```bash
pytest 09_graph_optimization/test_graph_optimization.py -v
```

## 运行演示

```bash
python 09_graph_optimization/graph_demo.py
```

## 运行基准测试

```bash
python 09_graph_optimization/benchmark_graph_optimization.py
```

## 参考文献

- **XLA**："XLA: Optimizing Compiler for Machine Learning"，TensorFlow，https://www.tensorflow.org/xla
- **TensorRT**："NVIDIA TensorRT Developer Guide"，https://docs.nvidia.com/deeplearning/tensorrt/
- **torch.compile**："TorchDynamo and TorchInductor"，https://pytorch.org/docs/stable/torch.compiler.html
- **TVM**：Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018
- **MLIR**：Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation", CGO 2021
- **TASO**：Jia et al., "TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions", SOSP 2019
- **图优化**：Aho, Lam, Sethi, Ullman - "Compilers: Principles, Techniques, and Tools"（龙书），第 9-10 章
