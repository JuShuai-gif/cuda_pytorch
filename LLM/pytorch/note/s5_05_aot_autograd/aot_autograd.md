# AOTAutograd: 从 Eager Autograd 到编译后反向图

> 核心实现: `torch/_functorch/aot_autograd.py`
> Partitioner: `torch/_functorch/partitioners.py`
> Decomposition: `torch/_decomp/__init__.py`, `torch/_decomp/decompositions.py`
> 参考实现: `torch/_refs/`

## 0. 一句话总览

AOTAutograd 是将 PyTorch 的自动微分从 eager 模式提升到编译器可用的图模式的桥梁。它 forward+backward 整体 trace 成一个 joint FX graph，再切分成 forward-only 和 backward-only 两张子图，交给 Inductor 等后端编译。核心难点在于：从可变、有副作用的 eager 语义中捕获出纯函数式的可编译图。

## 1. 最小例子

```python
import torch
import torch.nn.functional as F

def f(x, w, b):
    return F.layer_norm(x, x.shape[-1:], w, b).sin().sum()

x = torch.randn(4, 8, requires_grad=True)
w = torch.ones(8, requires_grad=True)
b = torch.zeros(8, requires_grad=True)

loss = f(x, w, b)
loss.backward()
```

使用 AOTAutograd 显式 trace:

```python
from torch._functorch.aot_autograd import aot_function

def fw_compiler(gm, example_inputs):
    print("Forward graph:")
    gm.print()
    return gm

def bw_compiler(gm, example_inputs):
    print("Backward graph:")
    gm.print()
    return gm

aot_f = aot_function(f, fw_compiler, bw_compiler)
loss = aot_f(x, w, b)
loss.backward()
```

## 1.5 实战例子

### 1.5.1 用 AOTAutograd 打印 joint graph 调试反向

当反向传播结果异常时，直接查看 AOTAutograd 的 joint graph 和切分后的图：

```python
import torch
from torch._functorch.aot_autograd import aot_function

def f(x, w):
    return (x @ w).sin().sum()

def debug_fw(gm, example_inputs):
    print("=== Forward Graph ===")
    gm.graph.print_tabular()
    return gm

def debug_bw(gm, example_inputs):
    print("=== Backward Graph ===")
    gm.graph.print_tabular()
    return gm

x = torch.randn(4, 8, requires_grad=True)
w = torch.randn(8, 3, requires_grad=True)

aot_f = aot_function(f, debug_fw, debug_bw)
loss = aot_f(x, w)
loss.backward()
```

通过观察 forward graph 中哪些节点被保留（保存激活），backward graph 中哪些节点重算，可以诊断显存异常或计算冗余。

### 1.5.2 Partitioner 显存规划分析

对比不同 `mode` 下 partitioner 的行为差异：

```python
import torch
from torch._functorch.partitioners import min_cut_partition

# 假设一个包含大中间激活的模型
def large_model(x):
    for i in range(10):
        x = torch.nn.functional.linear(x, torch.randn(4096, 4096, device="cuda"))
        x = torch.nn.functional.relu(x)
    return x.sum()

# 在默认模式 vs reduce-overhead 下
# partitioner 会选择不同的保存/重算策略
# mode="reduce-overhead" 倾向保存更多激活 (减少重算)
# mode="max-autotune" 倾向尝试所有策略
```

实际场景中，当遇到 "CUDA out of memory" 时，切换到 `torch.compile(mode="max-autotune")` 可能因为保存更少激活而通过。

### 1.5.3 Decomposition 导致精度误差的排查

某些 op 被 decomposition 拆解后，由于浮点运算顺序改变产生精度差异：

```python
import torch
import torch._decomp as decomp

# 查看某个 op 是否有 decomposition
has_decomp = decomp.get_decompositions([torch.ops.aten.layer_norm])
print(f"layer_norm decomposition exists: {has_decomp is not None}")

# 对比 eager 和 decomposition 的结果
def f(x):
    return torch.nn.functional.layer_norm(x, x.shape[-1:])

x = torch.randn(4, 8)
out_eager = f(x)

# 禁用 decomposition 后执行
with torch._dynamo.config.patch("decompose", False):
    out_nodecomp = ...  # 绕过 decomposition

print("Max diff:", (out_eager - out_nodecomp).abs().max())
```

当训练 loss 曲线与预期偏离时，可以此方法判断是否由 decomposition 引入的精度问题导致。

## 2. 从 Python API 到源码的调用链

```
torch.compile(model)                             # 入口
  -> Dynamo 捕获 FX graph
  -> call_user_compiler(gm, example_inputs)       # 调用 backend
  -> AOTAutograd (torch/_functorch/aot_autograd.py)

AOTAutograd 内部流程:
  1. aot_function() / aot_module()
  2. 创建 AOTDispatchAutograd 或 AOTDispatchSubclassWrapper
  3. run_functionalized_fw_and_collect_metadata()
     -> 开启 functionalization, 消除 mutation
     -> 用 make_fx + _autograd_grad 同时 trace forward 和 backward
     -> 产出 joint graph (forward + backward 合体)
  4. create_joint_graph_key -> partition (partitioners.py)
  5. 前向图 -> fw_compiler
     反向图 -> bw_compiler (min_cut 或 default partitioner)

Partitioner 决策:
  -> min_cut_partition: 在激活内存和重算之间做 tradeoff
  -> 某些 op 会被同时保留在前向（保存激活）和反向（重算）
```

## 3. 核心源码文件

```
torch/_functorch/aot_autograd.py          # AOTAutograd 主逻辑
torch/_functorch/partitioners.py          # joint graph 切分器
torch/_functorch/eager_transforms.py      # 函数式 grad/vmap/etc
torch/_decomp/__init__.py                 # decomposition table 入口
torch/_decomp/decompositions.py           # 各算子的 decomposition
torch/_refs/                              # 参考实现（算子分解的目标）
tools/autograd/derivatives.yaml           # Autograd 公式注册
torch/fx/__init__.py                      # FX graph 基础设施
```

## 4. 关键机制源码解读

### 4.1 Eager Autograd vs AOTAutograd 的边界

| 特性 | Eager Autograd | AOTAutograd |
|------|---------------|-------------|
| 图构建 | 动态构建 Node（运行时） | 一次性 trace 出完整图（编译时） |
| 反向执行 | AutogradEngine 按拓扑遍历 | 编译为独立的反向 FX graph |
| 支持动态 control flow | 天然支持 | 不支持（graph break） |
| 内存优化 | 自动释放中间激活 | 分区器决定保留/重算 |
| 编译器接入 | 无 | 输出给 Inductor 等 |

### 4.2 decomposition table 的作用

Decomposition 是将一个高阶或复杂 op 拆解为更 primitive op 的过程。例如：

```python
# torch/_decomp/decompositions.py
@register_decomposition(aten.layer_norm)
def layer_norm_decomp(x, normalized_shape, weight, bias, eps):
    # 拆解为 mean, rstd, 加减乘除等基本 op
    ...
```

三处使用 decomposition:

1. **Dynamo**: trace 时分解某些 op 以减少 graph break
2. **AOTAutograd**: 在 joint graph 阶段分解，使 backward 可以更细粒度做算子融合
3. **Inductor**: lowering 阶段分解 aten op 为更底层的 IR ops

### 4.3 Partitioner 的前向/反向切分

```python
# torch/_functorch/partitioners.py
def min_cut_partition(joint_module, ..., activation="store_true"):
    # 1. 构建 joint graph 的成本模型
    # 2. 标记哪些 tensor 需要在 forward 保存（作为 backward 的输入）
    # 3. 运行 min-cut 算法：在"保存更多激活"和"重算"之间找最优解
    # 4. 输出 fw_module 和 bw_module
    ...
```

核心权衡：
- **保存激活 → 更多显存，更少计算**
- **重算（recompute）→ 更少显存，更多计算**
- `torch.compile` 的 `mode="reduce-overhead"` 倾向于保存更多激活

### 4.4 Functionalization 在 AOTAutograd 中的作用

在 trace 之前，AOTAutograd 通过 functionalization 将 in-place op 替换为 functional 版本：

```python
# 原始代码
y.add_(1)  # in-place

# Functionalization 后
y = y.add(1)  # functional
```

这是为了让 FX graph 成为无副作用的纯函数图，方便后续的 partitioner 和 compiler 做分析和变换。

## 5. 和已有笔记的连接

```
dynamo/         — Dynamo 产出 FX graph 后传给 AOTAutograd
fx_graphs/      — AOTAutograd 产出的是 FX graph
inductor/       — Inductor 消费 AOTAutograd 产出的 fw/bw graph
torch.compile/  — AOTAutograd 是 torch.compile 的中间层
autograd/       — Eager Autograd 是理解 AOTAutograd 的前置知识
dispatcher/     — Decomposition 过程中使用 Dispatcher
functionalization/ — AOTAutograd 依赖 functionalization 消除 mutation
```

## 6. 常见坑点

- **AOTAutograd trace 时需要真实输入**（example_inputs）来做形状推导。
- **遇到 data-dependent control flow 会 graph break**，AOTAutograd 无法处理。
- **Partitioner 的 min-cut 选型可能在极端情况下退化**：全部重算或全部保存。
- **自定义 autograd.Function 在 AOTAutograd 中可能无法正确 trace**，需要注册 decomposition。
- **AOTAutograd 的 joint graph 可能会非常大**，导致编译内存炸裂（OOM），对大模型尤其明显。
- `torch.compile` 默认使用 AOTAutograd 作为 backend 的中间层；指定 `backend="eager"` 会跳过它。

## 7. 阅读源码时建议搜索的关键词

```bash
# AOTAutograd 主入口
rg -n "def aot_function" torch/_functorch/aot_autograd.py

# joint graph 的 partition 逻辑
rg -n "def min_cut_partition" torch/_functorch/partitioners.py

# decomposition 注册
rg -n "register_decomposition" torch/_decomp/decompositions.py | head -10

# derivatives.yaml 中 autograd 公式
rg -n "name: add\\.Tensor" tools/autograd/derivatives.yaml

# functionalization 的启用
rg -n "functionalize" torch/_functorch/aot_autograd.py

# 内存规划（保存/重算）的决策
rg -n "save_for_backward|recompute" torch/_functorch/partitioners.py
```
