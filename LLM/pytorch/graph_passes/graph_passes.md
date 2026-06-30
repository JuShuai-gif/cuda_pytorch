# FX Graph Passes 图优化技术源码分析

> 源码路径: `torch/fx/graph.py` (Graph + DCE), `torch/fx/passes/` (optimization passes)
> 核心 Pass: DCE (`graph.py:2677`), CSE (`dialect/common/cse_pass.py`), const_fold (`experimental/const_fold.py`)
> Pass 基础设施: `passes/infra/pass_manager.py` (新版 PassManager), `passes/infra/pass_base.py` (PassBase)

## 0. 一句话总览

FXGraph Pass = 对计算图的**迭代变换**：DCE 消除无用的 dead node，CSE 去重公共子表达式，constant folding 预计算常量子图，pattern-based replace 做算子融合。组合这些 pass → 优化后的图 → 更少的内存分配 + 更少的 kernel launch。

---

## 一、Dead Code Elimination (DCE)

`torch/fx/graph.py:2677`:

```python
class Graph:
    def eliminate_dead_code(self):
        """
        从输出节点反向遍历，标记所有被使用的节点。
        未被标记的节点（无用户且无副作用）被删除。
        """
        # 1. 标记 output + placeholder 节点
        # 2. 从所有 output + effect 节点反向 BFS
        # 3. 删除未遍历到的节点
```

**关键**: DCE 以 output 节点和 effect 节点（如 `print`、`torch.save`）为根，反向遍历图。只有被引用到的节点保留，其余删除。

### 为什么需要 DCE:
- symbolic_trace 可能捕获了不需使用的常量/计算
- 融合后的图可能留有不再使用的节点
- 图变换过程中产生孤儿节点

---

## 二、Common Subexpression Elimination (CSE)

`dialect/common/cse_pass.py`:

```python
class CSEPass(PassBase):
    def call(self, gm):
        """对图中所有具有相同输入和 target 的节点去重"""
        # 哈希 key = (target, args_hash, kwargs_hash)
        # 相同 key → 删除重复节点 → 将所有 user 重定向到第一个
```

**示例**:

```
# Before CSE:
a = x + y       # add node 1
b = x + y       # add node 2 (identical to 1)
c = a * b

# After CSE:
a = x + y
c = a * a       # b 被替换为 a
```

---

## 三、Constant Folding

`experimental/const_fold.py`:

```python
def split_const_subgraphs(gm):
    """将图中只依赖常量的子图分离出来，在第一次调用时执行一次，结果存入常量"""
```

**原理**:
```
# Before:
x = input
c = torch.tensor([1,2,3])
y = c * 2          # 常量计算 — 每次 forward 都重算
z = x + y

# After constant folding:
x = input
y = torch.tensor([2,4,6])  # 预计算
z = x + y
```

---

## 四、Pattern-Based Replacement (subgraph_rewriter)

`torch/fx/subgraph_rewriter.py`:

```python
matches = replace_pattern(gm, pattern_gm, replacement_gm)
```

**原理**: 在目标图中搜索与 `pattern_gm` 拓扑匹配的子图，替换为 `replacement_gm`。匹配基于**拓扑结构 + 算子类型**，不关心具体数值。

```python
# 模式: Conv → BN → ReLU
pattern = ...
# 替换为: FusedConvBNReLU
replacement = ...

replace_pattern(gm, pattern, replacement)
```

---

## 五、Pass 基础设施

### 5.1 新版 PassManager (`passes/infra/pass_manager.py:154`)

```python
pm = PassManager(
    passes=[CSEPass(), FusePass()],
    steps=2,  # 迭代多轮
    run_checks_after_each_pass=True,
)
result = pm(gm)
# result.graph_module — 优化后的图
# result.modified — 是否发生变化
```

**PassManager 自动调度**:
- 如果定义了 constraint（如 `this_before_that_pass_constraint`），自动拓扑排序
- `steps` 控制迭代轮数（多轮 pass 可消除前一轮产生的优化机会）

### 5.2 PassBase 抽象 (`passes/infra/pass_base.py`)

```python
class PassBase(abc.ABC):
    def __call__(self, gm):
        self.requires(gm)    # 前置检查
        res = self.call(gm)  # 核心逻辑（子类实现）
        self.ensures(gm)     # 后置检查
        return res
```

### 5.3 Transformer 模式 (`interpreter.py:518`)

```python
class Transformer(Interpreter):
    def call_function(self, target, args, kwargs):
        if target == torch.add:
            return torch.mul(*args, **kwargs)  # 替换 add → mul
        return super().call_function(target, args, kwargs)

gm_transformed = Transformer(gm).transform()
```

Transformer 是**访问者模式**的图重写器，遍历所有节点，允许按 op 类型定制替换逻辑。

---

## 六、常见优化流水线

### Inference optimization (`experimental/optimization.py:329`):

```python
def optimize_for_inference(gm):
    # 1. fuse conv+bn → conv_bn
    # 2. fuse conv+bn+relu → conv_bn_relu
    # 3. fuse linear+relu → linear_relu
    # 4. fuse conv+relu → conv_relu
    # 5. remove dropout
    # 6. replace with MKL-DNN layout
    # 7. eliminate dead code
    return optimized
```

### 带量化的优化流水线:

```
trace → DCE → const_fold → fuse conv+bn → CSE → insert quant/dequant → DCE
```

---

## 七、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `Graph.eliminate_dead_code` | `torch/fx/graph.py` | 2677 |
| `CSEPass` | `torch/fx/passes/dialect/common/cse_pass.py` | — |
| `PassManager` (新版) | `torch/fx/passes/infra/pass_manager.py` | 154 |
| `PassBase` | `torch/fx/passes/infra/pass_base.py` | — |
| `Transformer` | `torch/fx/interpreter.py` | 518 |
| `replace_pattern` | `torch/fx/subgraph_rewriter.py` | — |
| `optimize_for_inference` | `torch/fx/experimental/optimization.py` | 329 |
| `const_fold` | `torch/fx/experimental/const_fold.py` | — |
| `ShapeProp` | `torch/fx/passes/shape_prop.py` | — |
| `reinplace` | `torch/fx/passes/reinplace.py` | — |
| `split_module` | `torch/fx/passes/split_module.py` | — |
| `fuse (conv+bn+relu)` | `torch/fx/experimental/optimization.py` | 76 |

---

## 八、可借鉴的工程技巧

1. **Pass 管道化**: 每个 pass 独立、可组合。类比：编译器 IR 优化 pass (LLVM pass pipeline)。

2. **DCE 反向遍历**: 从 output 反向 BFS → 只保留 reachable 节点。类比：GC 的 mark-sweep（标记可达对象）。

3. **CSE 哈希关键**: `(target, args_hash, kwargs_hash)` 作为去重 key → O(1) 查找重复项。

4. **Transformer 访问者**: 节点遍历 + 按 type 分发 → 无需修改图结构即可定制替换逻辑。类比：AST visitor。

5. **PassManager 自动排序**: constraint-based 拓扑排序 → 用户只需声明依赖关系，不用手动编排顺序。
