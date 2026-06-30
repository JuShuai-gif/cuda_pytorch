# FX Graph Passes 图优化技术源码分析

> 源码: `torch/fx/graph.py:2677` (DCE), `torch/fx/experimental/const_fold.py` (const fold)
> Pass 基础设施: `passes/infra/pass_manager.py:154`, `passes/infra/pass_base.py`
> Pattern 替换: `torch/fx/subgraph_rewriter.py` (replace_pattern)

## 0. 一句话总览

FX Graph Pass = 对计算图的**迭代变换**。DCE 从 output 反向 BFS 消除死节点，const_fold 分离出常量子图并预执行，CSE 基于 `(target, args_hash, kwargs_hash)` 去重公共子表达式。

---

## 一、`eliminate_dead_code` 源码分析 (`graph.py:2677`)

```python
# graph.py:2677
def eliminate_dead_code(
    self, is_impure_node: Callable[[Node], bool] | None = None
) -> bool:
    self.lint()  # 要求图拓扑有序

    # 决定哪些节点不可消除 (有副作用)
    def has_side_effect(node: Node) -> bool:
        if is_impure_node is not None:
            return is_impure_node(node)    # 自定义判定
        return node.is_impure(impure_random)  # 默认: 有效果则不可消除

    # ★ 核心: 反向遍历节点
    removed_nodes = set()
    for node in reversed(self.nodes):       # :2740 反向迭代
        if not has_side_effect(node) and len(node.users) == 0:  # :2741
            self.erase_node(node)           # :2742 删除节点
            removed_nodes.add(node.name)    # :2743 记录

    return len(removed_nodes) > 0
```

### 为什么反向遍历?

正向遍历时: 删除节点 A → A 的 inputs 的 `users` 可能变成 0 → 但已错过这些 inputs。
反向遍历: 删除节点 A → 它的 inputs 的 `users` 数减少 → 但后续循环会再遇到这些 inputs → 如果此时 `users==0` 就删除。

### `node.is_impure()` 检查什么?

`Node.is_impure()` 检查节点是否有副作用 — 有副作用的节点即使 `users==0` 也不能删除:
- `output` 节点 — 必须保留
- `torch.save` / `print` — 有 observable effect
- 产生随机数的 op — 在 `impure_random=True` 时不可消除

---

## 二、Constant Folding 源码分析 (`experimental/const_fold.py`)

核心函数 `split_const_subgraphs(gm)`:

### 2.1 算法流程

```
1. 遍历图中所有节点, 标记哪些节点只依赖常量
   - placeholder 节点 (输入) → 不是常量
   - get_attr 节点 (parameter/buffer) → 不是常量
   - 如果节点的所有 args 都是常量 → 此节点是常量

2. 将标记为常量的节点分组为互不重叠的子图

3. 对每个常量子图:
   a. 从原图中分割出来 → 创建单独的 GraphModule (const_gm)
   b. const_gm() 执行一次 → 得到结果 tensor
   c. 在原图中创建 get_attr 节点, 指向结果 tensor
   d. 删除原图中的常量节点
```

### 2.2 示例

```
# Before:
x = placeholder          (不是常量)
p = get_attr("weight")  (不是常量)
c1 = torch.tensor([1,2,3])  → 常量
c2 = c1 * 2                 → 常量 (所有输入都是常量)
y = x @ p + c2              → 混合

# After const_fold:
x = placeholder
p = get_attr("weight")
c2 = get_attr("_const_folded_0")  # c2 = [2,4,6] 预计算
y = x @ p + c2
```

---

## 三、CSE (Common Subexpression Elimination)

`torch/fx/passes/dialect/common/cse_pass.py`:

### 3.1 核心数据结构

```python
class CSEPass(PassBase):
    def call(self, gm):
        # 对每个节点计算 hash key
        seen: dict[CSEHash, Node] = {}  # hash → 第一个出现此模式的节点

        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                h = self._hash(node)  # hash = (target, tuple(flatten(args)), ...)
                if h in seen:
                    # 已有重复节点 → 删除当前节点, 将所有 user 重定向到 seen[h]
                    node.replace_all_uses_with(seen[h])
                    gm.graph.erase_node(node)
                else:
                    seen[h] = node
        return PassResult(gm, modified=...)
```

### 3.2 Hash 计算

```python
def _hash(self, node):
    target = node.target        # 如 torch.add
    args = tuple(flatten(node.args))  # 不关心 args 的 node 身份, 只关心实际值
    kwargs = tuple(sorted(node.kwargs.items()))
    return hash((target, args, kwargs))
```

---

## 四、`replace_pattern` (子图模式匹配替换)

`torch/fx/subgraph_rewriter.py`:

### 4.1 核心 API

```python
matches = replace_pattern(gm, pattern_gm, replacement_gm)
```

### 4.2 内部流程

```
1. SubgraphMatcher 在目标图中搜索 pattern_gm 的拓扑匹配
   - 匹配基于: 算子类型 (target) + 拓扑结构 (连接关系)
   - 不关心具体数值

2. 对每个 match:
   a. 创建 replacement_gm 的副本
   b. 将 pattern 的 placeholder 映射到 match 中的实际节点
   c. 将 replacement 的输出连接到原图中 match 的消费者
   d. 删除原图中的 match 节点

3. 返回 List[Match]
```

---

## 五、Pass 基础设施 (`passes/infra/pass_manager.py:154`)

```python
class PassManager:
    def __init__(self, passes, steps=1, ...):
        self.passes = self._resolve_constraints(passes)  # 拓扑排序 pass 依赖
        self.steps = steps  # 多轮迭代

    def __call__(self, gm):
        for step in range(self.steps):
            modified = False
            for p in self.passes:
                result = p(gm)           # PassBase.__call__
                if result.modified:
                    modified = True
                    gm = result.graph_module
            if not modified:
                break                    # 不再变化 → 提前退出
        return PassResult(gm, modified)
```

---

## 六、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `eliminate_dead_code` | `torch/fx/graph.py` | 2677 |
| `erase_node` | `torch/fx/graph.py` | — |
| `split_const_subgraphs` | `torch/fx/experimental/const_fold.py` | — |
| `CSEPass` | `torch/fx/passes/dialect/common/cse_pass.py` | — |
| `PassBase` | `torch/fx/passes/infra/pass_base.py` | — |
| `PassManager` (新版) | `torch/fx/passes/infra/pass_manager.py` | 154 |
| `replace_pattern` | `torch/fx/subgraph_rewriter.py` | — |
| `SubgraphMatcher` | `torch/fx/passes/utils/matcher_utils.py` | — |
| `Transformer` | `torch/fx/interpreter.py` | 518 |

---

## 七、实战常见坑点

*(见历史版本)*
