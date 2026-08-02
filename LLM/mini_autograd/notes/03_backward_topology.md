# 03 反向传播与拓扑排序

## 为什么不能简单递归？

假设图是一个菱形：

```
        x
       / \
      a   b      (a = x*x, b = x*2, y = a + b)
       \ /
        y
```

x 的梯度应该是两条路径的梯度之和：

```
dy/dx = da/dx + db/dx = 2x + 2
```

如果从 y 出发**见一个节点就递归一个**，可能会先走 `a` 这条路径，把 x 的梯度算成了
`2x`；再走 `b` 路径时又得"重新补上" `2`。这种"边算边补"的方式在更复杂的图上很难
保证正确性，甚至可能重复处理同一个节点。

## 拓扑排序要保证什么

**反向拓扑序**保证：

> 一个 Function 节点只有当它所有"下游消费者"都处理完之后才被处理。

这样，当我们处理某个节点时，它的输出 Tensor 上已经**汇总了所有路径传来的上游梯度**，
我们只要一次 `backward()` 就能把完整的梯度继续往前传。

正向图里的依赖方向是"从叶子到输出"，反向则完全相反：

```
正向执行顺序:  x → a, x → b → y
反向处理顺序:  y(Add) → a, b(Mul) → x(叶子)
```

`_reverse_topological_order`（`tensor.py`）用 DFS 实现：

```python
def dfs(t):
    fn = t.grad_fn
    if fn is None or id(fn) in visited:
        return
    visited.add(id(fn))
    for inp in fn.inputs:      # 先递归到所有输入（更靠近叶子）
        dfs(inp)
    order.append(fn)           # 再把当前节点放进去（后序）

dfs(root)
order.reverse()                # 输出侧的节点排在最前面
```

## backward() 的完整流程

1. **校验与播种**：标量输出默认上游梯度为 `1`；非标量必须显式传入 `gradient`。
2. **反向拓扑排序**：收集从输出可达的所有 Function 节点。
3. **清空旧梯度快照**：把每个节点输出上的 `grad` 暂存并清零，避免重复 `backward` 时复用脏数据。
4. **依次处理节点**：
   ```
   up_grad = fn.output.grad                      # 上游（已汇总）
   grads   = fn.backward(up_grad)                # 每个输入的局部梯度 × 上游
   for inp, g in zip(fn.inputs, grads):
       inp._accumulate_grad(g)                   # 累加，绝不覆盖
   ```
5. **恢复快照**：把上次 backward 留下的旧梯度加回去，实现跨次调用累加。

## 链式法则的具体执行

以 `loss = z²`，`z = x*y` 为例：

```
dz/dx = y           dz/dy = x
dloss/dz = 2z
dloss/dx = dloss/dz * dz/dx = 2z * y = 2xy
dloss/dy = dloss/dz * dz/dy = 2z * x = 2xy
```

用代码验证：

```python
x = Tensor(3.0, requires_grad=True)
y = Tensor(4.0, requires_grad=True)
z = x * y
loss = z ** 2
loss.backward()
print(x.grad)   # 2 * 12 * 4 = 96
print(y.grad)   # 2 * 12 * 3 = 72
```

## 每个 Function 的 backward 长什么样

```python
class Mul(Function):
    """z = a * b    dz/da = b,  dz/db = a"""
    def forward(self, a, b):
        self.save_for_backward(a=a, b=b)   # 存下 a, b 供反向用
        return a * b

    def backward(self, grad_output):
        return (grad_output * self.saved["b"],   # 对 a 的梯度
                grad_output * self.saved["a"])   # 对 b 的梯度
```

`backward` 的返回值数量**必须与 forward 的输入数量一致**，一一对应。

## 梯度累加（绝不覆盖）

`x` 在 `y = x*x + x` 中出现两次，它的梯度是 `2x + 1`。原因是引擎对每个输入做的是
**累加**而不是赋值：

```python
def _accumulate_grad(self, g):
    if self.grad is None:
        self.grad = g.copy()
    else:
        self.grad = self.grad + g   # 相加！
```

`x*x` 这条路径贡献 `2x`，`+ x` 这条路径贡献 `1`，加起来才正确。
这也是为什么优化器每次 `step` 前都要 `zero_grad()`——否则上次的梯度会和这次的叠加。
