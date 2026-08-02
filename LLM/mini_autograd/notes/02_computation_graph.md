# 02 动态计算图

## 什么是动态计算图？

"动态"的意思是：**计算图不是预先定义好的，而是随着代码逐行执行、逐步搭建出来的。**

```python
z = x * y + x
```

这一行代码执行完毕后，内存里已经存在下面这个结构：

```
x ──┐
    ├── Mul ──> t ──┐
y ──┘              ├── Add ──> z
x ─────────────────┘
```

每个圆角框都是一个 `Function` 节点，它记录了：

- 输入是哪几个 Tensor（`inputs`）
- 输出是哪个 Tensor（`output`）
- 反向时需要的信息（`saved`，比如输入的值、形状）

在 `mini_autograd` 中，这个图是**通过引用来连接**的：

```python
z.grad_fn   # <Add>，指向 z 的父节点
z.grad_fn.inputs   # (t, x)，Add 的两个输入
t.grad_fn   # <Mul>
```

只要顺着 `tensor.grad_fn.inputs[].grad_fn...` 递归，就能遍历整张图。

## 为什么说"执行即建图"？

看 `_from_function`（`tensor.py`）的实现：

```python
def _from_function(fn, *inputs):
    raw_out = fn.forward(*[t.data for t in tensors])   # 1. 算前向
    needs_grad = is_grad_enabled() and any(t.requires_grad for t in tensors)
    out = Tensor(raw_out, requires_grad=needs_grad)
    if needs_grad:
        fn.inputs = tensors    # 2. 记下输入
        out.grad_fn = fn       # 3. 输出指向父节点
    return out
```

所以每调用一次 `x * y`、`x + 1`、`x @ w`，就顺带往图上添加了一个节点。
**建图和计算是同一件事，不需要单独的"编译"步骤。**

## 动态 vs 静态

| | 动态图（PyTorch / 本项目） | 静态图（TensorFlow 1.x） |
| --- | --- | --- |
| 建图时机 | 运行时逐行 | 先定义再执行 |
| 支持 if/for 分支 | 自然支持 | 困难 |
| 调试 | 方便，断点即所见 | 需要先编译 |
| 性能优化空间 | 小 | 大（图优化、算子融合） |

## requires_grad 与 grad_fn 的关系

- `requires_grad=True`：表示该 Tensor 需要计算梯度，参与建图。
- `grad_fn`：表示该 Tensor 是某个算子算出来的（非叶子）。

规则很简单：

```
输出.requires_grad = 梯度开关打开 and 任一输入.requires_grad
```

```python
x = Tensor(2.0)                      # 叶子，不需要梯度
y = x * x                            # 所有输入都不需要梯度
y.requires_grad                      # False，不建图

a = Tensor(2.0, requires_grad=True)
b = a * 2.0                          # 有输入需要梯度
b.requires_grad                      # True
b.grad_fn                            # <Mul>
b.is_leaf                            # False
```

## no_grad 如何让"不建图"生效？

`no_grad` 只是把全局开关 `_grad_mode.enabled` 暂时关掉（`grad_mode.py`）：

```python
with no_grad():
    y = model(x)     # _from_function 检测到开关关闭 -> 不设 grad_fn
```

参数更新时我们希望 `p = p - lr * g` 这行**不进入计算图**（否则每次更新都留下一个节点，
内存越积越多），所以 `SGD.step()` 也在 `no_grad()` 里执行。
