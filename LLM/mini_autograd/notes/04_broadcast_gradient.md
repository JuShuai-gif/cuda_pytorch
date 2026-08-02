# 04 广播（Broadcast）与梯度还原

## 前向：NumPy 的广播规则

广播让形状不同的张量能够做逐元素运算。规则是**从右往左对齐维度，尺寸要么相等，
要么有一个是 1**。

```python
x.shape = (4, 3)
b.shape = (3,)          # 对齐后视为 (1, 3)
y = x + b               # 形状变成 (4, 3)
```

更复杂的例子：

```python
a.shape = (2, 3, 4, 4)
g.shape = (1, 3, 1, 1)
a * g                   # 形状变成 (2, 3, 4, 4)
```

## 反向：为什么要"求和还原"

前向时 `b` 的每个元素被"拉伸"成 `(4, 3)` 中的一整列。反向时，上游梯度是
`(4, 3)` 形状的，但 `b` 只有 `(3,)`。**无法直接把一个 `(4,3)` 的梯度加到 `(3,)`
的参数上。**

关键是：`b[j]` 影响了 4 个输出（`x[:, j] + b[j]`），所以 `b[j]` 的梯度是这 4 个
输出梯度的**和**：

```
dy/db[j] = sum_i dy[i, j]/db[j] * grad[i, j]
         = sum_i grad[i, j]        (局部梯度为 1)
```

所以反向时要做广播的逆运算：**把梯度沿拉伸过的维度求和**，还原成原始形状。

## unbroadcast 的实现

```python
def unbroadcast(grad, target_shape):
    # 1) 去掉多余的前导维度（这些维度只存在于广播后）
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # 2) 对 target 尺寸为 1 的尾部维度求和
    for axis, (g_size, t_size) in enumerate(zip(grad.shape, target_shape)):
        if t_size == 1 and g_size != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)
```

举例：`target_shape=(3,)`，`grad.shape=(4,3)`：

```
grad.sum(axis=0)  ->  (3,)     # 把广播出来的行维度求和
```

举例：`target_shape=(1,3,1,1)`，`grad.shape=(2,3,4,4)`：

```
sum(axis=0)  ->  (3,4,4)       # 去掉 batch 维
sum(axis=1, keepdims=True)  ->  (1,4,4)
sum(axis=2, keepdims=True)  ->  (1,1,4)
sum(axis=3, keepdims=True)  ->  (1,1,1)
reshape     ->  (1,3,1,1)
```

## 每个算子的 backward 都要 unbroadcast

只要 forward 用了 NumPy 广播，backward 就必须把梯度还原到各输入的原始形状。
例如 `Add.backward`：

```python
def backward(self, grad_output):
    return (unbroadcast(grad_output, self.saved["a_shape"]),
            unbroadcast(grad_output, self.saved["b_shape"]))
```

## 验证示例

```python
x = Tensor(np.ones((4, 3)), requires_grad=True)
b = Tensor(np.ones(3), requires_grad=True)
y = x + b
y.sum().backward()
print(b.grad)   # [4, 4, 4]  —— 每一列被用了 4 次
```

`matmul` 的 batch 广播也一样：`a:(1,3,4)`、`b:(2,4,5)` 做矩阵乘时，`a` 的 batch
维被广播，反向时要把 batch 维的梯度求和，才能得到 `(1,3,4)` 的 `a.grad`。
