# 05 神经网络模块与优化器

## 训练一个模型的标准循环

无论框架多复杂，训练的核心就是这五步：

```python
for step in range(epochs):
    pred  = model(x)            # 1. 前向传播（同时建图）
    loss  = loss_fn(pred, y)    # 2. 计算损失
    opt.zero_grad()             # 3. 清空梯度（避免累加）
    loss.backward()             # 4. 反向传播，计算每个参数的梯度
    opt.step()                  # 5. 参数更新：p = p - lr * grad
```

下面看 `mini_autograd` 如何组织这些模块。

## Parameter：可训练的张量

```python
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)   # 默认需要梯度
        self.is_leaf = True
```

叶子节点 + `requires_grad=True`，这样每个参数都会在 `backward` 时累积出自己的梯度。

## Module：把参数组织起来

`Module` 用 `__setattr__` 钩子自动注册子参数和子模块：

```python
class Module:
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)
```

这样只要 `self.fc1 = Linear(...)`，`self.fc1.weight` 就会自动出现在
`model.parameters()` 里：

```python
def parameters(self):
    for p in self._parameters.values():
        yield p
    for m in self._modules.values():
        yield from m.parameters()
```

`zero_grad()`、`train()`、`eval()` 都会递归到所有子模块。

## Linear 层

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Parameter(np.random.uniform(-k, k, (out_features, in_features)))
        self.bias   = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        y = ops.matmul(x, ops.transpose(self.weight))
        return y + self.bias if self.bias is not None else y
```

权重形状和 PyTorch 一致：`(out_features, in_features)`，前向是 `x @ W.T + b`。

## 损失函数

### MSELoss

```python
def forward(self, pred, target):
    diff = pred - target
    return (diff * diff).mean()     # 或用 .sum()
```

反向由 autograd 自动完成：`d/dpred = 2*(pred-target)/N`。

### CrossEntropyLoss

用可微算子逐步搭建：softmax 分母的 log 形式更稳定，而且不需要实现带梯度的 `max/gather`：

```python
shifted    = logits - row_max            # 数值稳定，row_max 是常数，不影响梯度
log_softmax = shifted - log(exp(shifted).sum(axis=1, keepdims=True))
loss = -(log_softmax * onehot).sum(axis=1).mean()
```

## 优化器 SGD

```python
class SGD:
    def step(self):
        with no_grad():                  # 更新不能进计算图！
            for p in self.params:
                g = p.grad
                if self.weight_decay:
                    g = g + self.weight_decay * p.data
                if self.momentum:
                    v = momentum * v + g       # 动量缓存
                    g = v
                p.data = p.data - self.lr * g
```

要点：

- **`no_grad()`**：`p.data = p.data - lr*g` 这行如果在图里，每步都会残留一个节点，
  图会越攒越大；关掉梯度后只是普通 NumPy 操作。
- **`zero_grad()` 必须在 backward 之前**：否则梯度会在多次 backward 间累加。
- 动量相当于让"上一次的更新方向"也参与本次移动，能加速并平滑震荡。

## 一个完整的 MLP

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

训练后打印准确率即可（见 `examples/05_mlp_classification.py`）。
