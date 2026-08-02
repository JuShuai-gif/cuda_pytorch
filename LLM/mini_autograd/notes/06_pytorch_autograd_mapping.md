# 06 与 PyTorch Autograd 的对应关系

## 对照表

| 本项目 | PyTorch | 说明 |
| --- | --- | --- |
| `Tensor` | `torch.Tensor` | 数据 + 梯度 + 图信息 |
| `grad_fn` | `Tensor.grad_fn` | 指向产生该张量的 Function |
| `Function` | `torch.autograd.Function` | 算子的前向 + 局部梯度 |
| `backward()` | `Tensor.backward()` | 反向传播入口 |
| `Tensor.zeros/ones/randn` | `torch.zeros/ones/randn` | 工厂方法 |
| `no_grad()` | `torch.no_grad()` | 关闭梯度追踪 |
| `enable_grad()` | `torch.enable_grad()` | 重新开启梯度 |
| `set_grad_enabled()` | `torch.set_grad_enabled()` | 全局开关 |
| `detach()` | `Tensor.detach()` | 切断梯度流 |
| `zero_grad()` | `Tensor.zero_grad()` | 清空梯度 |
| `nn.Parameter` | `torch.nn.Parameter` | 默认需要梯度的叶子张量 |
| `nn.Module` | `torch.nn.Module` | 参数/子模块容器 |
| `nn.Linear` | `torch.nn.Linear` | 全连接层 |
| `nn.ReLU/Sigmoid/Tanh` | `torch.nn.ReLU/Sigmoid/Tanh` | 激活函数 |
| `nn.MSELoss` | `torch.nn.MSELoss` | 均方误差 |
| `nn.CrossEntropyLoss` | `torch.nn.CrossEntropyLoss` | 交叉熵 |
| `optim.SGD` | `torch.optim.SGD` | 随机梯度下降 |

## 关键差异

| 维度 | PyTorch | 本项目 |
| --- | --- | --- |
| 后端 | C++/CUDA | 纯 NumPy |
| 存储 | 共享内存，支持视图 | 数据总是拷贝/独立数组 |
| 非叶子梯度 | 默认不保存，需 `retain_grad()` | 所有参与图的张量都保存 `grad` |
| 重复 backward | 默认报错，需 `retain_graph=True` | 自动支持，梯度累加 |
| 图生命周期 | backward 后默认释放 | 一直保留在 `grad_fn` 引用中 |
| 数据类型 | 多种 dtype + device | 固定 float64 |
| in-place 操作 | 支持并有版本检测 | 不支持（简单起见） |
| 二阶梯度 | 支持 | 暂不支持 |
| 稀疏/量化/算子融合 | 支持 | 不支持 |

### 关于非叶子梯度

PyTorch 默认只给**叶子节点**保存 `grad`；中间节点要 `t.retain_grad()` 才能读到。
本项目为了教学直观，**所有**参与图的张量都会累积 `grad`（包括中间节点和输出）。
在对比测试里我们只比较叶子节点的梯度，结果与 PyTorch 完全一致。

### 关于 detach 与复制

PyTorch 的 `detach()` 返回一个**共享底层存储**但 `requires_grad=False` 的张量，
`data` 和 `detach()` 结果改动互相可见。本项目为了简单和安全，`detach()` 直接拷贝数据
（`data.copy()`），避免别名带来的意外副作用。两者在"切断梯度流"这一核心语义上一致。

### 关于重复 backward

PyTorch 默认 `backward()` 后释放计算图，再调用一次会报错。本项目保留图，并在每次
`backward()` 时先快照、清零再恢复节点梯度，因此可以反复调用，且叶子梯度会正确累加。

## 对比测试

`tests/test_compare_pytorch.py` 对每个核心算子做双框架对比：

```python
# mini_autograd
mx = Tensor(x0, requires_grad=True)
ops.sigmoid(mx).sum().backward()

# PyTorch
tx = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
torch.sigmoid(tx).sum().backward()

np.testing.assert_allclose(mx.grad, tx.grad.numpy(), rtol=1e-5, atol=1e-6)
```

覆盖：逐元素算子、matmul、broadcast、reshape/transpose、sum/mean、Linear、两个
loss、SGD 参数更新、detach。运行：

```bash
pytest -v
```

如果未安装 torch，这部分测试会自动 skip，其余测试不受影响。
