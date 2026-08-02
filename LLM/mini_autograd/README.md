# mini_autograd

一个**从零实现**的微型 Autograd（自动微分）引擎，用来帮助你理解 PyTorch 的
`torch.autograd` 核心原理。核心逻辑只用 Python + NumPy 手写，**不依赖**
`torch.autograd`、micrograd、tinygrad 等自动微分库。

```python
from mini_autograd import Tensor

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a * b + a
c.backward()

print(a.grad)   # 4.0   (dc/da = b + 1)
print(b.grad)   # 2.0   (dc/db = a)
```

## 目标：理解什么？

- Tensor 如何保存数据与梯度
- 计算图如何在前向执行时动态构建
- 每个算子如何定义"局部梯度"
- 反向传播如何按**拓扑顺序**执行
- 链式法则如何把梯度从输出传回叶子
- 梯度如何在多条路径上**累加**
- 广播的反向梯度如何**求和还原**
- `requires_grad` / `grad_fn` / 叶子节点 / 非叶子节点的含义
- `no_grad`、`detach`、`zero_grad` 的原理
- 参数如何通过 SGD 梯度下降更新

## Autograd 整体原理（30 秒版）

1. **前向即建图**：每次调用 `x * y`、`x @ w` 等算子时，结果 Tensor 会记录
   `grad_fn`，指向一个 `Function` 节点，节点再指向它的输入——有向无环图就这样
   逐行搭好了。
2. **反向拓扑排序**：`backward()` 从输出出发，把图里的 Function 节点排成一个
   "输出在前、叶子在后"的顺序，保证每个节点的上游梯度先汇总完再处理。
3. **链式法则**：每个 `Function.backward(upstream)` 返回对每个输入的局部梯度
   × 上游梯度，累加（绝不覆盖）到输入 Tensor 上。

## 目录结构

```text
mini_autograd/
├── README.md
├── requirements.txt
├── setup.py
├── mini_autograd/
│   ├── __init__.py          # 包入口
│   ├── tensor.py            # Tensor 类 + 反向引擎 backward()
│   ├── function.py          # Function 基类（图节点）
│   ├── ops.py               # 全部算子及其 backward
│   ├── utils.py             # unbroadcast 等工具
│   ├── grad_mode.py         # no_grad / enable_grad / set_grad_enabled
│   ├── nn/
│   │   ├── module.py        # Module 基类（参数/子模块注册）
│   │   ├── parameter.py     # Parameter
│   │   ├── linear.py        # Linear
│   │   ├── activation.py    # ReLU / Sigmoid / Tanh
│   │   └── loss.py          # MSELoss / CrossEntropyLoss
│   └── optim/
│       ├── __init__.py
│       └── sgd.py           # SGD（momentum / weight_decay）
├── examples/                # 6 个教学示例
├── tests/                   # pytest 测试（含与 PyTorch 的对比）
└── notes/                   # 6 篇中文原理讲解
```

## 安装

```bash
cd /home/ghr/code/cuda_pytorch/LLM/mini_autograd

# 推荐用 uv 或 pip 创建虚拟环境
uv venv .venv && source .venv/bin/activate

pip install -e .          # 安装 numpy 依赖
pip install -r requirements.txt   # 额外安装 pytest 与 torch（对比测试用）
```

## 测试

```bash
cd /home/ghr/code/cuda_pytorch/LLM/mini_autograd
pytest -v
```

- 不装 torch：纯功能测试照常运行，对比测试自动 skip。
- 装 torch：全部测试运行，每个算子与 PyTorch 在 `rtol=1e-5, atol=1e-6` 下对齐。

## 示例

```bash
cd /home/ghr/code/cuda_pytorch/LLM/mini_autograd
python examples/01_scalar_autograd.py        # 标量：a*b+a 的完整图遍历
python examples/02_tensor_operations.py      # 张量算子与梯度累加
python examples/03_broadcast_backward.py     # 广播梯度还原
python examples/04_linear_regression.py      # 拟合 y = 3x + 2
python examples/05_mlp_classification.py     # 两层 MLP 分类
python examples/06_compare_with_pytorch.py   # 与 PyTorch 逐项对比
```

## 已实现功能

- 算子：`add sub mul div neg pow matmul sum mean reshape transpose exp log relu
  sigmoid tanh`，全部支持广播与 Python 运算符重载
- 反向引擎：标量/非标量 `backward(gradient)`、拓扑排序、梯度累加、重复 backward
- 梯度模式：`no_grad` / `enable_grad` / `set_grad_enabled` / `detach`
- 神经网络：`Module` / `Parameter` / `Linear` / `ReLU` / `Sigmoid` / `Tanh` /
  `MSELoss` / `CrossEntropyLoss`
- 优化器：`SGD`（支持 momentum、weight decay），更新在 `no_grad` 中执行
- 与 PyTorch 的逐算子对比测试

## 暂未实现功能

- `torch.optim` 其余优化器（Adam 等）
- 卷积、池化、Embedding、Dropout 等算子
- `getitem` / 视图 / in-place 操作
- 数据类型（固定 float64）
- 二阶梯度

## 阅读顺序建议

1. `notes/01_autograd_overview.md` —— 总览
2. `examples/01_scalar_autograd.py` + `notes/02_computation_graph.md` —— 图怎么建
3. `mini_autograd/function.py` + `ops.py` —— 算子的 forward/backward 模式
4. `mini_autograd/tensor.py` 的 `backward()` + `notes/03_backward_topology.md` —— 引擎核心
5. `notes/04_broadcast_gradient.md` —— 广播梯度
6. `mini_autograd/nn/` + `optim/` + `notes/05_nn_and_optimizer.md` —— 神经网络
7. `notes/06_pytorch_autograd_mapping.md` + `tests/test_compare_pytorch.py` —— 与 PyTorch 对照
