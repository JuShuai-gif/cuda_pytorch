# functorch: vmap、grad 与函数变换的源码视角

> Python 端: `torch/_functorch/`、`torch/func/`
> C++ 端: `aten/src/ATen/functorch/`
> Dispatch Key: `c10/core/DispatchKey.h` (FuncTorchBatched、FuncTorchGradWrapper 等)

## 0. 一句话总览

functorch 提供函数式变换（vmap、grad、vjp、jacfwd 等），核心思想是用 **BatchedTensor** 等 wrapper tensor 和 **DispatchKey 拦截**改写算子行为，而非使用 Python for-loop 或复杂的图变换。vmap 通过 BatchedTensor 在 dispatch 层面自动传播 batch 维度，grad 通过 autograd 机制计算梯度，两者可以任意组合。

## 1. 最小例子

```python
import torch
from torch.func import vmap, grad

def loss_fn(w, x):
    return (w * x).sum().sin()

w = torch.randn(8)
xs = torch.randn(32, 8)  # 32 个样本

# 对每个样本计算梯度: 等价于 for-loop 但在算子层面做向量化
per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0))(w, xs)
print(per_sample_grad.shape)  # [32, 8]
```

## 1.5 实战例子

### 1.5.1 Vmap vs for-loop 性能对比

在实际训练中，per-sample gradients 用 vmap 实现 vs for-loop 的性能差异：

```python
import torch
from torch.func import vmap, grad
import time

def loss_fn(w, x, y):
    return ((x @ w) - y).pow(2).mean()

w = torch.randn(8, requires_grad=True)
xs = torch.randn(256, 8)  # 256 个样本
ys = torch.randn(256)

# For-loop 版本
def per_sample_grad_loop(w, xs, ys):
    grads = []
    for i in range(len(xs)):
        g, = torch.autograd.grad(loss_fn(w, xs[i], ys[i]), w)
        grads.append(g)
    return torch.stack(grads)

# Vmap 版本
per_sample_grad_vmap = vmap(grad(loss_fn), in_dims=(None, 0, 0))

t0 = time.time()
g_loop = per_sample_grad_loop(w, xs, ys)
t1 = time.time()
g_vmap = per_sample_grad_vmap(w, xs, ys)
t2 = time.time()

print(f"For-loop: {t1-t0:.3f}s")
print(f"Vmap: {t2-t1:.3f}s")
print(f"Speedup: {(t1-t0)/(t2-t1):.1f}x")
# 实际测试中, vmap 通常比 for-loop 快 5-20x
```

### 1.5.2 用 jacfwd/jacrev 计算完整 Jacobian

在科学计算或对偶学习场景中需要完整 Jacobian 矩阵：

```python
import torch
from torch.func import jacfwd, jacrev, vmap

def f(x):
    return torch.sin(x).sum(dim=1)

x = torch.randn(8, 4)

# 前向模式 Jacobian (适合 output_dim >> input_dim)
J_fwd = jacfwd(f)(x)
print("jacfwd shape:", J_fwd.shape)  # [8, 8, 4]

# 反向模式 Jacobian (适合 input_dim >> output_dim)
J_rev = jacrev(f)(x)
print("jacrev shape:", J_rev.shape)

# 组合: 计算 Hessian
def loss(x):
    return torch.sin(x).sum()

hessian = jacfwd(jacrev(loss))(torch.randn(4))
print("Hessian shape:", hessian.shape)  # [4, 4]
```

### 1.5.3 排查缺失 batching rule 时的 fallback 行为

当使用自定义 op 或稀有 op 在 vmap 中时，检查是否走 fallback：

```python
import torch
from torch.func import vmap

lib = torch.library.Library("myops", "DEF")
lib.define("special_op(Tensor x) -> Tensor")

@torch.library.impl("myops::special_op", "CPU")
def special_op_cpu(x):
    # 一些复杂操作，无 batching rule
    return x * 2

def f(x):
    return torch.ops.myops.special_op(x)

# 尝试 vmap, 如果无 batching rule 会走 for-loop fallback
try:
    result = vmap(f)(torch.randn(8, 4))
    print("vmap 结果 shape:", result.shape)
except Exception as e:
    print(f"vmap 失败: {e}")
    # 可能报错: "Batching rule not registered"

# 通过设置环境变量检测 fallback:
# TORCH_SHOW_DISPATCH=1 python script.py 显示每个 op 的 dispatch 决策
```

实际排查时，设置 `TORCH_SHOW_DISPATCH=1` 可看到某个 op 是否命中 `FuncTorchBatched` key。

## 2. 从 Python API 到源码的调用链

```
torch.func.vmap(f)
   |
   v
torch/_functorch/vmap.py (Python transform)
   |
   v  (进入 vmap context 后)
BatchedTensor 包装:
  - 对输入 x, 在 batch 维度上包装为 BatchedTensor(x, batch_dim=0, batch_size=32)
   |
   v
算子调用 (例如 torch.mul(w, x))
   |
   v
Dispatcher: DispatchKeySet 中包含 FuncTorchBatched
   |
   v
Batching rule: aten/src/ATen/functorch/BatchRules*.cpp
   |
   v
规则传播:
  - mul 的 batching rule: 将 w broadcast 到匹配 batch dim
  - sum 的 batching rule: 沿着非 batch 维求和
  - sin 的 batching rule: 逐元素 sin（batch dim 不变）
   |
   v
unwrap BatchedTensor 输出
```

`torch.func.grad` 与 `loss.backward()` 对比：

```
loss.backward():
  -> forward 构建 Autograd Node
  -> Tensor.grad 被填充

torch.func.grad(f):
  -> 函数变换: 返回一个新函数
  -> 新函数内部使用 autograd.grad 计算梯度
  -> 输出梯度（而不是填充 .grad 属性）
  -> 更函数式: 无副作用
```

## 3. 核心源码文件

```
torch/_functorch/                       # 函数变换核心实现
  ├── vmap.py                          # vmap 实现
  ├── grad.py                          # grad 实现
  ├── jac.py                           # jacfwd/jacrev
  ├── eager_transforms.py              # transforms 集成
  ├── partitioners.py                  # 与 partitioner 的关系
  └── aot_autograd.py                  # transform + AOTAutograd
torch/func/                            # 用户 API 入口
  ├── __init__.py
  └── functional_call.py
aten/src/ATen/functorch/               # C++ batching rules
  ├── BatchRulesViews.cpp              # view op 的 batching rule
  ├── BatchRulesOps.cpp                # 算子的 batching rule
  ├── BatchRulesRandom.cpp             # 随机数的 batching rule
  ├── BatchRulesPooling.cpp            # 池化的 batching rule
  ├── BatchRulesNorm.cpp               # 归一化的 batching rule
  └── BatchRulesExtras.cpp             # 其他
c10/core/DispatchKey.h                 # FuncTorch 相关 DispatchKey
```

## 4. 关键机制源码解读

### 4.1 BatchedTensor 与 DispatchKey 拦截

BatchedTensor 是一个包装 tensor，它在 `__torch_function__` 和 `__torch_dispatch__` 层面拦截所有算子调用：

```python
class BatchedTensor(torch.Tensor):
    def __init__(self, value, batch_dim, batch_size):
        self.value = value        # 原始 tensor
        self.batch_dim = batch_dim  # batch 维度位置
        self.batch_size = batch_size
```

Dispatcher 遇到 `FuncTorchBatched` key 时，查找对应的 batching rule。如果有，调用 batching rule；如果没有，报错 `"Batching rule not registered"`。

### 4.2 batching rule 示例

```cpp
// aten/src/ATen/functorch/BatchRulesOps.cpp
// mul 的 batching rule 简化示意
Tensor mul_batching_rule(const Tensor& self, const Tensor& other) {
    auto self_ = unwrap_batched(self);    // remove batch dim 包装
    auto other_ = unwrap_batched(other);
    // 自动 broadcast 以对齐 batch dim
    auto result = at::mul(self_, other_);
    return wrap_batched(result, self.batch_dim());
}
```

缺失 batching rule 时，functorch 使用 **fallback**：通过 `for-loop` 逐 batch 计算，丧失了 vmap 的性能优势。`rg "FALLBACK_BATCHING" aten/src/ATen/functorch/` 可查看 fallback 逻辑。

### 4.3 vmap + grad 的组合

```python
per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0))(w, xs)
```

执行顺序：
1. `grad(loss_fn)` 创建一个计算梯度的新函数
2. `vmap(...)` 将这个梯度函数向量化
3. 实际执行时：
   - vmap 将 `w` 包装（`in_dims=(None,)`: w 不增加 batch dim）
   - vmap 将 `xs` 包装（`in_dims=(0,)`: xs 的 dim 0 视为 batch dim）
   - 内部 `grad(loss_fn)` 为每个 batch 元素计算梯度
   - 最终输出 `[32, 8]` 的 per-sample gradients

### 4.4 functorch 与 AOTAutograd、Functionalization 的关系

- **AOTAutograd** 利用 functorch 的 transform 来做 joint graph tracing
- **Functionalization** 在 AOTAutograd 中启用，消除 mutation 使得 transform 可以正确 trace
- `torch._functorch.eager_transforms.py` 中实现了 eager 版本的 grad/vmap，作为 AOTAutograd 的 fallback

```python
# torch/_functorch/eager_transforms.py
def grad_impl(f, argnums=0):
    # 创建 autograd.grad 的 wrapper
    def grad_wrapper(*args):
        return autograd.grad(f(*args), args[argnums])
    return grad_wrapper
```

## 5. 和已有笔记的连接

```
autograd/                     — functorch.grad 基于 autograd 实现
torch.compile/                — functorch transform 在 AOTAutograd 中使用
dispatcher/                   — FuncTorchBatched dispatch key 是关键机制
functionalization/            — AOTAutograd 结合 functionalization 和 transform
aot_autograd/                 — AOTAutograd 使用 functorch 做 joint graph trace
dynamo/                       — Dynamo trace 时需要处理 functorch transform
```

## 6. 常见坑点

- **不是所有 op 都有 batching rule**，缺失时退化为 for-loop，vmap 失去了性能意义。
- **vmap 嵌套不能无限深度**，有 `_C._functorch._vmap_max_level` 限制。
- **random op 在 vmap 下行为特殊**：同一 batch 中不同样本可能得到不同随机值（`BatchRulesRandom.cpp` 控制）。
- **grad 和 vmap 的组合顺序影响结果形状**：`vmap(grad(f))` vs `grad(vmap(f))` 语义不同。
- **BatchedTensor 不能直接 torch.save 序列化**，需要先 unwrap。
- **functorch 和旧版 `torch.vmap` API 有区别**，推荐使用 `torch.func.vmap`。

## 7. 阅读源码时建议搜索的关键词

```bash
# 查看所有已注册的 batching rule
rg -n "batching_rule" aten/src/ATen/functorch/BatchRulesOps.cpp

# 查看 fallback batching rule（没有专用 rule 时的兜底）
rg -n "FALLBACK_BATCHING" aten/src/ATen/functorch/

# DispatchKey 定义
rg -n "FuncTorch" c10/core/DispatchKey.h

# vmap transform 入口
rg -n "def vmap" torch/_functorch/vmap.py

# grad transform 入口
rg -n "def grad" torch/_functorch/grad.py

# 与 AOTAutograd 的集成
rg -n "functorch" torch/_functorch/aot_autograd.py
```
