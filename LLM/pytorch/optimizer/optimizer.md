# Optimizer 源码分析: SGD & Adam

> 源码路径: `/home/ghr/code/pytorch/torch/optim/sgd.py` (544 行), `torch/optim/adam.py` (~300 行)
> 基类: `torch/optim/optimizer.py` — `Optimizer` 类, `Optimizer.zero_grad`, `Optimizer.step`

## 0. 一句话总览

Optimizer 的核心是 `param_groups` 数据结构：每组参数独立持有 `lr`, `momentum`, `weight_decay` 等配置 + `state` 字典（存储 momentum buffer 等）。`step()` 遍历所有 param_groups，对每个参数执行更新公式。

---

## 一、默认参数初始化 (`SGD.__init__:25`)

```python
defaults = {
    "lr": lr,
    "momentum": momentum,
    "dampening": dampening,
    "weight_decay": weight_decay,
    "nesterov": nesterov,
    "maximize": maximize,
    "foreach": foreach,     # 多 tensor 并行路径
    "differentiable": differentiable,
    "fused": fused,         # CUDA fused kernel
}
super().__init__(params, defaults)
```

`super().__init__()` 将 `params` 组织为 `param_groups` 列表，每个 group 是一个 dict: `{"params": [...], "lr": ..., "weight_decay": ..., ...}`。

---

## 二、SGD 核心算法

### 2.1 公式

带有 momentum 的 SGD:

```
if weight_decay != 0:
    grad = grad + weight_decay * p           (weight decay)

if momentum != 0:
    if momentum_buffer is None:
        buf = grad                           (dampening=0)
    else:
        buf = momentum * buf + (1-dampening) * grad
    grad = buf

p = p - lr * grad
```

### 2.2 Nesterov 动量

Nesterov 的区别在于**先用 momentum 预估下一步位置，在那里算梯度**：

```
if nesterov:
    grad = grad + momentum * momentum_buffer
```

而非标准 momentum 的 `grad = momentum_buffer`。

### 2.3 三种执行路径

`SGD.step()` 根据条件选择路径:

| 路径 | 条件 | 特点 |
|------|------|------|
| **fused** | `fused=True`, CUDA, 非 sparse | 单个 CUDA kernel 完成整个 update，省 intermediate alloc |
| **foreach** | `foreach=True` or 自动选择 | 一次处理整组 tensor（利用 `torch._foreach_*` ops） |
| **single** | 回退 | 逐参数循环 |

`torch._foreach_mul_`, `torch._foreach_add_` 等是对 tensor list 的批量操作，一个 launch 处理多个 tensor。

---

## 三、Adam 核心算法

Adam = SGD momentum + RMSProp：

```
# 1. 更新 biased 矩估计
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2

# 2. 偏差校正
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

# 3. 更新
p = p - lr * m_hat / (sqrt(v_hat) + eps)
```

### 3.1 AdamW 的区别

AdamW 把 weight decay **从梯度中解耦**（Decoupled Weight Decay）：

```python
# AdamW: weight decay 直接作用在参数上，不进动量
p = p * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

# 而不是 Adam 的做法: grad += weight_decay * p
```

这避免了 weight decay 与自适应学习率 `1/sqrt(v_hat)` 的耦合。

### 3.2 `state` 字典

`optimizer.state[p]` 存储每个参数的状态：

```python
state = {
    "step": torch.tensor(0),         # 步数计数器
    "exp_avg": torch.zeros_like(p),  # 一阶矩 m
    "exp_avg_sq": torch.zeros_like(p),  # 二阶矩 v
    # (amax for AMSGrad, etc.)
}
```

**关键**: `state[p]` 中的 buffer 必须与 `p` 在同一 device 上。`state_dict()` / `load_state_dict()` 序列化 state 时会记录 device。

---

## 四、Optimizer 基类关键方法

### 4.1 `zero_grad()`

```python
for group in self.param_groups:
    for p in group["params"]:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
```

或设置 `set_to_none=True`:

```python
p.grad = None  # 更省显存
```

### 4.2 `state_dict()`

序列化 `param_groups`（含 lr/momentum 等配置）和 `state`（含 momentum buffer 等）。

```python
state_dict = {
    "state": {param_id: {"step": ..., "exp_avg": ...}},
    "param_groups": [{"lr": 0.001, "params": [param_id_0, param_id_1]}],
}
```

参数用 `id(p)` 作为 key（稳定，不受 tensor 内容变化影响）。

### 4.3 `load_state_dict()`

- 恢复 `param_groups` 配置
- 恢复 `state` 中的 buffer，按 device 正确放置
- 检查 `param_groups` 的长度和结构是否匹配（不匹配时报错或警告）

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `SGD.__init__` | `torch/optim/sgd.py` | 25 |
| `SGD.step` (fused path) | `torch/optim/sgd.py` | — |
| SGD single tensor update | `torch/optim/_functional.py` | — |
| `Adam.__init__` | `torch/optim/adam.py` | — |
| `AdamW.__init__` | `torch/optim/adamw.py` | — |
| `Optimizer.__init__` | `torch/optim/optimizer.py` | — |
| `Optimizer.zero_grad` | `torch/optim/optimizer.py` | — |
| `Optimizer.state_dict` | `torch/optim/optimizer.py` | — |
| `Optimizer.load_state_dict` | `torch/optim/optimizer.py` | — |

---

## 六、可借鉴的工程技巧

1. **`param_groups` 设计**: 一组参数 + 独立配置 = 一个 group。不同层可以不同 lr / weight_decay。类比：不同模块不同配置的键值对系统。

2. **多路径自动选择**: `foreach` / `fused` / `single` 根据环境和配置自动选最优路径，用户不必关心。

3. **`state` 字典**: 每个参数的训练状态（动量 buffer）与参数本身解耦存储，按 `id(p)` 索引，不受参数内容变化影响。

4. **fused kernel**: 将多个 update 操作融合进一个 CUDA kernel，省中间 alloc 和 launch overhead。

5. **Decoupled Weight Decay** (AdamW): 把正则化项和自适应学习率解耦，提高 Adam 的正则化效果。

---

## 七、实战常见坑点

### 1. `zero_grad()` 和 `step()` 顺序反了
**现象**: 模型不收敛 / loss 震荡。
**原因**:
```python
# 错误顺序
loss.backward()
opt.step()       # 用当前梯度更新
opt.zero_grad()  # 清空 —— 下个 step 没有梯度!
# 正确顺序
opt.zero_grad()  # 先清空
loss.backward()  # 再算梯度
opt.step()       # 再更新
```
**排查**: 打印 `param.grad` 看是否为零。

### 2. AdamW 的 weight_decay 与 L2 regularization 不同
**现象**: 把 SGD+L2 的 weight_decay 值直接搬到 AdamW → loss 发散。
**原因**: AdamW 的 decoupled weight decay 直接乘在参数上 (`p *= 1 - lr*wd`)，而 SGD 的 L2 是加在梯度上 (`grad += wd * p`) → AdamW 对 weight_decay 更敏感。
**经验**: AdamW 的 weight_decay 通常设 0.01-0.1 (SGD 设 1e-4-1e-3); 迁移训练时需重新调。

### 3. `param_groups` 漏了 lr/wd 设置
**现象**: 某个 layer 的 lr 和其他层一样，明明在 param_groups 里设了不同的值。
**原因**: param_groups 的后一个 dict 覆盖了前一个 dict。
```python
# BUG: lr 被第二个 group 覆盖
opt = Adam([
    {"params": model.head.parameters(), "lr": 1e-3, "weight_decay": 0.01},
    {"params": model.body.parameters()},  # 没设 lr → 用默认 1e-3
])
# 如果 body 在 head 后面且没设 lr → 继承默认值, 可能是 1e-3
```
**解决**: 每个 group 显式设置所有你关心的参数。

### 4. `load_state_dict` 后 `param_groups` 的 `params` 为空
**现象**: 加载 optimizer 后 `opt.param_groups[0]['params']` 为空 list。
**原因**: `load_state_dict` 用 `id(p)` 匹配参数 → 如果你重新创建了模型（新的 tensor 对象），`id` 变了。
**解决**:
```python
# 正确的 load pattern
model = MyModel()
opt = Adam(model.parameters(), lr=1e-3)
ckpt = torch.load("opt.pt")
opt.load_state_dict(ckpt)  # 必须在 model 创建后立即 load

# 错误：先 load model state_dict 会导致 param id 变化
```

### 5. `foreach=True` 在不支持的 dtype 上静默回退
**现象**: 以为用了高效的 fused kernel，实际回退到了单 tensor 循环。
**排查**:
```python
# 检查 optimizer step 是否走了 foreach 路径
opt = torch.optim.Adam(model.parameters(), foreach=True)
# 如果某个 param 是 sparse / complex → foreach 会被跳过
for p in model.parameters():
    if p.is_sparse or p.is_complex():
        print(f"foreach disabled for {p.dtype}")
```

### 6. 混合精度 (AMP) + optimizer step 的 scale 不一致
**现象**: AMP 训练中 optimizer.step() 报 "inf/nan detected"。
**原因**: AMP 的 GradScaler 在 step 前做 `unscale_` (除以 scale)。如果手动调用了 `scaler.unscale_(opt)` 后又调了一次 → 梯度变成 0 或无穷。
**正确**:
```python
scaler.scale(loss).backward()
scaler.step(opt)       # 内部自动 unscale + step
scaler.update()        # 更新 scale
# 不要手动调用 scaler.unscale_(opt)!
```

