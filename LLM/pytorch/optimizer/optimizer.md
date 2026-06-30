# Optimizer 源码分析: SGD & Adam

> 源码路径: `/home/ghr/code/pytorch/torch/optim/sgd.py` (544 行), `torch/optim/adam.py`
> 基类: `torch/optim/optimizer.py` — `Optimizer` 类, `zero_grad`, `step`, `state_dict`

## 0. 一句话总览

`Optimizer.step()` 的核心是 `param_groups` 遍历 → 提取梯度 → 执行更新公式 → 写回 `param.data`。SGD 有三种执行路径 (single/foreach/fused) 根据环境自动选择，Adam 额外维护 `state[param]` 存储一阶/二阶矩估计。

---

## 一、SGD 源码分析 (`sgd.py`)

### 1.1 `SGD.step()` — 入口 (:106)

```python
# sgd.py:106
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()  # 重新计算 loss (用于 LBFGS 等需要)

    for group in self.param_groups:
        params, grads = [], []
        momentum_buffer_list = []
        has_sparse_grad = self._init_group(group, params, grads, momentum_buffer_list)
        # _init_group: 从 group["params"] 收集梯度不为 None 的参数

        sgd(params, grads, momentum_buffer_list,  # ← 进入三种路径
            weight_decay=group["weight_decay"],
            momentum=group["momentum"], lr=group["lr"],
            dampening=group["dampening"], nesterov=group["nesterov"],
            maximize=group["maximize"], has_sparse_grad=has_sparse_grad,
            foreach=group["foreach"], fused=group["fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
        )

    # 把更新后的 momentum_buffer 写回 self.state[p]
    return loss
```

### 1.2 三种执行路径的分发

`sgd()` 函数在 `_functional.py` 中实现，根据参数选择路径:

| 路径 | 条件 | 实现函数 |
|------|------|----------|
| **fused** | `fused=True` + CUDA + no sparse | `_fused_sgd` (:479) — 单 kernel 完成 update |
| **foreach** | `foreach=True` or 自动选 | `_multi_tensor_sgd` (:382) — `torch._foreach_*` API |
| **single** | 回退 | `_single_tensor_sgd` (:322) — for 循环逐参数 |

`_init_group` 还负责检测 sparse grad — sparse 参数只能走 single tensor 路径，禁用 foreach/fused。

### 1.3 `_single_tensor_sgd` — 逐参数循环 (:322)

```python
# sgd.py:322
def _single_tensor_sgd(params, grads, momentum_buffer_list, *,
    weight_decay, momentum, lr, dampening, nesterov, maximize, has_sparse_grad):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]  # :344

        if weight_decay != 0:                           # :346
            grad = grad.add(param, alpha=weight_decay)  # L2 weight decay

        if momentum != 0:                               # :357
            buf = momentum_buffer_list[i]
            if buf is None:                             # 首次: buf = grad
                buf = grad.detach().clone()              # :361
            else:                                       # 后续: EMA
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)  # :364

            if nesterov:                                # Nesterov lookahead
                grad = grad.add(buf, alpha=momentum)    # :367
            else:
                grad = buf                              # :369

        param.add_(grad, alpha=-lr)                     # p -= lr * grad  :379
```

**关键点**:
1. `maximize=True` → 梯度取反 → 梯度上升
2. 首次调用时 `momentum_buffer_list[i]` 为 `None` → `buf = grad.clone()` 初始化
3. `dampening != 0` → `buf = mu*buf + (1-dampening)*grad` (标准: dampening=0)
4. Nesterov: `grad += momentum * buf` (用动量预估的下一步位置算梯度)
5. `param.add_(grad, alpha=-lr)` — 最终 in-place 更新

### 1.4 `_multi_tensor_sgd` — foreach 批量路径 (:382)

```python
# sgd.py:382-478
def _multi_tensor_sgd(params, grads, ...):
    # 使用 torch._foreach_* API — 一个 kernel launch 处理多个 tensor
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)
    if momentum != 0:
        torch._foreach_mul_(momentum_buffer_list, momentum)
        torch._foreach_add_(momentum_buffer_list, grads, alpha=1 - dampening)
    ...
    torch._foreach_add_(params, grads, alpha=-lr)
```

**优势**: 多个小 tensor 的更新合并到一个 kernel launch → 减少 CPU→GPU launch overhead。

---

## 二、Adam 源码分析

### 2.1 `Adam.step()` — 核心公式

Adam = SGD momentum + RMSprop, 核心在 `adam.py:215`:

```python
# adam.py:215 (简化)
def step(self, closure=None):
    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # 初始化 state
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p)      # 一阶矩 m
                state["exp_avg_sq"] = torch.zeros_like(p)   # 二阶矩 v

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            beta1, beta2 = group["betas"]

            # 偏差校正
            bias_correction1 = 1 - beta1 ** state["step"]   # :297
            bias_correction2 = 1 - beta2 ** state["step"]

            # 更新带 bias 的 moment estimates
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)           # m_t
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

            # 更新 (带 weight decay)
            if group["weight_decay"] != 0:
                p.mul_(1 - lr * weight_decay)  # decoupled weight decay

            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)
    return loss
```

### 2.2 AdamW 与 Adam 的关键差异

```python
# Adam:  weight_decay 加在 grad 上
grad = grad + weight_decay * p
p -= lr * m_hat / (sqrt(v_hat) + eps)

# AdamW: weight_decay 直接乘在参数上 (decoupled)
p *= (1 - lr * weight_decay)
p -= lr * m_hat / (sqrt(v_hat) + eps)
```

AdamW 的解耦避免了 weight_decay 与自适应学习率 `1/sqrt(v_hat)` 的耦合 — 这是 AdamW 比 Adam + L2 效果更好的根本原因。

---

## 三、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `SGD.__init__` (默认参数) | `sgd.py` | 25 |
| `SGD.step` (入口) | `sgd.py` | 106 |
| `_init_group` (梯度收集) | `optimizer.py` | — |
| `_single_tensor_sgd` | `sgd.py` | 322 |
| `_multi_tensor_sgd` (foreach) | `sgd.py` | 382 |
| `_fused_sgd` (CUDA kernel) | `sgd.py` | 479 |
| `Adam.__init__` | `adam.py` | — |
| `Adam.step` (入口) | `adam.py` | 215 |
| `adam()` (single tensor) | `adam.py` | 903 |
| `AdamW.__init__` | `adamw.py` | — |
| `Optimizer.zero_grad` | `optimizer.py` | — |
| `Optimizer.load_state_dict` | `optimizer.py` | — |

---

## 四、实战常见坑点

*(见历史版本)*
