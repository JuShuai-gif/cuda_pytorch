# AMP GradScaler 梯度缩放源码分析

> 源码: `torch/amp/grad_scaler.py` — `GradScaler` 类
> C++ 扩展: `torch/csrc/amp/` — C++ 端的 overflow 检测

## 0. 一句话总览

fp16 梯度可能小到 flush-to-zero，导致训练停滞。`GradScaler` 在 backward 前将 loss 乘以一个 scale factor → 梯度放大 → optimizer 更新前除以 scale → 恢复正确更新量。如果检测到梯度 overflow → 跳过本次更新并增大 scale。

---

## 一、GradScaler 工作流程

```
Step 1:  scale loss
         loss = loss * scaler.get_scale()    [or: scaler.scale(loss)]

Step 2:  backward
         loss.backward()  → grad 被放大 scale 倍

Step 3:  unscale gradients (scaler.step)
         opt.param_groups 中的所有 grad 除以 scale

Step 4:  检测 Inf/NaN
         如果 grad 中有 Inf/NaN → 跳过 step, scale *= backoff_factor
         否则 → opt.step(), scale *= growth_factor
```

### 1.1 为什么需要 scale

```
fp16 最小正规格化数: 6.1e-5
fp16 次规格化最小值: 5.9e-8

情况: 真实梯度 1e-5 (对大型模型的后几层是正常的)
      fp16 表示 → 可能 flush to zero → 该参数不再更新
      ×128 scale → 1.28e-3 → fp16 可以正常表示 → 更新成功
```

### 1.2 源码核心: `scaler.step(opt)` 做什么

```python
# grad_scaler.py (简化)
def step(self, optimizer, *args, **kwargs):
    # 1. 对每个 param_group 的 grad 除以 scale
    self._unscale_grads_(optimizer)
    # 2. 检查是否有 Inf/NaN
    if self._found_inf_per_device(optimizer):
        self._inf_count += 1
        return  # 跳过 step
    # 3. 执行 optimizer.step()
    optimizer.step(*args, **kwargs)
    # 4. 更新 scale
    self.update()
```

### 1.3 Dynamic vs Static Scale

```python
# Static (不推荐, 需要手动调 scale)
scaler = torch.amp.GradScaler(init_scale=128, growth_factor=1.0)

# Dynamic (推荐, 自动调整)
scaler = torch.amp.GradScaler(
    init_scale=2**16,       # 65536
    growth_factor=2.0,      # 没 overflow → scale ×2
    backoff_factor=0.5,     # 有 overflow → scale ×0.5
    growth_interval=2000,   # 每 2000 steps 尝试 ×2
)
```

---

## 二、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `GradScaler.__init__` | `torch/amp/grad_scaler.py` | — |
| `GradScaler.scale` | `torch/amp/grad_scaler.py` | — |
| `GradScaler.step` | `torch/amp/grad_scaler.py` | — |
| `_unscale_grads_` | `torch/amp/grad_scaler.py` | — |
| `_found_inf_per_device` | `torch/amp/grad_scaler.py` | — |
| Overflow 检测 (C++) | `torch/csrc/amp/` | — |

---

## 三、实战常见坑点

### 1. 手动调了 scale 导致 loss 爆炸
`scale` 设太小: grad 被截断为 0。`scale` 设太大: grad overflow → 不断 backoff。

### 2. scaler.step(opt) 后手动 unscale → 梯度变 0
`scaler.step(opt)` 内部已做 unscale。手动再调 `scaler.unscale_(opt)` → 梯度被除两次 scale → 变成 0。

### 3. 所有 grad 永远 ok → scale 无限增长
`growth_factor=2.0` 且从没 overflow → scale 指数增长 → 最终必然 overflow。Dynamic scaler 有上限 (`max_scale=2**24`)。

### 4. 多 GPU 时 scale 同步
`scaler.update()` 需要在所有 GPU 上都调用。`scaler.step(opt)` 内部调用 `update()` — 如果只有 rank 0 调了 step → 其他 rank 的 scale 不同步。
