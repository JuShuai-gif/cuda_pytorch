# Activation Checkpointing 源码分析

> 源码路径: `/home/ghr/code/pytorch/torch/utils/checkpoint.py` (1777 行)
> 核心类: `CheckpointFunction` (:234) — 继承 `torch.autograd.Function`
> 辅助函数: `detach_variable` (:71), `get_device_states` / `set_device_states`

## 0. 一句话总览

Activation Checkpointing = **用时间换空间**。`CheckpointFunction.forward` 中用 `torch.no_grad()` 包裹，不保存中间激活值。`backward` 中用 `torch.enable_grad()` 重新运行 forward 来重建激活值。RNG 状态被完整保存/恢复以保证 re-forward 的数值一致性。

---

## 一、`CheckpointFunction.forward` 源码 (:237)

```python
# checkpoint.py:237
@staticmethod
def forward(ctx, run_function, preserve_rng_state, *args):
    check_backward_validity(args)                              # 输入校验
    ctx.run_function = run_function                            # 保存 re-forward 函数
    ctx.preserve_rng_state = preserve_rng_state

    # 保存 autocast 状态
    ctx.device_type = _infer_device_type(*args)
    ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = \
        _get_autocast_kwargs(ctx.device_type)

    if preserve_rng_state:                                     # :246 保存 RNG
        ctx.fwd_cpu_state = torch.get_rng_state()              # CPU RNG
        ctx.had_device_in_fwd = False
        device_module = _get_device_module(ctx.device_type)
        if getattr(device_module, "_initialized", False):      # 惰性初始化检查
            ctx.had_device_in_fwd = True
            ctx.fwd_devices, ctx.fwd_device_states = \
                get_device_states(*args)                       # CUDA RNG per device

    # 分离 tensor 输入与非 tensor 输入 (:260-269)
    ctx.inputs = []          # 混合列表: tensor 位置为 None, 非 tensor 直接存值
    ctx.tensor_indices = []  # 记录哪些位置是 tensor
    tensor_inputs = []
    for i, arg in enumerate(args):
        if torch.is_tensor(arg):
            tensor_inputs.append(arg)
            ctx.tensor_indices.append(i)
            ctx.inputs.append(None)       # placeholder
        else:
            ctx.inputs.append(arg)         # 直接保存非 tensor

    ctx.save_for_backward(*tensor_inputs)  # autograd 管理 tensor 生命周期

    with torch.no_grad():                  # ★ 关键: 不建图, 不保存中间值
        outputs = run_function(*args)
    return outputs
```

**关键设计**:
1. `ctx.save_for_backward(*tensors)` — 通过 PyTorch 的 SavedVariable 机制管理 tensor 生命周期。`ctx.inputs` 中的 `None` 占位符在 backward 时被替换为实际的 `saved_tensors`。
2. `torch.no_grad()` — forward 中不建立 autograd 图 → 中间激活值不保存 → 显存大幅降低。

---

## 二、`CheckpointFunction.backward` 源码 (:278)

```python
# checkpoint.py:278
@staticmethod
def backward(ctx, *args):  # args = grad_outputs
    # Step 1: 从 ctx 恢复输入 (:287-293)
    inputs = list(ctx.inputs)
    tensor_indices = ctx.tensor_indices
    tensors = ctx.saved_tensors
    for i, idx in enumerate(tensor_indices):
        inputs[idx] = tensors[i]  # 用 saved_tensors 填充占位符

    # Step 2: 恢复 RNG 状态 (:298-307)
    rng_devices = []
    if ctx.preserve_rng_state and ctx.had_device_in_fwd:
        rng_devices = ctx.fwd_devices
    with torch.random.fork_rng(devices=rng_devices, ...):
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.fwd_cpu_state)           # CPU RNG
            if ctx.had_device_in_fwd:
                set_device_states(ctx.fwd_devices, ...)       # CUDA RNG

        # Step 3: detach 输入, 重新加 requires_grad (:308)
        detached_inputs = detach_variable(tuple(inputs))
        # detach_variable (:71): inp.detach(); x.requires_grad = inp.requires_grad

        # Step 4: 重新运行 forward (这次建图!) (:313-314)
        with torch.enable_grad(), ...:
            outputs = ctx.run_function(*detached_inputs)

    # Step 5: 只对 requires_grad 的输出做 backward (:320-331)
    outputs_with_grad = [o for o in outputs if o.requires_grad]
    args_with_grad = [args[i] for i, o in enumerate(outputs) if o.requires_grad]
    torch.autograd.backward(outputs_with_grad, args_with_grad)  # :331

    # Step 6: 收集各输入的梯度 (:332-333)
    grads = tuple(inp.grad for inp in inputs if isinstance(inp, torch.Tensor))
    return (None, None) + grads
```

**关键步骤**:
1. **RNG 恢复**: `torch.random.fork_rng` 创建子 RNG 环境 → re-forward 中的 dropout/BatchNorm 行为与原始 forward 完全一致 → 梯度正确
2. **detach + enable_grad**: `detach` 切断旧计算图, `enable_grad()` 重新建立 → 新 backward 图与原图隔离
3. **backward**: `torch.autograd.backward(outputs, grad_outputs)` — 在新图上执行 backward, 各输入 `.grad` 自动填充

---

## 三、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `CheckpointFunction` | `checkpoint.py` | 234 |
| `forward` (no_grad) | `checkpoint.py` | 237 |
| RNG 状态保存 | `checkpoint.py` | 246-256 |
| `save_for_backward` | `checkpoint.py` | 271 |
| `backward` (re-forward) | `checkpoint.py` | 278 |
| RNG 恢复 | `checkpoint.py` | 298-307 |
| `detach_variable` | `checkpoint.py` | 71 |
| `enable_grad` + re-run | `checkpoint.py` | 313-314 |
| `autograd.backward` (新图) | `checkpoint.py` | 331 |
| `fork_rng` | `torch/random.py` | — |
| `get_device_states` | `checkpoint.py` | — |

---

## 四、实战常见坑点

*(见历史版本)*
