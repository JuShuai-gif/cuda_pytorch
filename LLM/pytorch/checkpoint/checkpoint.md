# Activation Checkpointing 源码分析

> 源码路径: `/home/ghr/code/pytorch/torch/utils/checkpoint.py` (1777 行)
> 核心类: `CheckpointFunction` (继承 `torch.autograd.Function`)

## 0. 一句话总览

Activation Checkpointing = **用时间换空间**：正向时不保存中间激活值，反向时需要时重新计算。核心实现是一个自定义 autograd Function，在 forward 中不保存（或选择性保存）tensor，在 backward 中重新运行 forward 来重建激活值。

---

## 一、核心原理

### 正常训练

```
forward:  x -> L1 -> a1 -> L2 -> a2 -> ... -> loss
          (保存 a1, a2, a3... 用于 backward)

backward: a3 -> grad_L4 -> a2 -> grad_L3 -> a1 -> grad_L2 -> x -> grad_L1
```

### Checkpointing

```
forward:  x -> L1 -> a1 -> L2 -> a2 -> ... -> loss
          (只保存输入 x，不保存 a1, a2, a3...)

backward: a3 -> grad_L4
          └─ 重新运行 L1, L2, L3 获得 a1, a2, a3
          a2 -> grad_L3
          └─ 重新运行 L1, L2 获得 a1, a2
          a1 -> grad_L2
          └─ 重新运行 L1 获得 a1
```

**显存节省**: 从 O(N * activation_size) → O(1 * activation_size)，N 是层数。
**计算代价**: 每个 checkpoint 段需要多跑一次 forward（约 33% 额外计算）。

---

## 二、`CheckpointFunction` 实现 (checkpoint.py)

### 2.1 `forward()` — 不保存中间激活值

```python
# checkpoint.py:CheckpointFunction.forward
def forward(ctx, run_function, preserve_rng_state, *args):
    ctx.run_function = run_function
    ctx.preserve_rng_state = preserve_rng_state

    # 保存 RNG 状态（确保 re-forward 时随机数一致）
    ctx.had_autocast_in_fwd = torch.is_autocast_cache_enabled()

    if preserve_rng_state:
        ctx.fwd_cpu_state = torch.get_rng_state()
        ctx.had_cuda_in_fwd = torch.cuda.is_initialized()
        if ctx.had_cuda_in_fwd:
            ctx.fwd_cuda_devices = ...
            ctx.fwd_cuda_states = [torch.cuda.get_rng_state(d) for d in ...]

    with torch.no_grad():
        outputs = run_function(*args)

    return outputs
```

**关键**: 使用 `torch.no_grad()` 包裹 forward —— checkpoint segment 内不建 autograd 图。保存输入 args 但不保存中间结果。

### 2.2 `backward()` — 重新运行 forward

```python
def backward(ctx, *grad_outputs):
    # 1. 恢复 RNG 状态
    if ctx.preserve_rng_state:
        torch.set_rng_state(ctx.fwd_cpu_state)
        if ctx.had_cuda_in_fwd:
            for d, s in zip(ctx.fwd_cuda_devices, ctx.fwd_cuda_states):
                torch.cuda.set_rng_state(s, d)

    # 2. 对输入 detach + requires_grad（重新加入 autograd 图）
    inputs = []
    for inp in ctx.saved_tensors:
        x = inp.detach()
        x.requires_grad = inp.requires_grad
        inputs.append(x)

    # 3. 用 requires_grad 的输入重新运行 forward（这次建图）
    with torch.enable_grad():
        outputs = ctx.run_function(*inputs)

    # 4. 对新图调用 backward，传入上游梯度
    torch.autograd.backward(outputs, grad_outputs)

    # 5. 返回各输入的梯度
    return (None, None) + tuple(inp.grad for inp in inputs)
```

**步骤**:
1. **恢复 RNG** — 确保 re-forward 中 dropout/BatchNorm 的结果与原始 forward 一致
2. **detach + requires_grad** — 切断输入与旧 autograd 图的连接，重新加入新图
3. **重新运行 forward** — 这次用 `enable_grad()`，建立 autograd 图
4. **新图的 backward** — `torch.autograd.backward(outputs, grad_outputs)`
5. **返回梯度** — 收集 `inp.grad`

### 2.3 `saved_tensors` 的保存策略

`detach_variable` (`checkpoint.py:71`) 对输入做 `detach()` 后保存。这些 `saved_tensors` 通过 `ctx.save_for_backward()` 存储，由 PyTorch 的 SavedVariable 机制管理生命周期。

---

## 三、RNG 状态管理 (checkpoint.py)

### 3.1 为什么需要保存/恢复 RNG

如果 checkpoint segment 包含 dropout：
- 第一遍 forward: dropout mask = [1, 0, 1, 0]
- 如果 re-forward 不恢复 RNG: dropout mask = [0, 1, 1, 0] → 结果不同，梯度错误

### 3.2 保存的状态

```python
ctx.fwd_cpu_state = torch.get_rng_state()           # CPU RNG
ctx.fwd_cuda_states = [torch.cuda.get_rng_state(d)] # 每个 CUDA 设备的 RNG
ctx.fwd_cuda_devices = [...]                        # 记录哪些设备有 CUDA
```

### 3.3 `preserve_rng_state` 选项

`checkpoint(function, *args, preserve_rng_state=True)`:
- `True`: 保存/恢复 RNG（默认，安全但稍慢）
- `False`: 不保存 RNG（快，但如果 segment 含 dropout 则结果错误）

---

## 四、Selective Checkpointing

PyTorch 2.x 引入了**选择性 checkpointing**：不是丢弃所有中间激活值，而是根据策略保留部分：

```python
# checkpoint.py: SAC_IGNORED_OPS (ops that are cheap to recompute)
# 对于 cheap op（如 relu, add）：不保存，backward 时重算
# 对于 expensive op（如 matmul, conv）：保存，避免 re-forward
```

`CheckpointPolicy` (`checkpoint.py`) 定义了哪些 op 应该保存、哪些应该 recompute。

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `checkpoint()` 入口 | `torch/utils/checkpoint.py` | — |
| `CheckpointFunction.forward` | `torch/utils/checkpoint.py` | — |
| `CheckpointFunction.backward` | `torch/utils/checkpoint.py` | — |
| `detach_variable` | `torch/utils/checkpoint.py` | 71 |
| `get_device_states` | `torch/utils/checkpoint.py` | — |
| `set_device_states` | `torch/utils/checkpoint.py` | — |
| `CheckpointPolicy` | `torch/utils/checkpoint.py` | — |
| `SAC_IGNORED_OPS` | `torch/utils/checkpoint.py` | — |

---

## 六、可借鉴的工程技巧

1. **时间换空间 (trading compute for memory)**: 深度学习中的经典设计模式 — 当显存是瓶颈时，牺牲一部分计算来换显存。

2. **RNG 状态序列化/恢复**: 任何需要「确定性重放」的场景都需要保存/恢复 RNG 状态。类比：训练 resume、数据增强复现。

3. **detach + re-forward**: backward 中通过 detach 切断旧图，重新建图来实现 recomputation，比「backward 中手动推导梯度公式」更优雅。

4. **选择性策略**: 不是所有中间值都值得保存/丢弃。根据 op 的计算成本做选择性决策（SAC = Selective Activation Checkpointing）。

5. **SavedVariable 管理**: `ctx.save_for_backward()` 使用引用计数 + 弱引用管理 saved tensor 的生命周期，避免泄漏。

---

## 七、实战常见坑点

### 1. checkpoint 后显存不降反升
**现象**: 加了 `checkpoint()` 期望显存降低，peak memory 反而更高。
**原因**: (1) checkpoint segment 太短 → 每个 segment 都要保存输入 → 输入激活值本身很大。(2) RNG state 保存/恢复的开销。(3) `preserve_rng_state=True` 保存了所有 CUDA device 的 RNG → 额外显存。
**解决**:
```python
# 只 checkpoint 计算量大的 chunk（如 attention block）
checkpoint(attention_block, x, use_reentrant=False)
# 不在 trivial op 上做 checkpoint

# 如果 segment 内无 dropout, 可以关闭 RNG 保存
checkpoint(block, x, preserve_rng_state=False)
```

### 2. 训练 loss 不下降 / 结果不对
**现象**: checkpoint 前后 forward 输出一致，但训练 loss 曲线完全不同。
**原因**: `preserve_rng_state=False` 且 segment 内有 dropout → re-forward 时 mask 不同 → 梯度有偏。
**解决**: 包含 dropout 的 checkpoint 必须 `preserve_rng_state=True`（默认）。

### 3. 多 GPU 下 RNG 状态错乱
**现象**: 多卡训练时，checkpoint 段内的 dropout pattern 每张卡不同。
**原因**: `get_device_states()` 保存的 CUDA RNG 状态是 per-device 的。re-forward 时如果 worker 切换了设备，RNG 恢复错误。
**解决**: 确保 checkpoint 段内的 forward 始终在同一设备上运行。

### 4. 与 torch.compile 叠加的 recompile 风暴
**现象**: `torch.compile` + `checkpoint` 组合后每个 step 都触发重编译。
**原因**: checkpoint 的 re-forward 产生新的 autograd 图 → Dynamo 认为这是"不同的函数"。
**解决**: torch >= 2.2 改进了交互; 旧版本用 `use_reentrant=True`（虽然文档标记 deprecated）。

### 5. 梯度 NaN 但只出现在 checkpoint 段
**现象**: 不 checkpoint 时梯度正常, checkpoint 后梯度 NaN。
**原因**: re-forward 时 batch norm 的 running stats 可能被再次更新(如果 BN 在 `training=True`)，与第一次 forward 的统计量不一致。
**排查**:
```python
# 检查 checkpoint 段内 BN 的 training 状态
for m in checkpoint_segment.modules():
    if isinstance(m, nn.BatchNorm2d):
        print(m.training)  # 应为 False
```
**解决**: checkpoint 前设置 `model.eval()` 或确保 BN 的 `track_running_stats=True`。

