# torch.compile 编译调试体系源码分析

> 源码: `torch/_dynamo/` — 完整的 TORCH_LOGS 基础设施
> 入口: `torch._dynamo.utils.py` (5795 行) — logging 注册
> Explain: `torch._dynamo.explain()` — graph break 分析

## 0. 一句话总览

`torch.compile` 的三个核心调试命令: `TORCH_LOGS` 控制日志输出粒度, `torch._dynamo.explain()` 分析 graph break 原因, `TORCHDYNAMO_VERBOSE=1` 看完整编译过程。遇到 compile 失败时，按此顺序排查即可定位 90% 的问题。

---

## 一、TORCH_LOGS 完整指南

### 1.1 环境变量设置

```bash
# 基础: 看所有 graph break
TORCH_LOGS=graph_breaks python script.py

# 看重新编译原因
TORCH_LOGS=recompiles python script.py

# 看生成的 Triton/Inductor 代码
TORCH_LOGS=output_code python script.py

# 看编译的图
TORCH_LOGS=graph_code python script.py

# 组合: 逗号分隔
TORCH_LOGS=graph_breaks,recompiles python script.py
```

### 1.2 所有可用的 LOG 选项

| LOG 名 | 作用 |
|--------|------|
| `graph_breaks` | 每次 graph break 的原因 + 位置 |
| `recompiles` | 每次 recompile 的 guard failure 原因 |
| `output_code` | Inductor 生成的 Triton/C++ 代码 |
| `graph_code` | Dynamo 捕获的 FX 图源码 |
| `guards` | 生成的 guard 条件 |
| `dynamic` | 动态 shape 相关 |
| `+all` | 所有日志 (非常大) |

### 1.3 在代码中设置

```python
import torch._dynamo.config as dynamo_config
dynamo_config.log_level = logging.DEBUG  # 等价于 TORCH_LOGS=+all

import logging
logging.getLogger("torch._dynamo").setLevel(logging.DEBUG)
```

---

## 二、`torch._dynamo.explain()` 详解

```python
@torch.compile
def fn(x):
    y = x * 2
    if y.sum() > 0:  # graph break!
        y = y + 1
    return y

# explain 返回一个 Explanation 对象
explanation = torch._dynamo.explain(fn)(torch.randn(4))
print(f"Graph count: {explanation.graph_count}")
print(f"Graph breaks: {explanation.graph_break_count}")
for i, (reason, user_stack) in enumerate(
    zip(explanation.break_reasons, explanation.user_stacks)):
    print(f"Break {i}: {reason}")
```

### 2.1 Explanation 对象字段

| 字段 | 含义 |
|------|------|
| `graph_count` | 编译出的 FX 图数量 |
| `graph_break_count` | graph break 次数 |
| `break_reasons` | 每次 break 的原因 |
| `graphs` | 所有编译出的图 |
| `user_stacks` | 每次 break 的用户代码调用栈 |

---

## 三、编译失败的排查优先级

```
1. TORCH_LOGS=graph_breaks    ← 看看哪里 break 了
2. torch._dynamo.explain(fn)  ← 详细 break 原因 + 堆栈
3. TORCH_LOGS=recompiles      ← 是否有不必要的 recompile
4. TORCH_LOGS=output_code     ← 看看生成的 kernel 长什么样
5. TORCHDYNAMO_VERBOSE=1      ← 完整编译日志 (非常 verbose)
6. torch._dynamo.config.verbose=True ← 代码内开启 verbose
```

---

## 四、常见 graph break 原因及修复

| Break 原因 | 修复 |
|-----------|------|
| `Data-dependent control flow` | 用 `torch.cond` 替换 `if` |
| `Unsupported: tensor.item()` | 不在 compiled 函数内取标量值 |
| `Unsupported: .data` | 不用 `.data`, 用 `.detach()` |
| `Unsupported: print()` | 不在 compiled 函数内 `print` |
| `Shape change: ...` → recompile | 用 `dynamic=True` |
| `Unspecialized nn.Module` | 用 `@torch.compile(fullgraph=True)` 强制零 break |

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `TORCH_LOGS` 注册 | `torch/_dynamo/utils.py` | — |
| `torch._dynamo.explain` | `torch/_dynamo/__init__.py` | — |
| `Explanation` 类 | `torch/_dynamo/utils.py` | — |
| Graph break 检测 | `torch/_dynamo/symbolic_convert.py` | — |
| Guard 生成 | `torch/_dynamo/guards.py` | — |
| `torch._dynamo.config` | `torch/_dynamo/config.py` | — |

---

## 六、实战常见坑点

### 1. TORCH_LOGS 不输出任何东西
**原因**: 函数没有被执行 (compiled 但未调用), 或者 `@torch.compile` 装饰器在条件分支内未 hit。
**排查**: 加 `print("compiled fn called")` 在 compiled 函数内。

### 2. explain() 报错但 compile 不报错
**原因**: `explain()` 使用不同的 execution mode → 可能触发不同的路径。
**解决**: 改用 `torch.compile(backend="eager")` 对比。

### 3. graph_breaks 日志显示 `Unsupported: ...` 但代码中根本没有这个 op
**原因**: 内部 op 被分解出来的 (如 `F.gelu` → `aten::gelu`)。
**解决**: 检查 graph_code 看实际的 ATen op 是什么。
