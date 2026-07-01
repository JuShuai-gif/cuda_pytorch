# nn.Module 核心机制源码分析

> 源码路径: `/home/ghr/code/pytorch/torch/nn/modules/module.py` (3053 行)
> C++ 绑定: `/home/ghr/code/pytorch/torch/csrc/Module.cpp`

## 0. 一句话总览

`nn.Module` 本质是一部**状态机 + 钩子系统**：四个 OrderedDict 管理参数、buffer、子模块、hooks，`__call__` 在 `forward` 前后插入钩子执行点。

---

## 一、`__call__` 不是普通方法 (line 1917)

```python
# module.py:1917
__call__: Callable[..., Any] = _wrapped_call_impl
```

这不是实例方法，而是**类属性别名**。`model(x)` 会路由到 `_wrapped_call_impl`（:1774）：

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)  # torch.compile hook
    else:
        return self._call_impl(*args, **kwargs)
```

如果有 `torch.compile`，走编译路径；否则走钩子路径。

---

## 二、`_call_impl` 的六个执行阶段 (:1782-1915)

```
__call__ (trigger)
  │
  ├─ [Phase 1] forward pre-hooks (:1805-1825)  — 修改 args/kwargs
  │     global hooks 先于 module hooks 执行
  │     有 with_kwargs 的 hook 收到 (self, args, kwargs)
  │     没有的收到 (self, args)，返回非 None 则替换 args
  │
  ├─ [Phase 2] setup backward hooks (:1827-1830)
  │     创建 BackwardHook 包装器，挂在输入 tensor 上
  │
  ├─ [Phase 3] forward() (:1832)
  │     forward_call = self._slow_forward if tracing else self.forward
  │
  ├─ [Phase 4] forward hooks (:1833-1848)  — 修改 output
  │     hook(self, args, result) -> None or modified_result
  │     返回值非 None 则替换当前 result
  │
  ├─ [Phase 5] setup output hooks (:1850-1855)
  │     BackwardHook 挂在输出 tensor 上，用于捕获完整 grad_input/grad_output
  │
  └─ [Phase 6] exception handling (:1880-1914)
        如果有 always_call=True 的 forward hooks 未执行，在异常时补执行
```

核心代码 (`:1786-1789`)：

```python
# FAST PATH: 没有任何 hooks 时直接调用 forward，零开销
if not (self._backward_hooks or self._backward_pre_hooks or
        self._forward_hooks or self._forward_pre_hooks or ...):
    return forward_call(*args, **kwargs)
```

---

## 三、Hook 系统的五种 hook 类型

### 3.1 Forward Pre-Hooks (`:1624-1685`)

注册: `module.register_forward_pre_hook(hook, with_kwargs=False)`
钩子签名: `hook(module, args) -> None or modified_args`
存储: `self._forward_pre_hooks` (OrderedDict)

### 3.2 Forward Hooks (`:1687-1752`)

注册: `module.register_forward_hook(hook, always_call=False, with_kwargs=False)`
钩子签名: `hook(module, args, output) -> None or modified_output`
存储: `self._forward_hooks`

`always_call=True` 的 hook 即使 forward 抛异常也会执行 (:1885-1914)。

### 3.3 Full Backward Hooks (`:1460-1524`)

注册: `module.register_full_backward_hook(hook)`
钩子签名: `hook(module, grad_input, grad_output) -> tuple or None`
原理: 用 `BackwardHook` 包装器 (:1828) 在输入/输出 tensor 上注册 `Tensor.register_hook()`，从而在 backward 时获得完整的 `grad_input` 和 `grad_output`（不只是针对单个 tensor）。

### 3.4 Backward Pre-Hooks (`:1546-1551`)

注册: `module.register_full_backward_pre_hook(hook)`
钩子签名: `hook(module, grad_output) -> None or modified_grad_output`

### 3.5 State Dict Hooks (`:2173-2192`)

注册: `module._register_state_dict_hook(hook)` / `module._register_load_state_dict_pre_hook(hook)`

---

## 四、参数管理的三个关键路径

### 4.1 `__setattr__` 自动检测 Parameter (`:1971-2074`)

```python
# module.py:1981-1992
if isinstance(value, Parameter):
    remove_from(self.__dict__, self._buffers, self._modules, ...)
    self.register_parameter(name, value)
```

当你写 `self.weight = nn.Parameter(tensor)`，`__setattr__` 检测到 `isinstance(value, Parameter)`，自动调用 `register_parameter`，将其存入 `self._parameters` 而不是 `self.__dict__`。

同样逻辑处理 `Module`（→ `self._modules`）和 `Buffer`/`Tensor`（→ `self._buffers`）。

### 4.2 `register_parameter` (`:592-640`)

```python
# module.py:596-640
self._parameters[name] = param
```

关键校验:
- `param` 必须是 `None` 或 `isinstance(param, Parameter)` (`:607`)
- `param.grad_fn` 必须为 `None`（叶子张量）(`:618`)

### 4.3 `register_buffer` (`:528-590`)

```python
# module.py:586-590
self._buffers[name] = tensor
if persistent:
    self._non_persistent_buffers_set.discard(name)
else:
    self._non_persistent_buffers_set.add(name)
```

`persistent=False` 的 buffer 不会被 `state_dict()` 收集 (`_save_to_state_dict:2162`)。

---

## 五、`state_dict()` / `load_state_dict()` 流程

### 5.1 `state_dict` (`:2194-2282`)

```
state_dict(destination, prefix='')
  ├─ _state_dict_pre_hooks  (:2264)  — 预修改
  ├─ _save_to_state_dict    (:2266)  — 保存本模块的 params + persistent buffers
  ├─ 递归子模块             (:2267)  — child.state_dict(prefix=prefix+name+'.')
  └─ _state_dict_hooks      (:2274)  — 后修改
```

### 5.2 `_save_to_state_dict` (`:2143-2169`)

```python
# 保存参数
for name, param in self._parameters.items():
    if param is not None:
        destination[prefix + name] = param if keep_vars else param.detach()

# 保存 persistent buffers（跳过 non_persistent）
for name, buf in self._buffers.items():
    if buf is not None and name not in self._non_persistent_buffers_set:
        destination[prefix + name] = buf if keep_vars else buf.detach()
```

**关键**: 使用 `detach()` 而非直接保存 parameter，切断 autograd 图引用。

### 5.3 `load_state_dict` (`:2382-2506`)

1. 收集当前模型和 checkpoint 中**所有缺失和不匹配的 keys**
2. 支持 `strict=True/False`（严格模式下缺失/多余 key 报错）
3. 对每个 key，用 `__getattr__` 找到对应的 parameter/buffer，调用 `param.copy_(value)`
4. 支持 `_load_state_dict_pre_hooks` 在赋值前检查/过滤

---

## 六、`_apply()` 的批量类型转换 (`:930-1036`)

```python
# module.py:930-1036
def _apply(self, fn, recurse=True):
    # 1. 递归子模块
    if recurse:
        for module in self.children():
            module._apply(fn)

    # 2. 对 parameters: fn(param) -> param.data = fn(param)
    #    保持 Parameter 对象身份不变（optimizer 持有引用）
    for key, param in self._parameters.items():
        with torch.no_grad():
            param_applied = fn(param)  # e.g. .cuda(), .half()
        # swap_tensors for FakeTensor (:975-993)
        # param.data = param_applied  (:994-996)
        # 或用新的 Parameter 替换 (:997-1004)

    # 3. 对 buffers: 简单替换
    for key, buf in self._buffers.items():
        self._buffers[key] = fn(buf)

    return self
```

**关键设计**: 使用 `param.data = param_applied` 而非重建 `Parameter` 对象，因为 optimizer 持有 parameter 的引用，重建会导致 optimizer state 丢失 (`compute_should_use_set_data:937`)。

---

## 七、`parameters()` / `buffers()` 去重机制 (`:2645-2720`)

使用 `_named_members` 通用实现，核心去重逻辑 (:2657)：

```python
def _named_members(self, get_members_fn, prefix='', recurse=True, remove_duplicate=True):
    memo = set()
    for module_prefix, module in modules:
        members = get_members_fn(module)  # e.g., module._parameters.items()
        for k, v in members:
            if v is None or v in memo:
                continue
            if remove_duplicate:
                memo.add(v)  # 共享参数只出现一次
            yield name, v
```

`parameters()`、`named_parameters()`、`buffers()`、`named_buffers()` 都调用同一个 `_named_members`，只是传入不同的 getter。

---

## 八、`train()` / `eval()` (`:2885-2923`)

```python
def train(self, mode=True):
    self.training = mode
    for module in self.children():
        module.train(mode)  # 递归设置
    return self

def eval(self):
    return self.train(False)  # 就是 train(False)
```

---

## 九、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `__call__` alias | `module.py` | 1917 |
| `_wrapped_call_impl` | `module.py` | 1774 |
| `_call_impl` (6 phases) | `module.py` | 1782-1915 |
| `register_forward_pre_hook` | `module.py` | 1624 |
| `register_forward_hook` | `module.py` | 1687 |
| `register_full_backward_hook` | `module.py` | 1460 |
| `__setattr__` | `module.py` | 1971-2074 |
| `register_parameter` | `module.py` | 592 |
| `register_buffer` | `module.py` | 528 |
| `_apply` | `module.py` | 930 |
| `state_dict` | `module.py` | 2194 |
| `_save_to_state_dict` | `module.py` | 2143 |
| `named_parameters` | `module.py` | 2690 |
| `_named_members` | `module.py` | 2645 |
| `train` / `eval` | `module.py` | 2885 / 2907 |
| Init (hook dicts) | `module.py` | 482-524 |

---

## 十、可借鉴的工程技巧

1. **FAST PATH 设计** (`:1786`): 无 hook 时跳过所有检查，零开销。判断条件写在调用入口，不在 hook 内部。

2. **组合优于继承**: `FusedSchedulerNode` （见 amp/inductor 笔记）和这里的 `RemovableHandle` 都用组合包装，不改原有数据结构。

3. **`detach()` 断引用** (`:2159`): state_dict 用 `detach()` 而非直接存 parameter，防止保存 checkpoint 时持有 autograd 图。

4. **保对象身份** (`:930`): `_apply` 用 `param.data = new` 而非新建 Parameter，因为 optimizer 持引用。

5. **声明式钩子注册**: `__setattr__` 自动检测类型 → 自动路由到 `_parameters/_buffers/_modules`，用户无需关心底层存储。

---

## 十一、实战常见坑点

### 1. `state_dict` 键名不匹配导致 load 失败
**现象**: load_state_dict 报 `Missing key(s)` + `Unexpected key(s)`。
**原因**: 模型结构变了（加了/删了/改名了层），或者 checkpoint 来自不同版本的代码。
**排查**:
```python
ckpt = torch.load(path, weights_only=True)
model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt.keys())
print("Only in model:", model_keys - ckpt_keys)
print("Only in ckpt:", ckpt_keys - model_keys)
# 找 shape 不同的 key
for k in model_keys & ckpt_keys:
    if model.state_dict()[k].shape != ckpt[k].shape:
        print(f"SHAPE MISMATCH {k}: model {model.state_dict()[k].shape} vs ckpt {ckpt[k].shape}")
```
**解决**: `strict=False` 跳过不匹配的 key; 或用 `load_state_dict(ckpt, strict=False)` + 手动处理。

### 2. `nn.Sequential` 内的层无法按名字访问
**现象**: `model.layer1` 拿不到 Sequential 内的第一个 Linear。
**原因**: `nn.Sequential` 把子模块存为 `0`, `1`, `2`... 而不是传入的变量名。
```python
model = nn.Sequential(
    nn.Linear(10, 20),  # 存储在 model[0], 不是 model.linear
    nn.ReLU(),
)
```
**解决**: 用 `nn.ModuleDict` 或 `nn.ModuleList` + 自定义 `__init__` 给子模块命名:
```python
self.linear1 = nn.Linear(10, 20)
self.relu = nn.ReLU()
```

### 3. `register_buffer` 漏了 `persistent=False`
**现象**: 保存 checkpoint 很大，包含不需要的 buffer（如 optimizer step count）。
**原因**: 默认 `persistent=True` → buffer 被 `state_dict()` 收集。
**解决**: 只在需要 checkpoint 的 buffer 用 `persistent=True`:
```python
self.register_buffer("step_count", torch.tensor(0), persistent=False)
```

### 4. Hook 循环引用导致显存泄漏
**现象**: 训练稳定后显存持续增长。
**原因**: Hook 闭包捕获了 tensor → tensor 持有 hook → 循环引用 → GC 无法回收。
**排查**:
```python
import gc
for obj in gc.get_objects():
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        if obj.untyped_storage().size() > 1e6:  # >1MB
            # 追踪引用链
            print(torch.cuda.memory_summary())
```
**解决**: Hook 中用 `weakref` 避免强引用; 或及时 `handle.remove()`。

### 5. Module 的 `training` 属性不同步
**现象**: 子模块的 BN 在 `model.eval()` 后仍在更新 running stats。
**原因**: 手动修改了 `m.training = False` 但没递归子模块; 或者子模块被外部引用且单独设为了 `training=True`。
**解决**: 始终用 `model.train()` / `model.eval()` 递归设置, 不要手动改 `model.training`。

