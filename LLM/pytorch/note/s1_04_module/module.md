# nn.Module 完全源码分析

> 源码路径: `/home/ghr/code/pytorch/torch/nn/modules/module.py` (~3050 行)
> C++ 绑定: `/home/ghr/code/pytorch/torch/csrc/Module.cpp`

---

## 零、一句话总览

`nn.Module` 本质是一部**状态机 + 钩子系统**：四个 `dict`/`OrderedDict` 管理参数、buffer、子模块、hooks，`__call__` 在 `forward` 前后插入钩子执行点。

---

## 一、所有实例变量（`__init__` 中初始化，L466-508）

```python
# L489-505 (使用 super().__setattr__ 绕过 Module.__setattr__ 的自动检测逻辑)
super().__setattr__("training", True)                           # bool: train/eval 标志
super().__setattr__("_parameters", {})                           # dict[str, Parameter|None]
super().__setattr__("_buffers", {})                              # dict[str, Tensor|None]
super().__setattr__("_non_persistent_buffers_set", set())        # set[str]: 不进入 state_dict 的 buffer 名
super().__setattr__("_backward_pre_hooks", OrderedDict())        # {id: hook}
super().__setattr__("_backward_hooks", OrderedDict())            # {id: hook}
super().__setattr__("_is_full_backward_hook", None)              # None | True, 标记 backward hook 类型
super().__setattr__("_forward_hooks", OrderedDict())             # {id: hook}
super().__setattr__("_forward_hooks_with_kwargs", OrderedDict()) # {id: hook}
super().__setattr__("_forward_hooks_always_called", OrderedDict()) # {id: hook}
super().__setattr__("_forward_pre_hooks", OrderedDict())         # {id: hook}
super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict()) # {id: hook}
super().__setattr__("_state_dict_hooks", OrderedDict())          # {id: hook}
super().__setattr__("_state_dict_pre_hooks", OrderedDict())      # {id: hook}
super().__setattr__("_load_state_dict_pre_hooks", OrderedDict()) # {id: hook}
super().__setattr__("_load_state_dict_post_hooks", OrderedDict()) # {id: hook}
super().__setattr__("_modules", {})                              # dict[str, Module]
```

| 变量 | 类型 | 用途 |
|------|------|------|
| `training` | `bool` | train/eval 模式标志，默认 `True` |
| `_parameters` | `dict` | 参数名 → `Parameter` / `None` |
| `_buffers` | `dict` | buffer 名 → `Tensor` / `None` |
| `_modules` | `dict` | 子模块名 → `Module` |
| `_non_persistent_buffers_set` | `set` | 不进入 `state_dict` 的 buffer 名集合 |
| `_forward_pre_hooks` | `OrderedDict` | forward 前执行的 hook |
| `_forward_pre_hooks_with_kwargs` | `OrderedDict` | 收 kwargs 的 forward pre-hook |
| `_forward_hooks` | `OrderedDict` | forward 后执行的 hook |
| `_forward_hooks_with_kwargs` | `OrderedDict` | 收 kwargs 的 forward hook |
| `_forward_hooks_always_called` | `OrderedDict` | forward 异常时也执行的 hook |
| `_backward_pre_hooks` | `OrderedDict` | backward 前 hook |
| `_backward_hooks` | `OrderedDict` | backward 后 hook |
| `_is_full_backward_hook` | `None`/`True` | 标记是否为 full backward hook |
| `_state_dict_pre_hooks` | `OrderedDict` | `state_dict()` 前 hook |
| `_state_dict_hooks` | `OrderedDict` | `state_dict()` 后 hook |
| `_load_state_dict_pre_hooks` | `OrderedDict` | `load_state_dict()` 前 hook |
| `_load_state_dict_post_hooks` | `OrderedDict` | `load_state_dict()` 后 hook |

**注意**：`_parameters` / `_buffers` / `_modules` 是普通 `dict`，不是 `OrderedDict`（Python 3.7+ 保证插入顺序）。

---

## 二、构造函数 `__init__`（L466-508）

```python
def __init__(self, *args, **kwargs) -> None:
    torch._C._log_api_usage_once("python.nn_module")
    # 校验：call_super_init=False 时禁止传参
    if self.call_super_init is False and bool(kwargs):
        raise TypeError(...)
    # 初始化所有内部状态
    super().__setattr__("training", True)
    super().__setattr__("_parameters", {})
    # ... (所有 17 个变量，见上一节)
    super().__setattr__("_modules", {})
    if self.call_super_init:
        super().__init__(*args, **kwargs)  # 调用父类（object）__init__
```

关键设计：
- **不用 `self.training = True`**：因为 `Module.__setattr__` 有类型检测逻辑（Parameter/Module/Buffer 自动路由），初始化时这些 dict 还没建好，直接调用 `super().__setattr__` 跳过路由。
- `call_super_init` 是 `ScriptModule` 用的 flag，普通 `nn.Module` 默认为 `True`。

---

## 三、属性访问：`__setattr__` / `__getattr__` / `__delattr__`

### 3.1 `__setattr__`（L1944-2041）

`self.xxx = value` 时触发，按优先级路由：

```python
def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
    # 1. Parameter → register_parameter()
    if isinstance(value, Parameter):
        remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
        self.register_parameter(name, value)

    # 2. 名字已是 parameter，只允许 Parameter 或 None
    elif params is not None and name in params:
        if value is not None:
            raise TypeError(...)  # 只能赋 None

    # 3. Module → _modules[name] = value
    elif isinstance(value, Module):
        remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
        modules[name] = value

    # 4. 名字已是 module，只允许 Module 或 None
    elif modules is not None and name in modules:
        ...

    # 5. Buffer / Tensor → register_buffer()
    elif isinstance(value, Buffer) or buffers is not None and name in buffers:
        ...

    # 6. 其他类型 → 直接走 object.__setattr__
    else:
        super().__setattr__(name, value)
```

**重点**：
- `self.weight = nn.Parameter(...)` 自动存入 `_parameters["weight"]`，不是 `__dict__["weight"]`
- `self.sub = nn.Linear(3,3)` 自动存入 `_modules["sub"]`
- `self.cache = torch.zeros(10)`（torch >= 2.3）自动注册为 buffer（存入 `_buffers`）
- `self.extra_info = 42` 走普通 `__dict__`
- `remove_from` 会在新注册前清理旧类型：如果 `weight` 原来是 Parameter，`self.weight = nn.Linear(...)` 会把 `weight` 从 `_parameters` 删除，转入 `_modules`。**同名切换类型会静默迁移，这是常见 bug 源**

### 3.2 `__getattr__`（L1536-1551）

访问 `self.xxx` 时，如果 `__dict__` 找不到，按顺序查找：

```python
def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
    if "_parameters" in self.__dict__:
        _parameters = self.__dict__["_parameters"]
        if name in _parameters:
            return _parameters[name]
    if "_buffers" in self.__dict__:
        _buffers = self.__dict__["_buffers"]
        if name in _buffers:
            return _buffers[name]
    if "_modules" in self.__dict__:
        modules = self.__dict__["_modules"]
        if name in modules:
            return modules[name]
    raise AttributeError(...)
```

查找顺序：**`_parameters` → `_buffers` → `_modules`**。

坑：如果 parameter 和 buffer 同名，parameter 优先。

### 3.3 `__delattr__`（L1652-1661）

```python
def __delattr__(self, name):
    if name in self._parameters:
        del self._parameters[name]
    elif name in self._buffers:
        del self._buffers[name]
        self._non_persistent_buffers_set.discard(name)
    elif name in self._modules:
        del self._modules[name]
    else:
        super().__delattr__(name)
```

删除时自动从对应的内部 dict 移除。

---

## 四、参数管理

### 4.1 `register_parameter`（L574-622）

```python
def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
    # 校验
    if "_parameters" not in self.__dict__:
        raise AttributeError("cannot assign parameter before Module.__init__() call")
    elif "." in name:   raise KeyError('parameter name can\'t contain "."')
    elif name == "":    raise KeyError('parameter name can\'t be empty string ""')
    elif hasattr(self, name) and name not in self._parameters:
        raise KeyError(f"attribute '{name}' already exists")
    # 类型检查
    if param is None:
        self._parameters[name] = None                # None 允许（占位）
    elif not isinstance(param, Parameter):
        raise TypeError(...)
    elif param.grad_fn:
        raise ValueError("Cannot assign non-leaf Tensor to parameter '{name}'.")
    else:
        for hook in _global_parameter_registration_hooks.values():
            output = hook(self, name, param)
            if output is not None:
                param = output
        self._parameters[name] = param
```

关键校验：
- 名字不能含 `.`（因为 `state_dict` 用 `.` 分隔层级）
- 不能有同名其他属性
- `param.grad_fn` 必须为 `None`（必须是叶子张量，不能是运算结果）
- 支持 `None` 占位（`_apply` 会跳过 None）

### 4.2 `register_buffer`（L512-572）

```python
def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
    ...
    self._buffers[name] = tensor
    if persistent:
        self._non_persistent_buffers_set.discard(name)   # 确保不在非持久集合中
    else:
        self._non_persistent_buffers_set.add(name)        # 标记为非持久
```

- `persistent=True`（默认）：buffer 出现在 `state_dict()` 中（如 `BatchNorm.running_mean`）
- `persistent=False`：buffer 不出现在 `state_dict()` 中（如 optimizer step counter）
- `Buffer` 类（torch >= 2.x）可带 `persistent` 属性：`Buffer(tensor, persistent=False)`

### 4.3 `add_module`（L233-260）

```python
def add_module(self, name: str, module: Optional["Module"]) -> None:
    if not isinstance(module, Module) and module is not None:
        raise TypeError("{} is not a Module subclass".format(torch.typename(module)))
    if isinstance(module, Module):
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
    self._modules[name] = module
```

等同于 `self.name = module`（不走 `__setattr__` 的自动删除逻辑）。

---

## 五、前向执行：`__call__` → `_wrapped_call_impl` → `_call_impl` → `forward`

### 5.1 `__call__`（L1917）

```python
__call__: Callable[..., Any] = _wrapped_call_impl
```

**类属性别名**，不是实例方法。所有 `model(x)` 调用都路由到 `_wrapped_call_impl`。

### 5.2 `_wrapped_call_impl`（L1356-1362）

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)  # torch.compile 路径
    else:
        return self._call_impl(*args, **kwargs)           # 正常 hook 路径
```

`_compiled_call_impl` 由 `model.compile()`（L2603）设置，不为 `None` 时跳过所有 Python hook，走 compiled graph。

### 5.3 `_call_impl` 的六个阶段（L1364-1500）

```python
def _call_impl(self, *args, **kwargs):
    # FAST PATH: 没有任何 hooks 时直接调用 forward，零开销
    if not (self._backward_hooks or self._backward_pre_hooks or
            self._forward_hooks or self._forward_pre_hooks or ...):
        return forward_call(*args, **kwargs)

    # Phase 1: forward pre-hooks（L1376-1397）
    for hook_id, hook in self._forward_pre_hooks.items():
        if hook_id in self._forward_pre_hooks_with_kwargs:
            result = hook(self, args, kwargs)
            if result is not None:
                args, kwargs = result
        else:
            result = hook(self, args)
            if result is not None:
                args = result if isinstance(result, tuple) else (result,)

    # Phase 2: setup backward hooks（L1399-1402）
    bw_hook = None
    if self._backward_hooks or self._backward_pre_hooks:
        bw_hook = BackwardHook(self, self._backward_pre_hooks, self._backward_hooks)
        args = bw_hook.setup_input_hook(args)

    # Phase 3: forward()（L1404）
    forward_call = self._slow_forward if tracing else self.forward
    result = forward_call(*args, **kwargs)

    # Phase 4: forward hooks（L1405-1420）
    for hook_id, hook in self._forward_hooks.items():
        if hook_id in self._forward_hooks_with_kwargs:
            hook_result = hook(self, args, kwargs, result)
        else:
            hook_result = hook(self, args, result)
        if hook_result is not None:
            result = hook_result

    # Phase 5: setup output hooks（L1422-1426）
    if bw_hook is not None:
        result = bw_hook.setup_output_hook(result)

    # Phase 6: exception handling（L1428-1500）
    # 如果有 always_call=True 的 forward hooks 未执行，在异常时补执行
```

**执行流程图**：

```
model(x)
  │
  ├─ FAST PATH: 无 hook → 直接 forward()
  │
  └─ HOOK PATH:
       ├─ [1] forward_pre_hooks          → 修改 args/kwargs
       ├─ [2] setup backward hooks       → BackwardHook 包装
       ├─ [3] forward()                  → 用户逻辑
       ├─ [4] forward_hooks              → 修改 output
       ├─ [5] setup output hooks         → grad 追踪
       └─ [6] exception handlers         → always_call hooks 保底
```

---

## 六、Hook 系统（5 类 + 状态 dict hooks）

### 6.1 `register_forward_pre_hook`（L1201-1265）

```python
handle = module.register_forward_pre_hook(hook, prepend=False, with_kwargs=False)
```

- `hook(module, args) -> None or modified_args`（`with_kwargs=False`）
- `hook(module, args, kwargs) -> None or (new_args, new_kwargs)`（`with_kwargs=True`）
- 返回值非 `None` 则替换 args
- `prepend=True` 插入到队列开头（默认 append）

### 6.2 `register_forward_hook`（L1267-1333）

```python
handle = module.register_forward_hook(hook, prepend=False, with_kwargs=False, always_call=False)
```

- `hook(module, args, output) -> None or modified_output`
- `always_call=True`：即使 forward 抛异常也会执行（用于 debug/logging）
- 返回值非 `None` 则替换 output

### 6.3 `register_full_backward_hook`（L1041-1101）

```python
handle = module.register_full_backward_hook(hook, prepend=False)
```

- `hook(module, grad_input, grad_output) -> None or modified_grad_input`
- 使用 `BackwardHook` 包装器在输入/输出 tensor 上注册 `Tensor.register_hook()`
- 获得完整的 `grad_input` 和 `grad_output` tuple

### 6.4 `register_full_backward_pre_hook`（L966-1013）

```python
handle = module.register_full_backward_pre_hook(hook, prepend=False)
```

- `hook(module, grad_output) -> None or modified_grad_output`
- 在 backward 计算前执行，可以修改 `grad_output`

### 6.5 `register_backward_hook`（L1015-1039，已废弃）

```python
handle = module.register_backward_hook(hook)  # 废弃，改用 full 版本
```

- 旧版 backward hook，不支持完整的 grad_input/grad_output
- 用 `_maybe_warn_non_full_backward_hook` 发出警告

### 6.6 State Dict Hooks

```python
# state_dict 前后
handle = module.register_state_dict_pre_hook(hook)    # hook(module, prefix, keep_vars)
handle = module.register_state_dict_post_hook(hook)   # hook(module, destination, prefix, local_metadata)

# load_state_dict 前后
handle = module.register_load_state_dict_pre_hook(hook)  # hook(module, state_dict, prefix, ...)
handle = module.register_load_state_dict_post_hook(hook) # hook(module, incompatible_keys)
```

### 6.7 Hook 的内部存储

所有 hook 存储在 `OrderedDict` 中，key 是 `RemovableHandle` 的整数 id：

```python
model._forward_pre_hooks:   OrderedDict = {handle_id: hook_func}
model._forward_hooks:       OrderedDict = {handle_id: hook_func}
model._backward_hooks:      OrderedDict = {handle_id: hook_func}
```

**顺序保证**：注册顺序 = 执行顺序（`prepend=False`），`prepend=True` 插入队首。

**`RemovableHandle` 原理**（`torch.utils.hooks.RemovableHandle`）：
```python
class RemovableHandle:
    id: int        # 自增唯一 id
    hooks_dict: dict  # 弱引用到 hook 所在的 OrderedDict
    def remove(self):
        del self.hooks_dict[self.id]  # 从 OrderedDict 删除自己
```

---

## 七、状态序列化：`state_dict` / `load_state_dict`

### 7.1 `state_dict`（L2154-2241）

```python
def state_dict(self, destination=None, prefix="", keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    local_metadata = dict(version=self._version)
    # 1. state_dict_pre_hooks
    for hook in self._state_dict_pre_hooks.values():
        hook(self, prefix, keep_vars)
    # 2. 保存本层参数 + persistent buffers
    self._save_to_state_dict(destination, prefix, keep_vars)
    # 3. 递归子模块
    for name, module in self._modules.items():
        if module is not None:
            module.state_dict(destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
    # 4. state_dict_hooks
    for hook in self._state_dict_hooks.values():
        hook_result = hook(self, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination
```

### 7.2 `_save_to_state_dict`（L1719-1750）

```python
def _save_to_state_dict(self, destination, prefix, keep_vars):
    # 保存参数
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    # 保存 persistent buffers（跳过 non_persistent）
    for name, buf in self._buffers.items():
        if buf is not None and name not in self._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()
```

**`detach()` 的作用**：切断 autograd 图引用，防止保存 checkpoint 时持有完整的计算图。

### 7.3 `load_state_dict`（L2485-2598）

```python
def load_state_dict(self, state_dict, strict=True, assign=False):
    missing_keys, unexpected_keys, error_msgs = [], [], []
    def load(module, local_state_dict, prefix=""):
        module._load_from_state_dict(local_state_dict, prefix, local_metadata,
                                     True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {k: v for k, v in local_state_dict.items()
                                    if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)
        # 调用 post hooks
        for hook in module._load_state_dict_post_hooks.values():
            hook(module, _IncompatibleKeys(missing_keys, unexpected_keys))
    load(self, state_dict)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(...)
    return _IncompatibleKeys(missing_keys, unexpected_keys)
```

`_IncompatibleKeys` 是 `NamedTuple(missing_keys, unexpected_keys)`。

---

## 八、设备/类型转换

### 8.1 `_apply`（L912-1010）—— 所有转换的核心

```python
def _apply(self, fn, recurse=True):
    # 1. 递归子模块（可选）
    if recurse:
        for module in self.children():
            module._apply(fn)
    # 2. 转换 parameters，保持对象身份
    for key, param in self._parameters.items():
        if param is None: continue
        with torch.no_grad():
            param_applied = fn(param)
        # 三种策略：
        # a) swap_tensors: 适用于 subclass（FakeTensor 等）
        # b) param.data = param_applied: 保持对象身份（optimizer 持有引用）
        # c) 重建 Parameter: 不兼容类型时
    # 3. 转换 buffers（直接替换）
    for key, buf in self._buffers.items():
        self._buffers[key] = fn(buf)
    return self
```

三种转换策略（L917-975）：
1. **`swap_tensors`** (`should_use_swap_tensors=True`)：用于 `FakeTensor` subclass，交换底层 storage
2. **`param.data = new`** (`should_use_set_data=True`)：默认路径，保持 `Parameter` 对象身份不变，optimizer 仍持有引用
3. **`Parameter(new, requires_grad)`**：替换为新对象（不兼容类型时）

### 8.2 便捷方法（都调用 `_apply`）

```python
model.cuda(device=None)         # _apply(lambda t: t.cuda(device))
model.cpu()                     # _apply(lambda t: t.cpu())
model.float()                   # _apply(lambda t: t.float())
model.double()                  # _apply(lambda t: t.double())
model.half()                    # _apply(lambda t: t.half())
model.bfloat16()                # _apply(lambda t: t.bfloat16())
model.type(dst_type)            # _apply(lambda t: t.type(dst_type))
model.to(device, dtype, ...)    # _apply(lambda t: t.to(device, dtype, ...))
model.to_empty(device=device)   # _apply(lambda t: t.to(device) 但不复制数据)
```

**`model.to()` 特殊处理**（L1228）：
- 只转换 floating point / complex dtype（整数型 buffer 如 `num_batches_tracked` 不会被 `.half()` 转换）
- 支持 `memory_format=torch.channels_last`
- 返回 `self`（in-place 修改）

### 8.3 `model.compile()`（L2603）

```python
def compile(self, *args, **kwargs):
    # torch.compile 的 Module 级入口
    # 设置 self._compiled_call_impl
```

非 in-place，返回一个新的 `OptimizedModule` 或修改 `_compiled_call_impl`。

---

## 九、迭代器方法

### 9.1 `parameters()` / `named_parameters()`（L2229-2284）

```python
def parameters(self, recurse=True):
    for _, param in self.named_parameters(recurse=recurse):
        yield param

def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
    get_members_fn = lambda module: module._parameters.items()
    return self._named_members(get_members_fn, prefix=prefix, recurse=recurse,
                                remove_duplicate=remove_duplicate)
```

### 9.2 `buffers()` / `named_buffers()`（L2286-2338）

```python
def buffers(self, recurse=True):
    for _, buf in self.named_buffers(recurse=recurse):
        yield buf

def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
    get_members_fn = lambda module: module._buffers.items()
    return self._named_members(get_members_fn, ...)
```

### 9.3 `children()` / `named_children()`（L2340-2367）

```python
def children(self):
    for _, child in self.named_children():
        yield child

def named_children(self):
    memo = set()
    for name, module in self._modules.items():
        if module is not None and module not in memo:
            memo.add(module)
            yield name, module
```

**`children()` 会有去重**：如果同一个 Module 对象被多次赋值（如 `self.a = self.b = m`），只 yield 一次。

### 9.4 `modules()` / `named_modules()`（L2369-2443）

```python
def modules(self):
    for _, module in self.named_modules():
        yield module

def named_modules(self, memo=None, prefix='', remove_duplicate=True):
    if memo is None:
        memo = set()
    if self not in memo:
        memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            yield from module.named_modules(memo, submodule_prefix, remove_duplicate)
```

**根模块也会出现在 `named_modules()` 中**，key 是 `""`（空字符串）。

### 9.5 `_named_members` 去重机制（L2600-2618）

```python
def _named_members(self, get_members_fn, prefix="", recurse=True, remove_duplicate=True):
    memo = set()
    modules = self.named_modules(...) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue          # 跳过 None 和已出现的 tensor
            if remove_duplicate:
                memo.add(v)       # 共享参数只出现一次
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v
```

**权重绑定（weight tying）的支持**：`embed.weight` 和 `head.weight` 如果是同一个 `Parameter` 对象，`parameters()` 只返回一次。

---

## 十、训练模式

### 10.1 `train()` / `eval()`（L2445-2483）

```python
def train(self, mode=True):
    self.training = mode
    for module in self.children():
        module.train(mode)
    return self

def eval(self):
    return self.train(False)
```

- 递归设置所有子模块
- 影响 `BatchNorm`、`Dropout` 等层的行为
- 返回 `self` 以支持链式调用

### 10.2 `requires_grad_()`（L2485-2506）

```python
def requires_grad_(self, requires_grad=True):
    for p in self.parameters():
        p.requires_grad_(requires_grad)
    return self
```

用于冻结/解冻参数（fine-tuning）：

```python
# 冻结 backbone
model.backbone.requires_grad_(False)
# 只优化 head
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
```

### 10.3 `zero_grad()`（L2508-2534）

```python
def zero_grad(self, set_to_none=True):
    for p in self.parameters():
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()
```

`set_to_none=True`（默认）：释放 grad tensor 内存（推荐，节省显存）
`set_to_none=False`：保留 grad tensor 但设为 0

---

## 十一、其他重要方法

### 11.1 `apply()`（L616-654）

```python
def apply(self, fn):
    for module in self.children():
        module.apply(fn)
    fn(self)
    return self
```

遍历所有子模块（包括自身），对每个模块调用 `fn(module)`。用于：

```python
# 自定义初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

model.apply(init_weights)
# → Linear: init_weights(linear)
# → Conv2d: init_weights(conv2d)
# → root:   init_weights(root)
```

与 `_apply` 的区别：
- **`apply(fn)`**：`fn(module)`——处理的是 Module 本身
- **`_apply(fn)`**：`fn(param)`——处理的是 Parameter/Buffer tensor

### 11.2 `get_submodule()` / `get_parameter()` / `get_buffer()`（L265-480）

```python
sub = model.get_submodule("features.block1.conv3")   # 用点号路径获取子模块
param = model.get_parameter("features.block1.conv3.weight")
buf = model.get_buffer("features.block1.bn.running_mean")
```

实现：
```python
def get_submodule(self, target: str) -> "Module":
    if target == "":
        return self
    atoms = target.split(".")
    mod = self
    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(...)
        mod = getattr(mod, item)
        if not isinstance(mod, Module):
            raise TypeError(...)
    return mod
```

### 11.3 `set_submodule()`（L330-410）

```python
model.set_submodule("features.block1", new_block)
```

用点号路径设置子模块。

### 11.4 `extra_repr()`（L2543-2550）

```python
def extra_repr(self) -> str:
    return ""
```

在 `__repr__` 中显示额外信息。子类重写以显示超参：

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"
    # __repr__ → "MyLinear(in_features=10, out_features=5)"
```

### 11.5 `__repr__()`（L2552-2575）

```python
def __repr__(self):
    extra_lines = self.extra_repr().split("\n")
    if len(extra_lines) == 1:
        return f"{self._get_name()}({extra_lines[0]})"
    else:
        return f"{self._get_name()}({extra_lines[0]}\n" + \
               "\n".join(f"  {line}" for line in extra_lines[1:]) + "\n)"
```

```
# 单行
Linear(in_features=10, out_features=5, bias=True)

# 多行
Sequential(
  (0): Linear(...)
  (1): ReLU()
)
```

### 11.6 `share_memory()`（L2536-2538）

在多进程场景下共享参数（`multiprocessing.Queue` 等）。

### 11.7 `_get_name()`（L2540-2541）

返回类名（用于 `__repr__`）：

```python
def _get_name(self):
    return self.__class__.__name__
```

---

## 十二、C++ 绑定：`torch/csrc/Module.cpp`

`nn.Module` 的 C++ 绑定层在 `torch/csrc/Module.cpp`，主要作用：

1. **Python 到 C++ 的 ABI 桥接**：`__call__` 最终调用 C++ 的 `forward()` 实现
2. **`ScriptModule`**：TorchScript 模块的 C++ 端，`call_super_init=False`
3. **梯度与 autograd 注册**：hook 系统的底层实现
4. **设备/类型转换**：`_apply` 最终调用 C++ 的 tensor 操作

---

## 十三、完整的方法速查表

| 方法 | 行号 | 作用 |
|------|------|------|
| `__init__` | 466 | 初始化 17 个内部变量 |
| `__call__` | 1917 | `model(x)` 入口，别名到 `_wrapped_call_impl` |
| `_wrapped_call_impl` | 1356 | 检查 `_compiled_call_impl`，选择 compile/hook 路径 |
| `_call_impl` | 1364 | 6 阶段执行：pre-hooks → backward setup → forward → hooks → output hook → exception |
| `forward` | — | 由子类实现，`_call_impl` 调用 |
| `_slow_forward` | 1336 | tracing 时的 fallback forward |
| `__setattr__` | 1944 | 自动路由 Parameter/Module/Buffer 到内部 dict |
| `__getattr__` | 1536 | 按 `_parameters` → `_buffers` → `_modules` 查找 |
| `__delattr__` | 1652 | 从内部 dict 删除 |
| `register_parameter` | 574 | 注册 Parameter，校验叶子张量 |
| `register_buffer` | 512 | 注册 Buffer，支持 `persistent` 控制 |
| `add_module` | 233 | 注册子模块 |
| `register_forward_pre_hook` | 1201 | forward 前 hook |
| `register_forward_hook` | 1267 | forward 后 hook（支持 `always_call`） |
| `register_full_backward_hook` | 1041 | backward 完整 grad 钩子 |
| `register_full_backward_pre_hook` | 966 | backward 前钩子 |
| `register_backward_hook` | 1015 | 旧版（废弃） |
| `register_state_dict_pre_hook` | 1706 | `state_dict` 前 |
| `register_state_dict_post_hook` | 1682 | `state_dict` 后 |
| `register_load_state_dict_pre_hook` | 1872 | `load_state_dict` 前 |
| `register_load_state_dict_post_hook` | 1884 | `load_state_dict` 后 |
| `_apply` | 912 | 核心转换引擎 |
| `to` | 1228 | 设备/类型转换（in-place） |
| `cuda` / `cpu` / `float` / `half` / ... | 657-800 | 便捷转换（调用 `_apply`） |
| `to_empty` | 802 | 空设备分配（不复制数据） |
| `state_dict` | 2154 | 导出参数 + persistent buffers |
| `_save_to_state_dict` | 1719 | 实际保存逻辑 |
| `load_state_dict` | 2485 | 导入状态 |
| `_load_from_state_dict` | 1913 | 递归加载 |
| `parameters` | 2229 | 迭代所有参数 |
| `named_parameters` | 2254 | 带名字迭代参数 |
| `buffers` | 2286 | 迭代所有 buffer |
| `named_buffers` | 2309 | 带名字迭代 buffer |
| `children` | 2340 | 直接子模块 |
| `named_children` | 2349 | 带名字的子模块 |
| `modules` | 2369 | 递归所有模块（含自身） |
| `named_modules` | 2396 | 带名字的递归模块 |
| `_named_members` | 2600 | 去重通用实现 |
| `train` | 2445 | 设 training=True，递归 |
| `eval` | 2467 | 设 training=False |
| `requires_grad_` | 2485 | 设置所有参数的 `requires_grad` |
| `zero_grad` | 2508 | 清零梯度 |
| `apply` | 616 | 对每个子模块调用 `fn(module)` |
| `compile` | 2603 | `torch.compile` 入口 |
| `get_submodule` | 265 | 用 `.` 路径取子模块 |
| `set_submodule` | 330 | 用 `.` 路径设子模块 |
| `get_parameter` | 411 | 用 `.` 路径取参数 |
| `get_buffer` | 447 | 用 `.` 路径取 buffer |
| `get_extra_state` | 483 | 取自定义扩展状态 |
| `set_extra_state` | 504 | 设自定义扩展状态 |
| `extra_repr` | 2543 | `__repr__` 额外信息 |
| `__repr__` | 2552 | 模块字符串表示 |
| `share_memory` | 2536 | 共享内存（多进程） |
| `_get_name` | 2540 | 返回类名 |
| `__getstate__` / `__setstate__` | 1501-1534 | pickle 序列化 |

---

## 十四、可借鉴的工程技巧

1. **FAST PATH 设计**（`_call_impl` 开头）：无 hook 时跳过所有检查，零开销
2. **组合优于继承**：`RemovableHandle` 用组合包装，不改原有数据结构
3. **`detach()` 断引用**（`_save_to_state_dict`）：state_dict 用 `detach()` 保存，防止持有完整计算图
4. **保对象身份**（`_apply`）：用 `param.data = new` 而非新建 `Parameter`，因为 optimizer 持引用
5. **声明式钩子注册**：`__setattr__` 自动检测类型 → 自动路由，用户无需关心底层存储
6. **去重迭代器**（`_named_members`）：用 `memo = set()` 对 tensor 对象去重
7. **预初始化绕过**：`__init__` 用 `super().__setattr__` 而非 `self.x = x`，避免类型检测逻辑在 dict 尚未建好时触发

---

## 十五、实战常见坑点

### 1. `nn.Sequential` 内的层无法按名字访问

**现象**：`model.layer1` 拿不到 Sequential 内的第一个 Linear。
**原因**：`nn.Sequential` 把子模块存为 `0`, `1`, `2`... 而不是传入的变量名。

```python
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
# model[0], 不是 model.linear

# 用 ModuleDict 自定义命名
self.layers = nn.ModuleDict({
    "linear1": nn.Linear(10, 20),
    "relu": nn.ReLU(),
})
```

### 2. `state_dict` 键名不匹配

```python
ckpt = torch.load(path, weights_only=True)
model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt.keys())
print("Only in model:", model_keys - ckpt_keys)
print("Only in ckpt:", ckpt_keys - model_keys)
# 检查 shape
for k in model_keys & ckpt_keys:
    if model.state_dict()[k].shape != ckpt[k].shape:
        print(f"SHAPE MISMATCH {k}")
```

### 3. 同名属性类型冲突

```python
m = nn.Module()
m.register_parameter("weight", nn.Parameter(torch.randn(3)))
m.weight = nn.Linear(3, 3)   # 静默删除 _parameters["weight"]，转入 _modules["weight"]
# 参数丢了！optimizer 里的引用失效
```

### 4. `register_buffer` 漏了 `persistent=False`

```python
self.register_buffer("step_count", torch.tensor(0))              # 默认 persistent=True
self.register_buffer("step_count", torch.tensor(0), persistent=False)  # 正确
```

### 5. Non-leaf Tensor 注册为 Parameter

```python
# 错误：param 是运算结果，不是叶子张量
self.weight = nn.Parameter(x @ W.T)   # RuntimeError: grad_fn 不为 None

# 正确
self.weight = nn.Parameter(torch.randn(64, 256))
```

### 6. Hook 循环引用导致显存泄漏

**现象**：训练稳定后显存持续增长。
**原因**：Hook 闭包捕获了 tensor → tensor 持有 hook → 循环引用。
**解决**：用 `weakref` 或及时调用 `handle.remove()`。

### 7. `apply` 和 `_apply` 混淆

```python
# apply: fn(module) —— 对每个 Module 对象
model.apply(lambda m: print(m.__class__.__name__))

# _apply: fn(tensor) —— 对每个 Parameter/Buffer tensor
model._apply(lambda t: t.half())
```

### 8. 冻结后 optimizer 仍持有旧引用

```python
model.requires_grad_(False)
# 错误：optimizer 创建在冻结之前
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 正确：创建 optimizer 时用 filter
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
```

### 9. 参数转换后的状态保持

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.half()  # _apply 使用 param.data = new，保持 Parameter 对象身份
# optimizer 引用仍然有效！

model._apply(lambda t: Parameter(t.half(), requires_grad=t.requires_grad))
# 重建了 Parameter 对象，optimizer 引用失效
```

### 10. 子模块的 `training` 属性不同步

```python
model.eval()  # 递归设置所有子模块 training=False
# 不要手动改：
model.some_module.training = True  # 破坏递归一致性
```
