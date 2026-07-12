# nn 容器模块源码分析 — Sequential / ModuleList / ModuleDict / ParameterList / ParameterDict

> 源码: `/home/ghr/code/pytorch/torch/nn/modules/container.py` (1045 行)
> 依赖: `nn.Module.__setattr__` / `add_module` / `_modules` — 详见 [04_module](../04_module/)

## 0. 一句话总览

五个容器类**本身也是 `nn.Module`**，它们不新增参数存储机制，只是复用 `Module` 的
`_modules` / `_parameters` OrderedDict，把「一组子模块/参数」包装成一个可被
`.to()` / `.state_dict()` / `.parameters()` 递归遍历的整体。核心差异只在:
**有没有实现 `forward`** 以及 **用整数下标还是字符串 key 存储**。

---

## 一、五个容器的定位对比

| 类 | 存储字典 | 有 forward? | 索引方式 | 自动 wrap 成 Parameter? |
|----|---------|------------|---------|------------------------|
| `Sequential` | `_modules` | ✅ 链式调用 | `int` / `slice` | — |
| `ModuleList` | `_modules` | ❌ (fallback 报错) | `int` / `slice` | — |
| `ModuleDict` | `_modules` | ❌ | `str` | — |
| `ParameterList` | `_parameters` | ❌ (`__call__` 直接 raise) | `int` / `slice` | ✅ |
| `ParameterDict` | `_parameters` | ❌ (`__call__` 直接 raise) | `str` | ✅ |

**关键**: `ModuleList` / `ModuleDict` 故意**不定义 `forward`**（源码 :508 / :649 注释
"remove forward altogether to fallback on Module's _forward_unimplemented"），
直接调用它们会抛错 —— 它们只负责「注册」，forward 逻辑由你手写。

---

## 二、`Sequential` — 唯一实现 forward 的容器 (`:59`)

### 2.1 构造: 两种入参 (`:115`)

```python
# container.py:115
def __init__(self, *args):
    super().__init__()
    if len(args) == 1 and isinstance(args[0], OrderedDict):
        for key, module in args[0].items():
            self.add_module(key, module)          # 用 OrderedDict 的 key 命名
    else:
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)     # 用 "0","1","2"... 命名
```

这就是「`nn.Sequential` 内的层无法按变量名访问」的根源: 位置参数一律被命名为
`"0"`, `"1"`, ...，存进 `self._modules`。想要有名字就传 `OrderedDict`。

### 2.2 forward: 链式传递 (`:254`)

```python
# container.py:254
def forward(self, input):
    for module in self:            # __iter__ 遍历 _modules.values()
        input = module(input)      # 上一层输出 = 下一层输入
    return input
```

只支持**单输入单输出**的链式结构。多输入/分支/残差连接必须自己写 `Module` 或用
`ModuleList`。

### 2.3 索引与切片 (`:139`)

```python
# container.py:139
@_copy_to_script_wrapper
def __getitem__(self, idx: slice | int):
    if isinstance(idx, slice):
        return self.__class__(OrderedDict(list(self._modules.items())[idx]))  # 切片返回新 Sequential
    else:
        return self._get_item_by_idx(self._modules.values(), idx)             # 整数返回单个 Module
```

- `model[1:3]` → 返回**新的** `Sequential`（保留原 key）
- `model[0]` → 返回单个子模块
- `_get_item_by_idx` (:124) 支持负索引，越界抛 `IndexError`

### 2.4 增删导致的重新编号 (`:150`)

```python
# container.py:150  __delitem__
# 删除后重建 _modules 保持连续编号 "0","1","2"...
str_indices = [str(i) for i in range(len(self._modules))]
self._modules = OrderedDict(zip(str_indices, self._modules.values(), strict=True))
```

删掉中间某层后，剩余层会**重新连续编号**，不会留下空洞。`ModuleList.__delitem__`
(:398) 逻辑相同。

### 2.5 运算符重载: `+` `+=` `*` (`:167`-`:238`)

| 运算 | 方法 | 语义 |
|------|------|------|
| `a + b` | `__add__` (:167) | 拼接两个 Sequential → 新 Sequential |
| `a += b` | `__iadd__` (:189) | 原地拼接，按 `offset` 续编号 |
| `seq * 3` | `__mul__` (:201) | 重复 3 次 → 新 Sequential (因子须为正 int) |
| `seq *= 3` | `__imul__` (:222) | 原地重复 |

### 2.6 list 风格增改方法

- `append(module)` (:262) — 末尾追加，编号 `str(len(self))`
- `insert(index, module)` (:283) — 指定位置插入，后续元素后移
- `extend(iterable)` (:315) — 逐个 append，等价于 `+`
- `pop(key)` (:181) — 取出并删除

---

## 三、`ModuleList` — 只注册不定义 forward (`:341`)

### 3.1 用法: 手写 forward 里遍历

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):   # 可迭代
            x = self.linears[i // 2](x) + l(x)  # 可整数索引
        return x
```

**为什么不能直接用 Python `list`?** 普通 `list` 里的子模块**不会**被
`__setattr__` 检测到（04_module 第四节），因此不会注册进 `_modules`，导致
`.parameters()` / `.to(device)` / `.state_dict()` **看不到它们**。`ModuleList`
在 `add_module` 时正确注册。

### 3.2 `__iadd__` = `extend` (`:418`)

```python
# container.py:418
def __iadd__(self, modules):
    return self.extend(modules)
```

构造函数 `self += modules` (:370) 也走这条路 → `extend` (:492) 逐个
`add_module(str(offset + i), module)`。

### 3.3 折叠重复层的 `__repr__` (`:427`)

```python
# container.py:427  连续相同的层会压缩显示
# (0-9): 10 x Linear(in_features=10, out_features=10, bias=True)
```

打印含大量重复层的 `ModuleList` 时只显示一行 `(start-end): N x <repr>`，
避免刷屏。这是个值得借鉴的可读性技巧。

---

## 四、`ModuleDict` — 字符串 key 存子模块 (`:511`)

### 4.1 用法: 按名字选择分支

```python
self.choices = nn.ModuleDict({"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)})

def forward(self, x, choice, act):
    x = self.choices[choice](x)   # 按运行时 key 动态选层
    return x
```

### 4.2 有序性 (`:511` docstring)

`ModuleDict` 是**有序**的: 保留插入顺序; `update()` (:610) 若传
`OrderedDict` / `ModuleDict` / 键值对列表则保序，传普通无序 `dict` 则不保序。

### 4.3 dict 风格接口

`keys()` / `items()` / `values()` (:595-608)、`pop()` (:585)、`clear()` (:581)、
`__contains__` (:577) 全部委托给内部 `self._modules`。

---

## 五、`ParameterList` / `ParameterDict` — 存的是 Parameter (`:652` / `:800`)

### 5.1 自动 wrap 成 Parameter

```python
# ParameterList.__setitem__  container.py:715
def __setitem__(self, idx, param):
    idx = self._get_abs_string_index(idx)
    if isinstance(param, torch.Tensor) and not isinstance(param, Parameter):
        param = Parameter(param)      # 裸 Tensor 自动升级为 Parameter
    return setattr(self, str(idx), param)
```

`ParameterDict.__setitem__` (:860) 同理。**所有**添加入口（构造、`append`、
`extend`、`update`、下标赋值）最终都走 `__setitem__`，所以 wrap 逻辑只需写一处。
注意注释 (:719): 通过 `setattr()` 直接加的对象不走这里，不会被 wrap。

### 5.2 不能被调用

```python
# container.py:796
def __call__(self, *args, **kwargs):
    raise RuntimeError("ParameterList should not be called.")   # :797
# ParameterDict.__call__  :1030 同样 raise
```

它们纯粹是参数容器，没有 forward 语义，直接调用立即报错，防止误用。

### 5.3 `ParameterList` 用 `_size` 而非 `_modules` 计数 (`:684`)

```python
# container.py:682
def __init__(self, values=None):
    super().__init__()
    self._size = 0        # 自己维护长度，因为 parameter 存在 _parameters 里
    if values is not None:
        self += values
```

`ParameterDict` 则用 `self._keys` 字典 (:841) 维护顺序和成员集合。

### 5.4 `ParameterDict` 的集合运算 (`:1033`)

```python
# container.py:1033
d1 | d2    # __or__  → 合并成新 ParameterDict
d1 |= d2   # __ior__ → 原地合并
```

---

## 六、关键源码位置速查

| 机制 | 类 | 行号 |
|------|----|----|
| `Sequential.__init__` (两种入参) | Sequential | 115 |
| `Sequential.forward` (链式) | Sequential | 254 |
| `Sequential.__getitem__` (切片) | Sequential | 139 |
| `Sequential.__delitem__` (重编号) | Sequential | 150 |
| `Sequential.__add__` / `__mul__` | Sequential | 167 / 201 |
| `Sequential.append/insert/extend` | Sequential | 262 / 283 / 315 |
| `ModuleList.__init__` | ModuleList | 367 |
| `ModuleList.extend` | ModuleList | 492 |
| `ModuleList.__repr__` (折叠) | ModuleList | 427 |
| `ModuleDict.update` (保序) | ModuleDict | 610 |
| `ModuleDict.keys/items/values` | ModuleDict | 595-608 |
| `ParameterList.__setitem__` (wrap) | ParameterList | 715 |
| `ParameterList.__call__` (raise) | ParameterList | 796 |
| `ParameterDict.__setitem__` (wrap) | ParameterDict | 860 |
| `ParameterDict.__or__` | ParameterDict | 1033 |
| 已废弃的 `Container` | Container | 52 |

---

## 七、可借鉴的工程技巧

1. **单一写入口** (`:715` / `:860`): `ParameterList/Dict` 让所有添加路径都收敛到
   `__setitem__`，wrap-to-Parameter 逻辑只写一次，杜绝遗漏。

2. **删除后重建保编号** (`:150`): 用 `zip(str_indices, values)` 重建 OrderedDict，
   保证下标始终连续，索引语义稳定。

3. **repr 折叠重复块** (`:427`): 大列表打印压缩为 `(0-9): 10 x Linear(...)`，
   提升调试可读性。

4. **故意不实现 forward** (`:508`): `ModuleList/Dict` fallback 到
   `_forward_unimplemented` 报错，用类型约束表达「这不是可执行单元」的设计意图。

5. **`@_copy_to_script_wrapper`** (:139 等): 给 `__getitem__` / `__len__` /
   `__iter__` 打标记，让 TorchScript 编译时把这些魔术方法正确拷贝进脚本。

---

## 八、实战常见坑点

### 1. 用 Python `list`/`dict` 存子模块 → 参数丢失
**现象**: `model.to("cuda")` 后某些层还在 CPU; `optimizer` 收不到这些参数。
**原因**: 普通 `list = [nn.Linear(...)]` 里的模块不被 `__setattr__` 注册进 `_modules`。
**解决**: 用 `nn.ModuleList` / `nn.ModuleDict` 替代裸容器。
```python
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])   # 正确
# self.layers = [nn.Linear(10, 10) for _ in range(3)]                # 错误: 参数丢失
```

### 2. `nn.Sequential` 内的层无法按名字访问
**现象**: `model.linear1` 拿不到 Sequential 里的 Linear。
**原因**: 位置参数被命名为 `"0"`, `"1"`... (:121)。
**解决**: 传 `OrderedDict` 给命名，或改用显式属性:
```python
model = nn.Sequential(OrderedDict([("linear1", nn.Linear(10, 20)), ("relu", nn.ReLU())]))
model.linear1   # 现在可用
```

### 3. 直接调用 `ModuleList` / `ParameterList`
**现象**: `self.layers(x)` 抛 `NotImplementedError` 或 `RuntimeError`。
**原因**: 它们没有 forward (`:508`)，`ParameterList.__call__` 直接 raise (`:796`)。
**解决**: 在你自己的 forward 里手动遍历: `for l in self.layers: x = l(x)`。

### 4. 往 `ParameterList` 里存裸 Tensor 后忘了它已被 wrap
**现象**: 期望存进去的是普通 Tensor，结果 `.requires_grad` 变 True 且出现在 `parameters()` 里。
**原因**: `__setitem__` (:722) 自动把 `Tensor` 升级为 `Parameter`。
**解决**: 这是设计行为; 若确实要存不训练的张量，用 `register_buffer` 或 `ModuleList` 装包装 Module。

### 5. `ModuleDict` 用普通 `dict` update 后顺序错乱
**现象**: `named_parameters()` 顺序和预期不符，影响 checkpoint 对齐。
**原因**: `update()` 传普通无序 `dict` 不保序 (`:527` note)。
**解决**: 传 `OrderedDict` 或键值对列表 `[("k1", m1), ("k2", m2)]` 保序。

### 6. `Sequential + list` 报错
**现象**: `seq + [layer]` 抛 `ValueError`。
**原因**: `Sequential.__add__` (:167) 只接受另一个 `Sequential`。
**解决**: 用 `seq.append(layer)` / `seq.extend([...])`，或 `seq + nn.Sequential(layer)`。
