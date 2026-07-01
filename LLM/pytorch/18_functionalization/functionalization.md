# Functionalization: 编译器视角的 Mutation / View / Alias 语义处理

> C++ 核心: `aten/src/ATen/FunctionalTensorWrapper.cpp`、`FunctionalTensorWrapper.h`
> Fallback: `aten/src/ATen/native/FunctionalizeFallbackKernel.cpp`
> 与 functorch 关系: `torch/_functorch/eager_transforms.py`

## 0. 一句话总览

Functionalization 层将 PyTorch 的有副作用操作（in-place mutation、view alias）转化为纯函数式操作，使得计算图变成无副作用的 DAG，可以安全地被编译器（AOTAutograd / Inductor）做重排、融合、消除等变换。

## 1. 最小例子

```python
import torch

def f(x):
    y = x.view(-1)
    y.add_(1)         # in-place mutation, 通过 view 别名影响 x
    return x * 2

x = torch.ones(2, 3)
print(f(x))
print(x)  # x 被改变了
```

启用 functionalization：

```python
from torch._subclasses.functional_tensor import FunctionalTensorMode

with FunctionalTensorMode():
    x = torch.ones(2, 3)
    result = f(x)
    print(result)
    print(x)  # x 不会被修改（functionlization 产生了副本）
```

## 1.5 实战例子

### 1.5.1 用 FunctionalTensorMode 检测 mutation 导致的编译问题

当 `torch.compile` 下模型行为与 eager 不一致时，用 FunctionalTensorMode 隔离 mutation 问题：

```python
import torch
from torch._subclasses.functional_tensor import FunctionalTensorMode

# 一个有隐患的模型：in-place 操作可能导致图捕获错误
class BadModel(torch.nn.Module):
    def forward(self, x):
        x.add_(1)  # in-place mutation
        return x * 2

model = BadModel()
x = torch.randn(3)

# Eager 正常
out_eager = model(x)
print("Eager out:", out_eager)

# FunctionalTensorMode 下验证 mutation 是否被正确处理
with FunctionalTensorMode():
    x_fake = torch.randn(3).clone()
    out_fake = model(x_fake)
    print("Functional out:", out_fake)
    print("Original x after:", x_fake)  # 应保持不变
```

如果 FunctionalTensorMode 下结果异常，说明该 mutation 模式在编译时可能出错。

### 1.5.2 排查 torch.compile 下 view 导致的 alias 错误

当 `torch.compile` 报 "Cannot access data pointer of tensor that has been mutated" 时：

```python
import torch

def f(x):
    y = x[:, ::2]  # strided view
    z = y + 1
    x[:, ::2] = z   # 写回 slice
    return x.sum()

# Eager 正常
x = torch.ones(4, 4)
print("Eager:", f(x))

# Compile 可能报错
try:
    compiled_f = torch.compile(f)
    x = torch.ones(4, 4)
    print("Compiled:", compiled_f(x))
except Exception as e:
    print(f"Compile error: {e}")
    # 原因是: slice 赋值涉及 view + in-place, functionalization 需要展开为
    # view_copy + scatter + 写回, 某些复杂组合可能未被覆盖
```

解决方法：将 in-place 改写为显式 functional 操作，避免复杂 view + mutation 组合。

### 1.5.3 分析 Custom Op 的 alias annotation 错误

自定义 op 的 `native_functions.yaml` 中 alias annotation 漏标导致 functionalization 产生错误结果：

```yaml
# 错误的: 漏标 (a!)
- func: my_add_(Tensor self, Tensor other) -> Tensor
# 正确的:
- func: my_add_(Tensor(a!) self, Tensor other) -> Tensor(a!)
```

用以下方法验证 annotation 是否正确：

```python
import torch

# 重新注册正确的 annotation
lib = torch.library.Library("myops", "DEF")
lib.define("my_add_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

# 在 FunctionalTensorMode 下验证 mutation 被正确追踪
with FunctionalTensorMode():
    x = torch.ones(3)
    orig_id = id(x)
    torch.ops.myops.my_add_(x, torch.ones(3))
    assert id(x) == orig_id  # in-place, 同一对象
    # 如果 annotation 错误, x 可能未被正确 mutate
```

## 2. 从 Python API 到源码的调用链

```
y = x.view(-1)
y.add_(1)
    |
    v  (FunctionalTensorMode 激活时)
Dispatcher: key 中包含 Functionalize
    |
    v
FunctionalTensorWrapper::add_.Tensor (C++)
    |
    v  (Functionalization 改写)
1. 检测 y 是 x 的 view alias
2. 把 add_ 替换为:
   temp = x.view_copy(-1)      # view_copy 产生新 tensor
   temp = temp.add(1)          # 用 functional add 替代 add_
   x = x + 1 的等效替换        # 因为 x 也被 alias mutate
3. 更新别名映射表
    |
    v
输出 FunctionalTensorWrapper，记录 mutation 历史
```

## 3. 核心源码文件

```
aten/src/ATen/FunctionalTensorWrapper.cpp      # FunctionalTensorWrapper 实现
aten/src/ATen/FunctionalTensorWrapper.h         # 头文件
aten/src/ATen/native/FunctionalizeFallbackKernel.cpp  # fallback kernel
torch/_subclasses/functional_tensor.py          # Python 侧 FunctionalTensorMode
torch/_functorch/eager_transforms.py            # 与 function transform 的集成
aten/src/ATen/native/native_functions.yaml      # alias annotation 定义
aten/src/ATen/native/View.cpp                   # view op 的实现
```

## 4. 关键机制源码解读

### 4.1 native_functions.yaml 中的 alias annotation

```yaml
# view (别名, 不修改):
- func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)

# in-place (修改):
- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)

# out variant (写入提供的 tensor):
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
```

`(a)` 表示输出和输入是同一个 alias（view）。`(a!)` 表示输入被 mutate。Functionalization 通过这个标注来构建 alias graph：哪些 tensor 共享存储，哪些操作是 mutation。

### 4.2 Functionalization 如何改写 mutation

FunctionalTensorWrapper 内部维护一张 **alias map**：

```
原始 tensor x (id=1)
  └─ view alias y (id=2, base=id=1, offset=0)
```

当 `y.add_(1)` 发生时，Functionalization kernel 做以下操作：

1. 检查 y 是否有 base tensor → 发现 x
2. 将 `add_` 替换为：
   - `y' = y.view_copy(-1)` (detach alias)
   - `y'' = y'.add(1)` (functional add)
   - 记录 x 需要更新（因为 y 是 x 的别名）
3. 在 tensor wrapper 中保存 pending mutation
4. 最终触发 `sync()` 时，将 mutation 应用到 base tensor

### 4.3 view vs view_copy

在编译器上下文中，view 不被接受，因为 view 产生 aliased tensor，导致编译器无法确定内存是否独立。

```yaml
# view: 别名, 零拷贝, 不适合编译器
- func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)

# view_copy: 拷贝数据, 产生独立 tensor, 适合编译器
- func: view_copy(Tensor self, SymInt[] size) -> Tensor
```

Functionalization 会将 `view` 替换为 `view_copy` 加上 alias tracking，使得 graph 中只有 functional ops。

### 4.4 在 torch.compile 中的作用

```
Dynamo
  -> 捕获 FX graph (包含 add_、view 等)
  -> AOTAutograd 启用 Functionalization
  -> add_ 展开为 add + 赋值语义
  -> view 展开为 view_copy + 别名恢复
  -> 纯函数式 graph
  -> Partitioner + Inductor
```

## 5. 和已有笔记的连接

```
tensor/         — Tensor view/alias 语义是理解 functionalization 的基础
dispatcher/     — Functionalization 通过 Functionalize dispatch key 实现
autograd/       — Autograd 中的 in-place 检测与 functionalization 互补
torch.compile/  — Functionalization 是 torch.compile 图捕获的必经阶段
aot_autograd/   — AOTAutograd 依赖 functionalization 消除 mutation
meta_fake_tensor/ — FakeTensor 与 FunctionalTensor 都是通过 dispatch key 拦截
```

## 6. 常见坑点

- **Functionalization 不是默认开启的**，只有在 `torch.compile` 或显式 `FunctionalTensorMode` 下才激活。
- **Complex view 模式**（如多次 transpose + slice + expand）的 functionalization 成本高，可能产生大量 `view_copy`。
- **Alias annotation 错误**会导致 functionalization 产生不正确的结果。yaml 中漏标 `(a!)` 会导致 mutation 未被正确追踪。
- **FunctionalTensorWrapper 和 AutogradMeta 的交互**：autograd 中记录的版本号检测（version counter）在 functionalization 下行为不同。
- **自定义 op 如果包含内部 mutation 但未注册 alias annotation**，functionalization 会静默产生错误结果。

## 7. 阅读源码时建议搜索的关键词

```bash
# FunctionalTensorWrapper 核心实现
rg -n "class FunctionalTensorWrapper" aten/src/ATen/FunctionalTensorWrapper.h

# alias annotation 解析
rg -n "Tensor\(a\!\)" aten/src/ATen/native/native_functions.yaml | head -10

# view_copy 的注册
rg -n "view_copy" aten/src/ATen/native/native_functions.yaml

# functionalization fallback
rg -n "FunctionalizeFallbackKernel" aten/src/ATen/native/FunctionalizeFallbackKernel.cpp

# Python 侧 FunctionalTensorMode
rg -n "class FunctionalTensorMode" torch/_subclasses/functional_tensor.py

# sync mutation
rg -n "sync\b" aten/src/ATen/FunctionalTensorWrapper.cpp
```
