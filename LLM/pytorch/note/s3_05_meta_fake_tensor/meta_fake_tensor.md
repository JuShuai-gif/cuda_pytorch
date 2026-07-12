# Meta Kernel、FakeTensor 与 SymInt: PyTorch 编译栈的形状推导基石

> FakeTensor: `torch/_subclasses/fake_tensor.py`
> Meta Kernel 注册: `torch/_meta_registrations.py`, `aten/src/ATen/native/MetaTensor.cpp`
> SymInt: `c10/core/SymInt.h`, `c10/core/SymNodeImpl.h`
> Shape 符号化: `torch/fx/experimental/symbolic_shapes.py`

## 0. 一句话总览

在不执行真实 kernel 的前提下推断张量 shape/dtype/stride/device 的能力，是 `torch.compile` 图捕获和编译优化的核心前提。Meta tensor 通过注册的 meta kernel 完成形状推导，FakeTensor 在此基础上叠加 FakeTensorMode 的 dispatch key 拦截，SymInt 则将静态 shape 提升为符号表达式以支持动态 shape。

## 1. 最小例子

Meta tensor 基本用法：

```python
import torch

x = torch.empty(2, 3, device="meta")
y = torch.empty(3, 4, device="meta")
z = x @ y

print(z)
print(z.device, z.shape, z.stride())
```

FakeTensor 用法：

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

with FakeTensorMode():
    x = torch.randn(2, 3, device="cuda")
    y = torch.randn(3, 4, device="cuda")
    z = x @ y
    print(type(z), z.device, z.shape)
```

SymInt 动态 shape 示例：

```python
import torch

def f(x, y):
    return (x + y).sum()

# 使用 dynamic=True 让 Dynamo 捕获符号 shape
compiled_f = torch.compile(f, dynamic=True)
out = compiled_f(torch.randn(4, 8), torch.randn(4, 8))
```

## 1.5 实战例子

### 1.5.1 用 FakeTensorMode 调试模型 shape 推导错误

当模型在 `torch.compile` 下报 shape 不匹配错误时，用 FakeTensorMode 隔离问题：

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

model = torch.nn.Linear(10, 5).cuda()
x = torch.randn(3, 10, device="cuda")

with FakeTensorMode():
    try:
        out = model(x)
        print("FakeTensor 前向成功:", out.shape)
    except Exception as e:
        # 比如某个 op 缺少 meta kernel 会在此抛出
        print(f"FakeTensor 捕获错误: {e}")
        # 错误类型: "no meta kernel registered for op"
```

如果某层报错，直接用 `device="meta"` 逐层测试，快速定位缺少 meta kernel 的算子。

### 1.5.2 验证自定义 op 能否支持 torch.compile

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

lib = torch.library.Library("myops", "DEF")
lib.define("my_add(Tensor x, Tensor y) -> Tensor")

@torch.library.impl("myops::my_add", "CPU")
def my_add_cpu(x, y):
    return x + y

# 检查是否有 meta kernel
try:
    with FakeTensorMode():
        x = torch.randn(2, 3, device="cuda")
        y = torch.randn(2, 3, device="cuda")
        z = torch.ops.myops.my_add(x, y)
    print("有 meta kernel, 可以 compile")
except RuntimeError as e:
    print(f"缺少 meta kernel: {e}")
    print("torch.compile 将无法使用此 op")
```

### 1.5.3 分析 SymInt guard 追踪 shape 动态性

观察 `torch.compile` 在动态 shape 下生成的 guard：

```python
import torch

def f(x, y):
    return (x @ y).sum()

compiled = torch.compile(f, dynamic=True, backend="eager")

# 第一次: 编译
out = compiled(torch.randn(4, 8), torch.randn(8, 3))

# 查看 Dynamo 编译时的 guard
import torch._dynamo as dynamo
explanation = dynamo.explain(f, torch.randn(4, 8), torch.randn(8, 3))
for guard in explanation.guards:
    print(guard)
# 输出类似: "local 'x' shape[0] == 4", "local 'y' shape[1] == 3"
```

当输入 shape 变化时，检查 guard 是否命中来判断 recompile 原因。频繁 recompilation 时可据此调整 `torch.compile(dynamic=False)` 或设置合理动态维度范围。

## 2. 从 Python API 到源码的调用链

```
torch.empty(2, 3, device="meta")
  -> aten::empty.meta (native_functions.yaml 定义)
  -> Dispatcher 命中 Meta dispatch key
  -> meta kernel: MetaTensor.cpp 中的 empty_meta()
  -> 仅设置 sizes, strides, dtype, device, 不分配数据

FakeTensorMode()
  -> 在 DispatchKeySet 中插入 Fake key
  -> 创建 tensor 时: 先走 meta kernel 推断 metadata
  -> 遇到 op: 走 meta kernel 获取输出 shape
  -> 将结果包装为 FakeTensor（记录 device 信息）

dynamic=True 场景:
  -> Dynamo 遇到动态维度
  -> 创建 SymInt 节点（torch/fx/experimental/symbolic_shapes.py）
  -> 形状变为 SymInt 表达式: s0, s0*2 等
  -> 编译后生成 guard: s0 == input.shape[0]
```

## 3. 核心源码文件

```
torch/_subclasses/fake_tensor.py              # FakeTensor / FakeTensorMode
torch/_meta_registrations.py                  # Python 侧 meta kernel 注册
aten/src/ATen/native/MetaTensor.cpp           # C++ 侧 meta kernel
aten/src/ATen/native/*Meta*.cpp              # 各算子专用 meta kernel
c10/core/SymInt.h                             # SymInt 类型定义
c10/core/SymNodeImpl.h                        # SymNode 基类
c10/core/SymBool.h                            # SymBool (符号布尔)
torch/fx/experimental/symbolic_shapes.py      # ShapeEnv, guard 生成, 符号化简
torch/fx/experimental/recording.py            # SymNode 操作录制
```

## 4. 关键机制源码解读

### 4.1 Meta tensor vs FakeTensor 的区别

| 特性 | Meta Tensor | FakeTensor |
|------|-------------|------------|
| device 属性 | `meta` | 保留原 device（如 `cuda:0`） |
| 数据 | 无（stub） | 无（stub） |
| 生命周期 | 显式 `device="meta"` 创建 | 在 `FakeTensorMode` context 下自动替换 |
| 作用 | 形状推导 | 模拟真实 device 的 tracing |
| dispatch key | `Meta` | `Fake` (比 `Meta` 优先级高) |

核心区别：FakeTensor 知道自己是 "fake 的 CUDA tensor" 还是 "fake 的 CPU tensor"，而 Meta tensor 只知道自己是 meta。FakeTensor 结合了 meta kernel 的形状推导能力和 device 追踪能力。

### 4.2 Meta kernel 注册路径

C++ meta kernel:

```cpp
// aten/src/ATen/native/MetaTensor.cpp
TORCH_META_FUNC(add)(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    set_output(self.sizes(), self.options());
}
```

Python meta kernel:

```python
# torch/_meta_registrations.py
@torch.library.impl("aten::add", "Meta")
def meta_add(self, other, alpha=1):
    return self.new_empty(self.size())
```

当一个 op 缺少 meta kernel 时，`torch.compile` 在 tracing 阶段报错 "no meta kernel registered"。原因是：Dynamo/AOTAutograd 需要在不执行实际计算时推断输出 shape，meta kernel 是唯一的形状信息来源。

### 4.3 SymInt 动态 shape

SymInt 是 `int` 的超集：它可以是普通整数，也可以是 `SymNode`（符号表达式）。

```cpp
// c10/core/SymInt.h
class SymInt {
    union {
        int64_t value_;           // 静态整数
        SymNode node_;            // 符号表达式节点
    };
    bool is_symbolic_;
};
```

`torch/fx/experimental/symbolic_shapes.py` 中的 `ShapeEnv` 管理符号形状：

```python
class ShapeEnv:
    def __init__(self):
        self.guards: List[Guard] = []
        self.replacements: Dict[SymInt, SymInt] = {}  # 简化

    def create_symint(self, source: Source, val: int) -> SymInt:
        # 创建符号整数，记录其来源
        ...
```

编译过程中，guard 形如 `s0 == input.shape[0]` 和 `s1 == input.shape[1]`，运行时逐一校验。

### 4.4 FakeTensorMode dispatch 流程

```
FakeTensorMode.__torch_dispatch__:
  1. 跳过已处理的 op
  2. 检查是否有 meta kernel
  3. 调用 meta kernel 获取输出 shape/dtype/strides
  4. 包装输出为 FakeTensor
  5. 如果 meta kernel 不存在 -> raise UnsupportedOp
```

## 5. 和已有笔记的连接

```
dynamo/         — Dynamo tracing 依赖 FakeTensor 做形状推断
fx_graphs/      — FX graph 中节点产出的 shape 信息来自 meta kernel
inductor/       — Inductor lowering 依赖 symbolic shape 做融合决策
torch.compile/  — torch.compile 流水线中 FakeTensor+SymInt 是图捕获的支柱
dispatcher/     — Meta/Fake dispatch key 在 Dispatcher 层级中
tensor/         — Tensor metadata 模型是形状推导的底层基础
```

## 6. 常见坑点

- **Meta kernel 缺失不会在 eager 模式下报错**，只在 `torch.compile` 或 `FakeTensorMode` 下暴露。
- **FakeTensorMode 嵌套使用可能死锁**：`with FakeTensorMode():` 内不要再套一层。
- **SymInt guard 爆炸**：动态维度过多时 guard 数量指数增长，导致 recompile 频繁。用 `torch.compile(dynamic=True)` 时谨慎。
- **自定义 op 必须注册 meta kernel** 才能在 `torch.compile` 中使用。
- **Meta tensor 没有数据**，所以 `x.numpy()`、`.item()` 等在 meta tensor 上会崩溃。
- **FakeTensor 的 device 是 fake 的**，`fake_tensor.device` 返回比如 `cuda:0`，但实际无内存分配。

## 7. 阅读源码时建议搜索的关键词

```bash
# 查看某个 op 是否有 meta kernel
rg -n "def meta_" torch/_meta_registrations.py | head -20

# 查找 C++ meta kernel
rg -n "TORCH_META_FUNC" aten/src/ATen/native/MetaTensor.cpp

# FakeTensorMode 主 dispatch 逻辑
rg -n "class FakeTensorMode" torch/_subclasses/fake_tensor.py

# SymInt 核心实现
rg -n "class SymInt" c10/core/SymInt.h

# guard 生成逻辑
rg -n "def guard" torch/fx/experimental/symbolic_shapes.py

# 缺少 meta kernel 时的错误抛出
rg -n "no meta kernel" torch/_subclasses/fake_tensor.py
```
