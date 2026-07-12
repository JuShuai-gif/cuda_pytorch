# IValue 与 JIT 类型系统：Type Erasure 实践

> 源码: `aten/src/ATen/core/IValue.h`, `torch/csrc/jit/`

## 0. 一句话总览

IValue 是 PyTorch JIT 的通用值容器，用 tagged union（类型擦除）在 C++ 中实现动态类型。一个 IValue 可以存 Tensor、int、double、string、list、dict、甚至完整的 Module。它是 TorchScript 解释器的核心数据结构。

## 1. 最小例子

```python
import torch

# Python 侧 track JIT types
@torch.jit.script
def f(x: torch.Tensor, n: int) -> torch.Tensor:
    return x * n

print(f"Schema: {f.schema}")

# IValue 在 C++ 侧承载这些参数:
# IValue(tensor), IValue(int) -> IValue(result_tensor)
```

## 2. 实战例子

### 2.1 IValue 的 5 种常见形式

```python
import torch

# 在 TorchScript 中，所有值都是 IValue:
# 1. Tensor -> IValue::Tensor
x = torch.randn(3)

# 2. int -> IValue::Int
n = 42

# 3. float -> IValue::Double
f = 3.14

# 4. list[Tensor] -> IValue::TensorList
tensors = [torch.randn(3), torch.randn(3)]

# 5. dict[str, Tensor] -> IValue::GenericDict
state = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}

# 6. Optional[Tensor] -> IValue::None
opt = None

# JIT trace 把所有这些打包成 IValue 传给解释器
```

### 2.2 TorchScript 类型推断

```python
import torch

class MyModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, flag: bool = False) -> torch.Tensor:
        if flag:
            return x * 2
        return x + 1

model = MyModule()

# Script 会分析所有可能的输出类型
scripted = torch.jit.script(model)
print(scripted.graph)

# 每个节点产出 IValue，类型在编译时确定
```

### 2.3 IValue 的 tagged union 结构

```cpp
// IValue.h (简化)
class IValue {
    union Payload {
        at::Tensor t;
        int64_t i;
        double d;
        c10::intrusive_ptr<Object> o;
        // ... 20+ types
    };
    Tag tag_;  // 标识当前类型
public:
    bool isTensor() const { return tag_ == Tag::Tensor; }
    bool isInt() const { return tag_ == Tag::Int; }
    Tensor toTensor() const { return payload_.t; }
};
```

## 3. 核心源码文件

```
aten/src/ATen/core/IValue.h               # IValue tagged union
aten/src/ATen/core/ivalue_inl.h           # IValue inline 实现
torch/csrc/jit/runtime/interpreter.cpp    # TorchScript 解释器, 消费 IValue
torch/csrc/jit/ir/ir.h                    # JIT IR (Value = IValue 的引用)
```

## 4. 和已有笔记的连接

```
20_fx_graphs/     — FX Graph 不用 IValue (纯 Python), JIT 用 IValue (C++)
40_design_patterns/ — IValue 是 Type Erasure 设计模式的经典实现
14_dispatcher/    — Dispatcher 用 IValue 作为算子参数的通用载体
43_intrusive_ptr/ — IValue 内部用 intrusive_ptr 管理复杂对象
```

## 5. 搜索关键词

```bash
rg -n "class IValue" aten/src/ATen/core/IValue.h
rg -n "enum class Tag" aten/src/ATen/core/IValue.h
rg -n "isTensor\|isInt\|isDouble" aten/src/ATen/core/IValue.h
```
