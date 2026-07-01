# torchgen: ATen 算子代码生成系统源码分析

> Python 端: `torchgen/gen.py`、`torchgen/model.py`、`torchgen/api/`
> Schema 定义: `aten/src/ATen/native/native_functions.yaml`
> 生成产物: `build/aten/src/ATen/RegisterCPU.cpp`、`RegisterCUDA.cpp`、`Functions.cpp`、`Operators.cpp`

## 0. 一句话总览

torchgen 是一个**代码生成器**：读取 `native_functions.yaml` 中算子 schema 和 dispatch 配置，为每个算子自动生成 C++ dispatch 注册代码、Python binding、以及 Autograd/Functionalization 的样板代码。PyTorch 中约 2000+ 个算子的注册代码几乎全部来自 torchgen。

## 1. 最小例子

```python
import torch

x = torch.randn(2, 3, device="cuda")
y = torch.randn(2, 3, device="cuda")
z = torch.add(x, y)

print(torch.ops.aten.add.Tensor)
print(torch._C._dispatch_dump_table("aten::add.Tensor"))
```

查看 schema 定义：

```bash
rg -n "func: add\\.Tensor|func: add\\.out" aten/src/ATen/native/native_functions.yaml
```

## 2. 实战例子

### 2.1 追踪自定义 op 的代码生成全过程

假设为一个自定义硬件后端添加 `silu` 算子的支持：

```yaml
# native_functions.yaml
- func: silu(Tensor self) -> Tensor
  variants: function
  dispatch:
    CPU: silu_kernel
    CUDA: silu_kernel
    MyBackend: silu_mybackend
```

运行 torchgen 后，检查生成的 `RegisterMyBackend.cpp`：

```bash
# 搜索生成的 dispatch 注册代码
rg -n "silu" build/aten/src/ATen/RegisterMyBackend.cpp
# 输出应包含:
#   m.impl("aten::silu", kMyBackend, &at::native::silu_mybackend);
```

当自定义后端编译报错 "undefined symbol: silu_mybackend" 时，从 `RegisterMyBackend.cpp` 可以定位到 kernel 函数签名，据此补齐实现。

### 2.2 分析算子 dispatch key 分发表排查性能问题

当 `torch.add` 在 CUDA 上意外走 CPU fallback 时（例如 tensor device 不匹配），查看 dispatch table：

```python
import torch

# 构造 device 不匹配的场景
cpu_tensor = torch.randn(3)
cuda_tensor = torch.randn(3, device="cuda")

# 查看 dispatch table
table = torch._C._dispatch_dump_table("aten::add.Tensor")
# 分析: 如果某 key 的 kernel 不是 CUDA 预期的，说明 dispatch 路径有误

# 实际排查: 打印每个 dispatcher key 的 kernel 名字
for line in table.split("\n"):
    if "CUDA" in line:
        print(f"CUDA kernel: {line}")
```

通过 `rg "add_kernel" build/aten/src/ATen/RegisterCUDA.cpp` 确认生成的 CUDA kernel 注册是否存在。

### 2.3 阅读生成代码理解算子注册结构

查看某个复杂算子的 torchgen 生成结果来理解结构化 kernel 的注册逻辑：

```bash
# conv2d 是 structured kernel
rg -n "func: conv2d" aten/src/ATen/native/native_functions.yaml

# 查看生成的 meta 函数
rg -n "conv2d" build/aten/src/ATen/NativeFunctions.h

# 查看注册到 Dispatcher 的完整 kernel 链
rg -n "conv2d" build/aten/src/ATen/RegisterCPU.cpp | head -20
```

通过阅读 `build/aten/src/ATen/RegisterCPU.cpp` 中 `conv2d` 的注册代码，可以理解 `TORCH_META_FUNC` -> `TORCH_IMPL_FUNC` 的调用链是如何被 torchgen 串联的。

```cpp
// RegisterCPU.cpp 中的典型注册 (简化)
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("conv2d", TORCH_FN(aten::native::conv2d));
    m.impl("conv2d.padding", TORCH_FN(aten::native::conv2d_padding));
}
```

## 3. 从 Python API 到源码的调用链

```
torch.add(x, y)
  -> torch/_torch.py: torch.add 是 C extension 绑定
  -> THPVariable_add (torch/csrc/autograd/python_variable.cpp)
  -> ATen 层: at::add (build/aten/src/ATen/Functions.cpp, 由 torchgen 生成)
  -> Dispatcher: 根据 dispatch key 查找 kernel
  -> 实际 kernel: 在 RegisterCPU.cpp / RegisterCUDA.cpp 中注册（由 torchgen 生成）
```

生成流程:

```
native_functions.yaml
       |
       v
torchgen/gen.py (入口, main())
       |
       v
torchgen/model.py (将 YAML 解析为 NativeFunction、DispatchKey 等 Python 数据类)
       |
       v
torchgen/api/ 目录下各 Codegen API 生成器
  ├── cpp.py      → C++ function signatures
  ├── python.py   → Python bindings (pybind11)
  ├── meta.py     → Meta kernel registration
  └── structured.py → Structured kernel support
       |
       v
build/aten/src/ATen/ 下生成:
  ├── Functions.cpp       → at::add() 等顶层 C++ API
  ├── Operators.cpp       → torch::add() 等 operators API
  ├── RegisterCPU.cpp     → CPU kernel 注册到 Dispatcher
  ├── RegisterCUDA.cpp    → CUDA kernel 注册到 Dispatcher
  ├── NativeFunctions.h   → 算子函数声明
  └── RegistrationBerkeley.cpp → 其他后端
```

## 4. 核心源码文件

```
aten/src/ATen/native/native_functions.yaml   # 算子 schema 主文件（~6000 行）
torchgen/gen.py                               # torchgen 入口逻辑
torchgen/model.py                             # NativeFunction / DispatchKey 等数据模型
torchgen/api/cpp.py                           # 生成 C++ API 签名
torchgen/api/python.py                        # 生成 Python binding 签名
torchgen/api/meta.py                          # 生成 Meta kernel 桩
torchgen/api/structured.py                    # 生成 structured kernel 注册
torchgen/api/functionalize.py                 # Functionalization 代码生成
torchgen/api/autograd.py                      # Autograd 样板代码生成
torchgen/selective_build.py                   # 选择性编译（移动端）
build/aten/src/ATen/Functions.cpp             # 生成产物: 顶层 C++ API
build/aten/src/ATen/Operators.cpp             # 生成产物: Operators API
build/aten/src/ATen/RegisterCPU.cpp           # 生成产物: CPU dispatch 注册
build/aten/src/ATen/RegisterCUDA.cpp          # 生成产物: CUDA dispatch 注册
build/aten/src/ATen/NativeFunctions.h         # 生成产物: 函数声明
```

## 5. 关键机制源码解读

### 5.1 native_functions.yaml schema 结构

一条典型算子定义：

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_kernel
    CUDA: add_kernel
    SparseCPU: add_sparse_cpu
    SparseCUDA: add_sparse_cuda
    CompositeImplicitAutograd: add_composite_implicit  # 可选
```

`func:` 定义算子名、参数、返回值。`variants:` 控制是否生成 Python 方法绑定。`dispatch:` 将 dispatch key 映射到具体 C++ kernel 函数。

### 5.2 CompositeImplicitAutograd vs CompositeExplicitAutograd

`CompositeImplicitAutograd` 等价于 Autograd key 的 fallback：该 key 的 kernel 同时包含前向和 autograd 逻辑，因此**不需要**单独注册 Autograd 公式。比如 `add.Tensor` 标记为 `CompositeImplicitAutograd`，意味着它通过其他已有 op 组合实现 autograd。

`CompositeExplicitAutograd` 则只提供前向逻辑，autograd 必须显式注册。两者的区别在于 dispatch table lookup 的优先级：`CompositeImplicitAutograd` 在 Autograd 之后 fallback，`CompositeExplicitAutograd` 在 Autograd 之前 fallback。

```yaml
# CompositeImplicitAutograd: 不需要额外 Autograd 注册
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_kernel
    CUDA: add_kernel
    CompositeImplicitAutograd: add

# CompositeExplicitAutograd: 需要显式注册 autograd
- func: mul.Tensor(Tensor self, Tensor other) -> Tensor
  dispatch:
    CPU: mul_kernel
    CUDA: mul_kernel
    CompositeExplicitAutograd: mul
```

### 5.3 out= variant 与 alias annotation

`out=` variant 让用户提供输出 tensor：

```yaml
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
```

`Tensor(a!)` 中的 `a!` 是 alias annotation：

- `Tensor(a)`: 返回和输入同 alias 的 view
- `Tensor(a!)`: 输入被 mutate（写操作）
- `Tensor(b)`: 独立的 alias，无别名关系

这些标注影响：**Functionalization**（能否把 mutation 转成纯函数调用）、**Autograd**（是否需要 in-place 检测）、**编译器**（能否安全消除副作用）。

### 5.4 torchgen/model.py 数据模型

```python
class NativeFunction:
    func: FunctionSchema          # 解析后的函数签名
    dispatch: Dict[DispatchKey, str]  # dispatch key -> kernel 名
    variants: Set[str]            # 生成方式
    structured: bool              # 是否为 structured kernel
    ...

class FunctionSchema:
    name: SchemaName              # 如 add.Tensor
    arguments: List[Argument]     # 参数列表
    returns: List[Return]         # 返回值
    ...
```

### 5.5 torchgen/gen.py 主流程

```
gen() ->
  1. parse_native_yaml() 读取 native_functions.yaml
  2. 解析出 List[NativeFunction]
  3. 按 dispatch key 分组 → 每组调用对应的 codegen backend
  4. 写入 build/aten/src/ATen/ 下的生成文件
```

## 6. 和已有笔记的连接

```
dispatcher/    — torchgen 生成的是 Dispatcher 的注册代码（RegisterCPU.cpp 等）
tensor/        — Tensor op 的 C++ API 由 torchgen 生成
autograd/      — Autograd 注册代码由 torchgen 生成（derivatives.yaml）
torch.compile/ — torch.compile 依赖的 meta kernel 注册也来自 torchgen
```

## 7. 常见坑点

- **不是每个 op 都必须 dispatch 到所有后端**：`dispatch: {}` 或缺少某个 key 时，Dispatcher 使用 fallback key（如 CPU fallback）。
- **YAML 中 `differentiability_license` 不影响代码生成**，只用于 autograd 维护者标记。
- `CompositeImplicitAutograd` 的 kernel 是 Python 可组合的，但性能通常低于专用后端 kernel。
- **`out=` op 必须和 functional op 共享 schema**，否则 torchgen 无法生成正确的 `at::add_out` 签名。
- **修改 native_functions.yaml 后必须重新 build**，因为生成的 .cpp 文件是 build 产物。
- Structured kernel (`structured: True`) 有额外的代码生成逻辑：torchgen 生成 `set_storage`、`set_meta` 等桩函数。

## 8. 阅读源码时建议搜索的关键词

```bash
# 查看某个 op 的完整 schema 定义
rg -n "func: add\\.Tensor|func: add\\.out" aten/src/ATen/native/native_functions.yaml

# 查看生成后的 dispatch 注册
rg -n "aten::add.Tensor" build/aten/src/ATen

# 查看 torchgen 对某个 dispatch key 的处理逻辑
rg -n "CompositeImplicitAutograd" torchgen/gen.py

# 查看算子实际 kernel 实现
rg -n "add_kernel|add_stub" aten/src/ATen/native

# 查看 YAML 解析逻辑
rg -n "def parse_native_yaml" torchgen/gen.py

# trace torchgen 代码生成的主流程
rg -n "def gen\b" torchgen/gen.py
```

> 参考: `torchgen/README.md` 有对 torchgen 架构的官方说明。
