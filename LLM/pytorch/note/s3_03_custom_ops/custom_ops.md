# Custom C++/CUDA Extension 与自定义 Op 注册

> Python 端: `torch/library.py`、`torch/utils/cpp_extension.py`
> Dispatcher: `aten/src/ATen/core/dispatch/Dispatcher.h`
> TensorIterator: `aten/src/ATen/native/TensorIterator.cpp`
> CUDA Loops: `aten/src/ATen/native/cuda/Loops.cuh`

## 0. 一句话总览

PyTorch 允许通过 `torch.library`（Python）或 `TORCH_LIBRARY`（C++）注册自定义算子。自定义 op 和内置 op 使用同一个 Dispatcher，共享 dispatch key 机制、Autograd 注册、Meta kernel 注册等整套基础设施。TensorIterator 是编写 elementwise kernel 的抽象框架，自动处理广播、类型提升、内存布局。

## 1. 最小例子

Python 侧自定义 op:

```python
import torch

lib = torch.library.Library("myops", "DEF")
lib.define("scale(Tensor x, float alpha) -> Tensor")

@torch.library.impl("myops::scale", "CPU")
def scale_cpu(x, alpha: float):
    return x * alpha

x = torch.ones(3)
print(torch.ops.myops.scale(x, 2.0))
print(torch._C._dispatch_dump_table("myops::scale"))
```

C++ 侧等价注册:

```cpp
#include <torch/library.h>

Tensor scale_cpu(const Tensor& x, double alpha) {
    return x * alpha;
}

TORCH_LIBRARY(myops, m) {
    m.def("scale(Tensor x, float alpha) -> Tensor");
    m.impl("scale", torch::kCPU, &scale_cpu);
}
```

## 1.5 实战例子

### 1.5.1 用 TensorIterator + CUDA kernel 实现高性能 Elementwise Op

实现一个 elementwise `smooth_l1_loss` 的 CUDA kernel：

```python
# Python 注册 + C++ CUDA kernel 的完整方案
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>

void smooth_l1_cuda_kernel(torch::Tensor input, torch::Tensor target,
                            torch::Tensor output, double beta) {
    auto iter = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(input)
        .add_input(target)
        .build();

    gpu_kernel(iter, [beta] GPU_LAMBDA(float x, float y) -> float {
        float diff = fabsf(x - y);
        if (diff < beta)
            return 0.5 * diff * diff / beta;
        else
            return diff - 0.5 * beta;
    });
}

TORCH_LIBRARY(myops, m) {
    m.def("smooth_l1(Tensor input, Tensor target, float beta) -> Tensor");
    m.impl("smooth_l1", torch::kCUDA, &smooth_l1_cuda_kernel);
}
"""

module = load_inline(
    name="smooth_l1_ext",
    cpp_sources=cuda_source,
    functions=["smooth_l1_cuda_kernel"],
    with_cuda=True,
    verbose=False,
)

# 使用
input = torch.randn(1000, device="cuda")
target = torch.randn(1000, device="cuda")
output = torch.ops.myops.smooth_l1(input, target, 1.0)
print("Output device:", output.device, "shape:", output.shape)
```

TensorIterator 自动处理了广播、类型提升、内存布局，无需手动编写 CUDA launch 参数。

### 1.5.2 自定义 op 完整支持 torch.compile 的配置

确保自定义 op 在 `torch.compile` 下正常工作需要注册 meta kernel 和 autograd：

```python
import torch

lib = torch.library.Library("myops", "DEF")

# 1. Schema
lib.define("my_silu(Tensor x) -> Tensor")

# 2. CPU kernel
@torch.library.impl("myops::my_silu", "CPU")
def my_silu_cpu(x):
    return x * torch.sigmoid(x)

# 3. CUDA kernel
@torch.library.impl("myops::my_silu", "CUDA")
def my_silu_cuda(x):
    return x * torch.sigmoid(x)

# 4. Meta kernel - 必须! 否则 compile 失败
@torch.library.impl("myops::my_silu", "Meta")
def my_silu_meta(x):
    return x.new_empty(x.shape)

# 5. Autograd
@torch.library.impl("myops::my_silu", "Autograd")
def my_silu_autograd(x):
    class SiluFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.ops.myops.my_silu(x)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            sig_x = torch.sigmoid(x)
            return grad_output * sig_x * (1 + x * (1 - sig_x))

    return SiluFn.apply(x)

# 验证 compile
x = torch.randn(4, 8, requires_grad=True)

# eager
out = torch.ops.myops.my_silu(x)
out.sum().backward()

# compile
compiled_fn = torch.compile(lambda x: torch.ops.myops.my_silu(x).sum())
out_compiled = compiled_fn(x)
out_compiled.backward()
print("compile 成功!")
```

### 1.5.3 用 cpp_extension 编译并注册完整 CUDA Extension

一个完整的 C++/CUDA extension 项目结构：

```
my_extension/
  ├── setup.py            # 编译脚本
  ├── my_extension.cpp    # TORCH_LIBRARY 注册
  └── my_extension_kernel.cu  # CUDA kernel
```

```python
# setup.py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

setup(
    name="my_extension",
    ext_modules=[
        CUDAExtension(
            "my_extension",
            ["my_extension.cpp", "my_extension_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```cpp
// my_extension.cpp
#include <torch/extension.h>

void my_add_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor out);

void my_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out);

TORCH_LIBRARY(my_ext, m) {
    m.def("my_add(Tensor a, Tensor b) -> Tensor");
    m.impl("my_add", torch::kCPU, &my_add_cpu);
    m.impl("my_add", torch::kCUDA, &my_add_cuda);
}
```

安装和使用：

```bash
cd my_extension
python setup.py install

# 使用
python -c "
import torch
import my_extension  # 自动注册 my_ext::my_add
x = torch.randn(3, device='cuda')
y = torch.randn(3, device='cuda')
z = torch.ops.my_ext.my_add(x, y)
print(z)
"
```

## 2. 从 Python API 到源码的调用链

```
Python:
  torch.library.Library("myops", "DEF")
    -> torch/library.py: Library.__init__()
    -> _C._define_op("myops::scale", schema)

  @torch.library.impl("myops::scale", "CPU")
    -> torch/library.py: impl()
    -> _C._impl_op("myops::scale", "CPU", kernel_fn, ...)

C++:
  TORCH_LIBRARY(myops, m)
    -> 展开为 static 构造函数
    -> 调用 Dispatcher::registerLibrary("myops", fn)
    -> m.def() / m.impl() 注册 schema 和 kernel
    -> 最终写入 Dispatcher 的 op 分发表

调用:
  torch.ops.myops.scale(x, 2.0)
    -> torch/_ops.py: OpOverload.__call__
    -> Dispatcher::call<...>(op_name, ...)
    -> 查找 DispatchKey, 调用对应的 kernel
```

## 3. 核心源码文件

```
torch/library.py                                   # Python 侧注册入口
torch/_ops.py                                      # torch.ops 访问
torch/utils/cpp_extension.py                       # C++ extension 编译工具
aten/src/ATen/core/dispatch/Dispatcher.h           # Dispatcher 注册/查找
aten/src/ATen/core/dispatch/OperatorEntry.h        # 算子分发表项
aten/src/ATen/native/TensorIterator.cpp            # TensorIterator (elementwise)
aten/src/ATen/native/cuda/Loops.cuh                # CUDA elementwise 循环模板
c10/core/DispatchKey.h                             # DispatchKey 枚举
```

## 4. 关键机制源码解读

### 4.1 Python `torch.library` 与 C++ `TORCH_LIBRARY` 的对应

| 操作 | Python | C++ |
|------|--------|-----|
| 定义 schema | `lib.define(schema)` | `m.def(name, schema)` |
| 注册 CPU kernel | `lib.impl(name, "CPU", fn)` | `m.impl(name, kCPU, fn)` |
| 注册 CUDA kernel | `lib.impl(name, "CUDA", fn)` | `m.impl(name, kCUDA, fn)` |
| 注册 Meta kernel | `lib.impl(name, "Meta", fn)` | `m.impl(name, kMeta, fn)` |
| 注册 Autograd | `lib.impl(name, "Autograd", fn)` | `m.impl(name, kAutograd, fn)` |

两者最终都调用 `Dispatcher::registerOp` / `Dispatcher::registerKernel`。

### 4.2 自定义 op 需要注册什么

完整注册示例：

```python
lib = torch.library.Library("myops", "DEF")

# 1. Schema
lib.define("myop(Tensor x, float alpha) -> Tensor")

# 2. CPU kernel
@torch.library.impl("myops::myop", "CPU")
def myop_cpu(x, alpha):
    return x * alpha

# 3. CUDA kernel
@torch.library.impl("myops::myop", "CUDA")
def myop_cuda(x, alpha):
    return x * alpha

# 4. Meta kernel（为 torch.compile 支持）
@torch.library.impl("myops::myop", "Meta")
def myop_meta(x, alpha):
    return x.new_empty(x.shape)

# 5. Autograd 公式
@torch.library.impl("myops::myop", "Autograd")
def myop_autograd(x, alpha):
    class MyOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            return torch.ops.myops.myop(x, alpha)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return grad_output * ctx.alpha, None

    return MyOp.apply(x, alpha)
```

**Meta kernel 对 `torch.compile` 的重要性**：没有 meta kernel，Dynamo/AOTAutograd 无法推断输出 shape，编译失败。

### 4.3 自定义 op 也进入同一个 Dispatcher

```cpp
// aten/src/ATen/core/dispatch/Dispatcher.h
class Dispatcher {
    // 全局单例
    static Dispatcher& singleton();

    // 所有 op (包括自定义) 共享同一个注册表
    ska::flat_hash_map<OperatorName, OperatorHandle> op_registry_;

    template <class... Args>
    std::decay_t<decltype(std::declval<OperatorHandle>().call<Args...>(...))>
    call(const OperatorHandle& op, DispatchKeySet keys, Args... args) {
        // 查找 dispatchTable_[keys.highestPriority()]
        // 调用 kernel
    }
};
```

自定义 op 注册后，其 dispatch table 和内置 op 完全一样，`torch._C._dispatch_dump_table("myops::scale")` 可以查看。

### 4.4 TensorIterator 对 Elementwise Kernel 的价值

```cpp
// aten/src/ATen/native/TensorIterator.cpp
auto iter = TensorIteratorConfig()
    .add_output(result)
    .add_input(x)
    .build();

// 自动处理:
// - 广播 (broadcasting)
// - 类型提升 (type promotion)
// - 内存布局优化 (contiguous 等)
// - 64-bit 索引处理

// CUDA elementwise kernel:
// aten/src/ATen/native/cuda/Loops.cuh
gpu_kernel(iter, []GPU_LAMBDA(float a, float b) -> float {
    return a * b + a;
});
```

TensorIterator 抽象了 elementwise op 的共性问题，用户只需提供 lambda，迭代器处理并行、边界、类型转换。

## 5. 和已有笔记的连接

```
dispatcher/         — 自定义 op 进入同一个 Dispatcher 调度
triton_kernel/      — Triton 编写 kernel 与 TORCH_LIBRARY 注册的关系
autograd/           — 自定义 op 需要显式注册 Autograd 公式
torch.compile/      — 自定义 op 提供 meta kernel 才能在 compile 下工作
torchgen/           — 内置 op 由 torchgen 生成，自定义 op 手动注册
meta_fake_tensor/   — Meta kernel 是 custom op 支持 compile 的关键
```

## 6. 常见坑点

- **自定义 op 的 schema 字符串必须符合 TorchSchema 语法**，与 native_functions.yaml 一致。
- **没有 meta kernel 的 custom op 在 `torch.compile` 下会报错 "no meta kernel registered"**。
- **注册 Autograd 时，forward 必须调用 `torch.ops.myops.myop` 而不是内部实现**，否则 autograd graph 中缺失该节点。
- **C++ extension 编译需要 torch 的 include/lib 路径**，`cpp_extension.py` 提供了 CMake-like 封装。
- **自定义 op 名字不能和已有 op 冲突**（包括 `aten::` 命名空间）。
- **TensorIterator 默认使用 float 类型提升**，自定义类型提升策略需额外配置。

## 7. 阅读源码时建议搜索的关键词

```bash
# Python torch.library 注册逻辑
rg -n "class Library" torch/library.py

# Dispatcher 注册 op
rg -n "registerOp|registerKernel" aten/src/ATen/core/dispatch/Dispatcher.h

# TensorIterator 构建
rg -n "class TensorIterator" aten/src/ATen/native/TensorIterator.cpp

# CUDA gpu_kernel 模板
rg -n "gpu_kernel\b" aten/src/ATen/native/cuda/Loops.cuh

# cpp_extension 编译
rg -n "CppExtension|CUDAExtension" torch/utils/cpp_extension.py

# 查看已注册的自定义 op
rg -n "TORCH_LIBRARY\(" aten/src/ATen/native/
```
