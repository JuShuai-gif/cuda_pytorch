### 阅读 PyTorch 源码指南

随着我越来越多地使用 torchtitan，有时调用的内核并非我预期的那样。我意识到我需要学习性能分析，并至少对如何端到端追踪一个运算有中等程度的理解。（这也是你是否能深入理解的一个重要信号）

PyTorch 是一个庞大的库，所以直接克隆然后逐行阅读代码可能行不通。这是我对 PyTorch 架构的剖析指南，以及如何入手阅读它。

PyTorch 主要有四层架构：
- **Python 前端** - 这是面向 Python 的 API，你在构建心爱的 LLM 时需要写的 `nn.Module`、`torch.tensor` 等都在这里。
- **调度器（Dispatcher）** - 当你调用一次 `matmul` 时，它需要决定使用 CPU、CUDA 还是 MPS 来计算。调度器帮你做这个决定。
- **C++ 后端（ATen / c10）** - 我认为所有数学运算和内存管理都发生在这里。
- **编译器栈** - 这是 2.0 版本引入的新特性。Torch Dynamo 捕获计算图，而 Inductor 优化并生成代码。

请始终牢记这四大层。

**1. Python 层**

从 `torch/nn/modules/module.py` 开始——所有模型的基类以及钩子都定义在这里。
你也可以选择一个感兴趣的运算，追踪其 `__call__` 方法。
关于 C++ 绑定侧，`torch/csrc/` 目录中包含了将 Python 对象转换为 C++ 指针的 pybind11 代码。

**2. 调度器**

`torch.matmul(a, b)` 并不会直接跳转到手写的内核。

有一个名为 `native_functions.yaml` 的映射表，位于 `aten/src/ATen/native/native_functions.yaml`。它列出了每个运算符、其调度键以及实现它的 C++ 函数。

例如，`grouped_mm` 调度到 `_scaled_grouped_mm_cuda`：

```
- func: _scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
  variants: function
  dispatch:
    CUDA: _scaled_grouped_mm_cuda
  tags: needs_exact_strides
```

但如果你想看到真正的 C++ 代码，需要编译一个 PyTorch 的 debug 版本。

**3. ATen**

所有数学运算符和函数都需要在这里定义。源码在 `aten/src/ATen/native/`。

该文件夹中有一个很好的 README.md 指南教你如何添加新运算。

**4. 编译器栈**

`torch.compile` 用好了几乎是免费的性能提升。

Torch Dynamo 读取 Python 字节码并捕获计算图。Inductor 将计算图编译成 Triton 内核。

要调试计算图，使用以下命令：

```
TORCH_LOGS="+dynamo,+inductor" python your_model.py
```

代码在 `torch/_dynamo/` 和 `torch/_inductor/` 下。全部是 Python 代码但密度极高。我还没摸清从哪里开始读这部分最好。

**更多资源**

- PyTorch advanced 章节相当不错
- Edward Yang 主持的 PyTorch 开发者播客。我多希望他们能继续更新，但似乎已经停更了
- ezyang 关于 PyTorch 内部机制的博客

你还需要编译一个 PyTorch 的 debug 版本，并保留在编译过程中生成的源码，否则很难在函数调用栈中找到某些函数的源代码。

你可以尝试：

1. 选择 main 分支，然后：

```
export DEBUG=1
python setup.py bdist_wheel
uv pip install dist/torch*.whl
```

2. 启动你要调试的 torch 脚本，用 gdb 启动，添加断点，观察整个函数调用栈。

最好的做法是只追踪一个足够复杂的运算，端到端地走一遍，而不是试图通读整个代码库然后自毁。随着我的探索深入，我会补充更多细节。

祝阅读愉快！

**延伸阅读**
1. FX 图论文 - https://arxiv.org/pdf/2112.08429
2. FX 图文档 - https://docs.pytorch.org/docs/2.12/fx.html
3. torch compile 手册 - https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.ivdr7fmrbeab
4. https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
5. https://docs.pytorch.org/functorch/nightly/notebooks/aot_autograd_optimizations.html
6. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html#dynamo-deep-dive
7. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html
8. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_faq.html
9. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_faq.html
