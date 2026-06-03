# 模块 10：从 IR 生成 Kernel 代码

## 工业背景

从 IR 生成代码是 **Triton**、**XLA**、**TVM** 和 **torch.inductor** 等 ML 编译器的最后一步。在图优化（模块 09）之后，编译器将优化后的图 lowering 为实际的 GPU kernel 源代码。本模块实现了一个简化但真实的代码生成器，接收我们的 IR 并生成 Triton kernel 代码字符串，然后编译并运行它们。

### torch.inductor 的 Triton Codegen 工作原理

torch.inductor 在多个阶段运行：

1. **Capture**：torch.compile 将 PyTorch 程序捕获为 FX 图
2. **Graph Optimization**：FX pass（CSE、constant folding、fusion）优化图
3. **Lowering**：每个 FX 节点被 lowering 为 inductor IR 节点
4. **Scheduling**：调度器将兼容操作分组为融合组
5. **Codegen**：每个融合组被翻译为 `@triton.jit` kernel 源代码字符串
6. **JIT Compilation**：Triton 编译为 PTX，然后 CUDA driver 编译为 SASS
7. **Caching**：已编译的 kernel 按源代码 + GPU 架构的哈希值进行缓存

### XLA HLO Lowering

XLA（由 JAX、TF 使用）遵循类似的路径：
- HLO 图 -> 优化 -> codegen -> LLVM/NVPTX 或 Triton（实验性）
- codegen 生成 LLVM IR（用于 CPU）或优化的 GPU kernel
- XLA 的融合更加激进，融合逐元素 + 归约操作

### 本模块的 Codegen 管线

```
IR Graph（来自模块 09）
    │
    ▼
TritonCodeGenerator.generate_elementwise_fusion()
    │  - 识别输入 tensor 及其角色
    │  - 按拓扑顺序排序操作
    │  - 生成包含所有输入/输出的 kernel 签名
    │  - 生成计算体（加载、算术、存储）
    │  - 为边缘元素发出适当的 mask
    ▼
Triton Kernel 源代码字符串（@triton.jit 装饰）
    │
    ▼
TritonCodeGenerator.compile_and_run()
    │  - 在 triton 命名空间中 exec() 源代码
    │  - 找到已编译的 @triton.jit 函数
    │  - 构建输出 tensor
    │  - 以正确的 grid 和参数调用 kernel
    ▼
GPU 输出 Tensor
```

## 支持的操作

**逐元素（可融合）：**
- 二元：add、sub、mul、div
- 一元激活：relu、gelu（tanh 近似）、silu、sigmoid、tanh、exp、log

**归约：**
- Softmax（在线稳定算法）
- LayerNorm（- mean / std）
- RMSNorm（- rms）

## 常见陷阱

### Broadcasting 处理
当 bias 是 1D 而 data 是 2D 时，生成的 kernel 必须正确地步进加载 bias 或扩展它。我们的 codegen 使用 `tl.load(bias_ptr + offsets)`，这要求两个 tensor 在内存中具有相同的形状。

### Stride 计算
对于非连续 tensor，kernel 必须使用正确的 strides。我们的 codegen 为简洁起见假设输入是连续的；在生产环境中，torch.inductor 从 FX 图元数据中计算 strides。

### 类型提升
混合精度操作需要显式类型转换。我们的 codegen 在内部使用 float32 计算，并以输入 dtype 返回输出。

### 编译缓存失效
Triton 按哈希值缓存已编译的 kernel。如果你更改了生成的源代码，Triton 将重新编译。缓存位于 `~/.triton/cache/`。对于调试，你可以设置 `TRITON_CACHE_DIR` 或使用会禁用缓存的 `triton.testing.do_bench`。

### BLOCK_SIZE 选择
block size 必须是 2 的幂且对 tensor 大小合理。太小 = 多个 block，低 occupancy。太大 = 更少的 block，可能无法饱和 GPU。常用值：128、256、512、1024、2048。

## 文件

| 文件 | 用途 |
|------|---------|
| `triton_codegen.py` | `TritonCodeGenerator` 类 - 从 IR 生成并运行 Triton kernel |
| `codegen_demo.py` | 完整 codegen 管线的演示 |
| `test_kernel_codegen.py` | 正确性的 pytest 测试 |
| `benchmark_kernel_codegen.py` | vs PyTorch eager 和 torch.compile 的性能基准测试 |

## 运行

```bash
# 演示
python 10_kernel_codegen/codegen_demo.py

# 测试
pytest 10_kernel_codegen/test_kernel_codegen.py -v

# 基准测试
python 10_kernel_codegen/benchmark_kernel_codegen.py
```
