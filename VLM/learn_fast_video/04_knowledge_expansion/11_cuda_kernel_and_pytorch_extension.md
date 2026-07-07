# CUDA Kernel 与 PyTorch Extension

> 知识点扩展：PyTorch 自定义算子、C++/CUDA extension、kernel 注册、tensor layout、shared memory/warp/block，回扣 fastvideo-kernel。

## 1. 为什么写自定义 kernel

PyTorch 内置算子不够——视频稀疏 attention、INT8 量化 GEMM 等需要专用 kernel 才能达到硬件峰值性能。FastVideo 的 `fastvideo-kernel` 就是这些 kernel 的集合。

## 2. PyTorch Extension 的两种注册方式

### pybind11（fastvideo-kernel 用这个）
```cpp
// csrc/common_extension.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_cuda", &quant);
    m.def("block_sparse_fwd", torch::wrap_pybind_function(block_sparse_attention_forward));
}
```
Python 侧：`from fastvideo_kernel._C import fastvideo_kernel_ops`。

### TORCH_LIBRARY（另一种，dispatcher 集成）
FastVideo 主要用 pybind11 + Python 侧 `torch.library.custom_op` 做 torch.compile 兼容。

## 3. torch.library.custom_op（torch.compile 兼容）

原生 kernel 不可被 dynamo 追踪，会 graph break。用 custom_op 包装成可追踪算子：
```python
@torch.library.custom_op("fastvideo_kernel::block_sparse_attn_triton", mutates_args=(), device_types="cuda")
def block_sparse_attn_triton(q, k, v, ...): ...
```

## 4. CUDA 编程基础概念

| 概念 | 含义 |
|------|------|
| **thread** | 最小执行单元 |
| **warp** | 32 个 thread，SIMT 执行 |
| **block** | 一组 thread（共享 shared memory） |
| **grid** | 一组 block |
| **shared memory** | block 内共享的快速片上内存 |
| **register** | thread 私有寄存器 |

kernel 启动：`kernel<<<grid, block, shared_mem, stream>>>(args)`。

### 4.1 GPU 内存层级（理解性能的关键）

```
寄存器（register）   —— 最快，thread 私有，KB 级
共享内存（SMEM）     —— 快，block 内共享，~百 KB
L2 cache             —— 中等
全局显存（HBM）      —— 慢（但带宽高），GB 级，所有 thread 可见
```
核心优化原则：**减少 HBM 访问，多用 SMEM/寄存器**。FlashAttention 快就是因为把 attention 计算留在 SMEM，不把大矩阵写回 HBM。

### 4.2 性能瓶颈类型

| 瓶颈 | 特征 | 优化方向 |
|------|------|---------|
| **compute-bound** | 算力打满 | 用 tensor core、降精度（FP16/INT8） |
| **memory-bound** | 带宽打满、算力闲 | 减少 HBM 访问、融合 kernel、tiling |
| **latency-bound** | 都没打满 | 增加并行度、隐藏延迟 |

attention、GEMM 常是 compute 或 memory bound；norm、逐元素算子常是 memory-bound（所以 FastVideo 把 RMSNorm/量化写成融合 kernel）。用 ncu 判断属于哪类（见 [`../06_practical_guides/06_how_to_profile_performance.md`](../06_practical_guides/06_how_to_profile_performance.md)）。

### 4.3 Hopper/Blackwell 新特性（FastVideo kernel 用到）

- **TMA（Tensor Memory Accelerator）**：Hopper 的异步批量 DMA，一条指令搬一整块 tile，重叠计算与访存。
- **wgmma（warpgroup MMA）**：4 个 warp 协作的大矩阵乘指令，比老的 mma 吞吐高。
- **warp specialization**：不同 warp 分工（producer 搬数据、consumer 算），流水线化。
- **FP8/FP4 tensor core**：Blackwell 原生低精度矩阵乘。

ThunderKittens（`csrc/attention/*.cu`）和 CuTe DSL（`attn_qat_infer/`）就是封装这些特性的 DSL。

## 5. Tensor Layout 与 contiguous

- CUDA kernel 通常要求 tensor 内存连续（`.contiguous()`）。
- attention 后端常需转置布局（NHD ↔ BHSD），如 `sage_attn3.py` 的 `permute(0,2,1,3).contiguous()`。
- layout 不对会导致 kernel 读错数据或性能差。

## 6. FastVideo kernel 的高级技术

### ThunderKittens（Hopper）
`csrc/attention/*.cu` 用 TK DSL：
- **TMA**（Tensor Memory Accelerator）：异步批量加载。
- **wgmma**：warpgroup 矩阵乘。
- **warp specialization**：producer warp（加载）+ consumer warp（计算）分工。
- **online softmax**：不物化全矩阵。

### CuTe DSL（Blackwell）
`attn_qat_infer/blackwell/` 用 CUTLASS CuTe：FP4 量化 attention。

## 7. TurboDiffusion kernel（最易读）

```
csrc/turbodiffusion/norm/rmsnorm.cu
```
RMSNorm：`y = w·x/√(mean(x²)+eps)`。用 warp shuffle reduction 求 `mean(x²)`，per-CTA 处理一行。这是入门读 CUDA kernel 的好起点。

## 8. Python → CUDA 完整链

```
Python 高层函数（ops.py）
  → torch.library.custom_op 包装
  → pybind 函数（common_extension.cpp）
  → C++ host 函数（block_sparse_h100.cu）
  → CUDA kernel <<<grid>>>
  → 返回 torch.Tensor
```

## 9. 编译

scikit-build-core + CMake（`build.sh`）。自动检测 GPU 架构，条件编译 TK/FP4。见 [`../02_source_by_directory/11_fastvideo_kernel.md`](../02_source_by_directory/11_fastvideo_kernel.md)。

## 10. Triton fallback

非 Hopper GPU 用 Triton kernel（`fastvideo-kernel/python/fastvideo_kernel/triton_kernels/`）。Triton 是 Python 内嵌的 GPU kernel DSL，比 CUDA 易写，跨架构。

## 11. 回扣源码
| 概念 | 源码 |
|------|------|
| pybind 注册 | `csrc/common_extension.cpp` |
| custom_op | `python/fastvideo_kernel/block_sparse_attn.py` |
| 简单 kernel | `csrc/turbodiffusion/norm/rmsnorm.cu` |
| 复杂 kernel | `csrc/attention/block_sparse_h100.cu` |
| Triton fallback | `triton_kernels/` |

## 12. 延伸
- kernel 架构：[`../01_architecture/04_kernel_architecture.md`](../01_architecture/04_kernel_architecture.md)
- 如何读 kernel：[`../06_practical_guides/07_how_to_read_cuda_kernel.md`](../06_practical_guides/07_how_to_read_cuda_kernel.md)
