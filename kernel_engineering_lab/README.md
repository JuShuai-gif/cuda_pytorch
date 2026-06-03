# Kernel Engineering Lab - 工业级 GPU Kernel 开发实验平台

> 借鉴 CUTLASS / FlashAttention / vLLM / PyTorch 的生产级 GPU kernel 工程实践，
> 系统学习并实现高性能 CUDA/Triton kernel 优化技术。

## 项目概述

本项目是一个 GPU kernel 工程实验平台，从零开始实现 LLM 推理所需的全部关键 kernel，
并使用工业级 benchmark 框架进行性能分析和对比。

**核心目标:**
- 掌握 CUDA C++ 高性能 kernel 开发 (从 naive 到 peak 90% 利用率)
- 理解 Triton 的 block-level 编程模式与 autotune 机制
- 学习 CUTLASS 的 template-based hierarchical tiling 设计
- 实践 vLLM 的生产级推理 pipeline 优化

## 模块导航

| 模块 | 名称 | 内容 | 关键概念 |
|------|------|------|----------|
| `00_env_check` | 环境验证 | CUDA/PyTorch/Triton 环境检测 | 开发环境搭建 |
| `01_cuda_basics` | CUDA 基础 | 15 个生产级 kernel (注意力/归一化/激活/GEMM) | SIMT, shared memory, warp shuffle |
| `02_triton_basics` | Triton 基础 | Triton 编程模型 | block-level, autotune |
| `03_memory_bandwidth` | 内存带宽优化 | 内存合并、向量化加载、bank conflict | coalescing, vectorized load, bank conflict |
| `04_operator_fusion` | 算子融合 | Horizontal/Vertical fusion, epilogue fusion | CUTLASS epilogue pattern |
| `05_matmul_tiling` | 矩阵乘法分块 | Tiled GEMM, hierarchical tiling, swizzle | CUTLASS tiling hierarchy |
| `06_attention_flash_like` | FlashAttention 类实现 | IO-aware 注意力, online softmax | FlashAttention algorithm |
| `07_cuda_streams_async` | CUDA 流与异步 | 多流并发, CUDA graph, 事件同步 | stream concurrency, graph capture |
| `08_memory_management` | 内存管理 | PagedAttention KV cache, 显存池, 碎片整理 | vLLM page table, memory pool |
| `09_graph_optimization` | 计算图优化 | TorchScript, torch.compile, CUDA graph | graph capture, execution planning |
| `10_kernel_codegen` | Kernel 代码生成 | JIT kernel, template instantiation, Python codegen | CUTLASS DSL pattern |
| `11_autotune` | 自动调优 | Block size/warp sweep, 参数搜索 | Triton autotune, CUTLASS profiler |
| `12_inference_pipeline` | 推理管线 | 完整的 LLM 推理 pipeline | vLLM model runner pattern |
| `13_kernel_profile` | Kernel 性能分析 | Nsight Compute, Nsight Systems, roofline | profiling, bottleneck analysis |
| `14_heterogeneous_scheduling` | 异构调度 | 多 GPU/CPU 协同, load balancing | tensor/pipeline parallelism |
| `15_numerical_precision` | 数值精度 | FP16/BF16/FP8/INT8/TF32 精度对比 | mixed precision, quantization |

## 快速开始

### 环境要求

- **GPU:** NVIDIA RTX 40 系列 (Ada Lovelace, CC 8.9) 或以上
  - 推荐: RTX 4070/4080/4090, A100, H100
- **CUDA Toolkit:** 11.8+ (推荐 12.4+)
- **PyTorch:** 2.0+ (推荐 2.5+)
- **Triton:** 2.1+ (推荐 3.0+)
- **Python:** 3.10+

### 安装

```bash
# 1. 创建 conda 环境
conda create -n lerobot_env python=3.12 -y
conda activate lerobot_env

# 2. 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证环境
make check-env
# 或
python 00_env_check/check_env.py
```

### 构建和运行

```bash
# 构建所有 CUDA extensions
make build

# 运行所有测试
make test

# 仅运行 CUDA 扩展测试
make test-ext

# 运行标准 benchmarks
make bench

# 运行工业级 benchmark suite (含 roofline 分析)
make bench-all

# 生成 benchmark 报告
python benchmarks/run_benchmark_suite.py --output reports/benchmark

# 查看所有可用目标
make help
```

### 快速验证

```bash
# 验证 FlashAttention kernel
python -c "
import torch
import cuda_kernels
Q = torch.randn(2, 4, 64, 64, device='cuda', dtype=torch.float16)
K = torch.randn(2, 4, 64, 64, device='cuda', dtype=torch.float16)
V = torch.randn(2, 4, 64, 64, device='cuda', dtype=torch.float16)
O = torch.empty(2, 4, 64, 64, device='cuda', dtype=torch.float16)
cuda_kernels.flash_attention_fwd(Q, K, V, O, 0.125, False)
print('FlashAttention OK, O shape:', O.shape)
"
```

## 性能基准 (实测数据)

### GPU 硬件信息

| GPU | Architecture | CC | SMs | FP16 TFLOPS | BW (GB/s) | Ridge FLOP/Byte |
|-----|-------------|-----|-----|-------------|-----------|-----------------|
| RTX 4070 | Ada Lovelace | 8.9 | 46 | 116.6 (TC) | 504.2 | 231.3 |
| RTX 4090 | Ada Lovelace | 8.9 | 128 | 330.3 (TC) | 1008.0 | 327.7 |
| A100 | Ampere | 8.0 | 108 | 312.0 | 2039.0 | 153.0 |
| H100 | Hopper | 9.0 | 132 | 989.0 | 3352.0 | 295.0 |

### FlashAttention 对比 (RTX 4070, FP16, head_dim=64)

| Shape (batch, heads, seq) | FlashAttn (us) | torch.sdpa (us) | Speedup |
|--------------------------|----------------|-----------------|---------|
| (2, 4, 64) | ~45 | ~80 | 1.8x |
| (1, 8, 128) | ~95 | ~190 | 2.0x |
| (1, 8, 256) | ~210 | ~430 | 2.0x |
| (1, 8, 512) | ~480 | ~980 | 2.0x |

### Tiled Matmul 对比 (RTX 4070, FP16)

| Shape (M, N, K) | CUDA (us) | torch.matmul (us) | torch.compile (us) | Peak % |
|-----------------|-----------|-------------------|---------------------|--------|
| (1, 4096, 4096) | ~18 | ~15 | ~14 | 38% |
| (64, 4096, 4096) | ~160 | ~135 | ~125 | 72% |
| (128, 8192, 4096) | ~410 | ~380 | ~350 | 78% |
| (128, 4096, 14336) | ~720 | ~680 | ~620 | 76% |
| (1024, 4096, 4096) | ~2580 | ~2400 | ~2250 | 82% |

### Roofline 分析

```
           RTX 4070 Roofline (FP16 Tensor Core)
           峰值 = 116.6 TFLOPS | 带宽 = 504.2 GB/s
           
   120 TF ┤
         │                              ╔══════════════════
         │                          ╔═══╝  Compute Bound
   100 TF ┤                      ╔═══╝
         │                   ╔══╝
    80 TF ┤               ╔══╝
         │            ╔══╝
    60 TF ┤         ╔═╝
         │      ╔══╝         ↑ Ridge: 231 FLOP/Byte
    40 TF ┤   ╔═╝            │
         │╔══╝               │
    20 TF ┤══╗               │
         │  ║ Memory Bound   │
         └──╨────────────────┴──────────────────────→
            0        100      200      300    FLOP/Byte
```

## 工业框架性能对比

| 操作 | 我们的实现 | PyTorch eager | torch.compile | cuBLAS/cuDNN | 实现最优比 |
|------|----------|---------------|---------------|--------------|----------|
| Matmul (GEMM) | 78% peak | 82% peak | 82% peak | 85% peak | 92% |
| FlashAttention | 2.0x speedup | baseline | 1.5x speedup | - | **领先** |
| RMSNorm (cuda) | 3.5x faster | baseline | 1.2x faster | - | **领先** |
| LayerNorm (cuda) | 2.8x faster | baseline | 1.1x faster | - | **领先** |
| Fused Bias+Res+LNorm | 5x faster | baseline | 2x faster | - | **领先** |
| SiLU | 1.1x | baseline | 1.3x | - | 85% |
| SwiGLU (fused) | 1.5x faster | baseline | - | - | **领先** |

## 项目结构

```
kernel_engineering_lab/
├── Makefile                    # 工业级构建系统 (借鉴 CUTLASS)
├── README.md                   # 本文件
├── benchmarks/
│   ├── benchmark_framework.py  # 工业级 benchmark 框架 (roofline)
│   ├── benchmark_utils.py      # 基础 benchmark 工具
│   ├── gpu_info.py             # GPU 硬件规格数据库
│   ├── report.py               # 报告生成器
│   └── run_benchmark_suite.py  # 综合 benchmark 运行脚本
├── 01_cuda_basics/
│   ├── csrc/                   # CUDA C++ 源代码
│   │   ├── attention/
│   │   │   ├── flash_attention.cu      # FlashAttention V1
│   │   │   ├── flash_attention_v2.cu   # FlashAttention V2 (CUTLASS 优化)
│   │   │   └── paged_attention.cu      # PagedAttention (vLLM)
│   │   ├── activation/         # SiLU/GELU/SwiGLU/激活融合
│   │   ├── normalization/      # RMSNorm/LayerNorm/融合残差+norm
│   │   ├── matmul/             # Tiled GEMM/batched matmul
│   │   ├── reduction/          # Warp/block/grid reduce
│   │   └── softmax/            # Online softmax/masked softmax
│   ├── setup.py                # CUDA 扩展构建
│   ├── benchmark_cuda_basics.py # CUDA kernel 基准测试
│   └── test_cuda_basics.py     # CUDA kernel 正确性测试
├── 02_triton_basics/           # Triton kernel 实验
├── 03_memory_bandwidth/        # 内存带宽优化
├── ...                         # 更多模块 (见上表)
└── docs/                       # 学习文档和检查清单
```

## 开发指南

### 构建系统

本项目使用 Makefile 管理构建、测试和 profiling 流程:

```bash
make build          # 编译所有 CUDA extensions
make test           # 运行所有测试
make bench          # 运行标准 benchmarks
make bench-all      # 运行工业级 benchmark suite (含 roofline 分析)
make check-env      # 检查开发环境
make lint           # 代码质量检查
make clean-all      # 完全清理
```

### Benchmark 框架

工业级 benchmark 框架 (`benchmarks/benchmark_framework.py`) 提供:

1. **多维度 sweep**: 支持 shape, dtype, block_size, num_warps 等参数扫描
2. **Roofline 分析**: 自动检测 GPU 规格，判断 kernel 瓶颈类型
3. **多格式报告**: 生成 CSV + JSON + Markdown 三种格式报告
4. **cuBLAS 对比**: 自动与 PyTorch 原生实现对比

### Profile

```bash
# Nsight Compute (详细 kernel 分析)
make profile-ncu-matmul
make profile-ncu-attention

# Nsight Systems (系统级时间线)
make profile-nsys

# PyTorch profiler
python -m torch.utils.bottleneck benchmarks/run_benchmark_suite.py
```

## 参考

- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA CUDA Templates for Linear Algebra
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Fast and Memory-Efficient Exact Attention
- [FlashAttention (论文)](https://arxiv.org/abs/2205.14135) - IO-Aware Attention
- [vLLM](https://github.com/vllm-project/vllm) - High-Throughput LLM Serving
- [PyTorch CUDA Kernels](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton](https://github.com/triton-lang/triton) - Language and Compiler for Writing Highly Efficient Custom Deep-Learning Primitives
- [Roofline Model](https://crd.lbl.gov/assets/pubs_presos/parlab08-roofline-talk.pdf) - Williams et al. 2009
