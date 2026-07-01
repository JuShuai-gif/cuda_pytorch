# CUDA 架构：GPU SM、Warp、Memory Hierarchy

> 关联: GPU SM 调度、occupancy、shared memory、register pressure、tensor core

## 0. 一句话总览

理解 GPU 的 SM (Streaming Multiprocessor) 架构、warp 调度、寄存器文件大小和 shared memory 容量，是优化自定义 CUDA kernel 和诊断 Inductor/Triton 生成的 kernel 性能的底层基础。

## 1. 最小例子

```python
import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SMs: {props.multi_processor_count}")
    print(f"Max threads per SM: {props.max_threads_per_multi_processor}")
    print(f"Warp size: 32")
    print(f"Shared memory / SM: {props.max_shared_memory_per_block_optin / 1024:.0f} KB")
    print(f"Registers / SM: {props.regs_per_multiprocessor // 1024}K")
```

## 2. 实战例子

### 2.1 Occupancy 计算

```python
# Occupancy = active_warps / max_warps_per_SM
# 受限于: registers, shared memory, block size

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    warp_size = 32

    # 示例: Triton kernel (block=128 threads, 64 regs/thread, 8KB shared)
    threads_per_block = 128
    regs_per_thread = 64
    shared_per_block = 8 * 1024

    max_blocks_regs = (props.regs_per_multiprocessor // (threads_per_block * regs_per_thread))
    max_blocks_shared = props.max_shared_memory_per_block_optin // shared_per_block
    max_blocks_threads = props.max_threads_per_multi_processor // threads_per_block

    active_blocks = min(max_blocks_regs, max_blocks_shared, max_blocks_threads)
    active_warps = active_blocks * (threads_per_block // warp_size)
    max_warps = props.max_threads_per_multi_processor // warp_size
    occupancy = active_warps / max_warps

    print(f"Estimated occupancy: {occupancy*100:.0f}% ({active_warps}/{max_warps} warps)")
    print(f"  Limiter: regs={max_blocks_regs}, shared={max_blocks_shared}, threads={max_blocks_threads}")
```

### 2.2 Shared Memory 演示

```python
if torch.cuda.is_available():
    # Triton kernel 使用 shared memory 的收益
    # 典型: matmul tiling: A[BLOCK_M, BLOCK_K] @ B[BLOCK_K, BLOCK_N]
    # 每个 block 用 shared memory 缓存 A tile 和 B tile

    import triton
    import triton.language as tl

    @triton.jit
    def demo_shared_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        # 用 shared memory 缓存
        a_tile = tl.zeros((BLOCK,), dtype=tl.float32)
        a_tile = tl.load(a_ptr + offs, mask=offs < n)
        tl.store(out_ptr + offs, a_tile * 2, mask=offs < n)

    print("Kernel uses shared memory for data reuse")
```

### 2.3 Tensor Core 利用

```python
if torch.cuda.is_available():
    # Tensor Core: fp16 @ fp16 -> fp32 accumulation
    # 需要: shape 是 8 的倍数, fp16, NHWC layout

    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    import time
    t0 = time.perf_counter()
    C = A @ B
    torch.cuda.synchronize()
    t = time.perf_counter() - t0

    tflops = (2 * 1024**3) / t / 1e12
    peak = 312  # A100 fp16 Tensor Core peak TFLOPS
    print(f"fp16 matmul: {tflops:.0f} TFLOPS ({tflops/peak*100:.0f}% of A100 fp16 peak)")
```

## 3. 核心源码路径

```
aten/src/ATen/native/cuda/            # CUDA kernel 实现
torch/_inductor/codegen/triton.py     # Inductor→Triton codegen, 隐含 warp/shared 优化
torch/_inductor/scheduler.py          # Scheduler 决定 kernel launch 参数
```

## 4. 和已有笔记的连接

```
25_inductor/         — Inductor 生成的 kernel 受 GPU 架构约束
26_triton_kernel/    — Triton 暴露 shared memory 和 block 控制
30_sdpa_attention/   — FlashAttention 极致利用 shared memory
03_cuda_stream/      — Stream 调度依赖 SM 资源分配
```

## 5. 搜索关键词

```bash
rg -n "shared_memory|smem" aten/src/ATen/native/cuda/
rg -n "tensor_core|wmma" aten/src/ATen/native/cuda/
rg -n "occupancy" torch/_inductor/codegen/
```
