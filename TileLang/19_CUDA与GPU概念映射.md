# 19 CUDA 与 GPU 概念映射（深度版）

> 本文目标：建立"写 TileLang 就是在写 CUDA 的抽象"的精确映射，重点是 layout 如何决定线程/寄存器分配。

## 1. 完整映射表

| CUDA 概念 | TileLang 对应 | 生成代码 |
| --- | --- | --- |
| gridDim | `T.Kernel(blocks_x, blocks_y)` | 网格配置 |
| blockIdx | `as (bx, by)` | `blockIdx.x/y` |
| blockDim | `threads=256` | `__launch_bounds__(256)` |
| threadIdx | `T.Parallel` 自动分配 | `threadIdx.x` |
| 全局内存 | `T.Tensor` 参数 | `float* A` |
| shared memory | `T.alloc_shared` | `__shared__ half A_shared[...]` |
| 寄存器 | `T.alloc_fragment` | 局部数组（寄存器） |
| `__syncthreads()` | ThreadSync pass 自动 | `__syncthreads()` |
| warp | layout 系统 | 32 线程 |
| 向量化 | 自动 | `float4` |
| cp.async | `T.copy` | `__pipeline_memcpy_async` |
| mma.sync | `T.gemm` | `mma.sync.aligned...` |
| TMA | Hopper T.copy | `tl::tma_load` |

## 2. fragment → 寄存器映射（深度）

以 m16n8k16 fp16 为例（`tilelang/cuda/intrinsics/layout/mma_layout.py:59-62`）：

```python
def mma_store_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
```

**验证（已确认）**：
- thread 0: local 0→(0,0), local 1→(0,1), local 2→(8,0), local 3→(8,1)
- thread 4: 行=1 → (1,0),(1,1),(9,0),(9,1)

即：16 行分两组各 8 行；4 线程一组（thread%4）各拿 2 列；(thread//4) 决定行偏移。128 元素 = 32 线程 × 4 本地。

## 3. A/B fragment 映射

A fragment（`mma_layout.py:99-101`，m16k16，每线程 8 元素）：
```python
def shared_16x16_to_mma_a_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)
```
- 与 PTX m16n8k16 的 A 寄存器布局完全一致（4 个 32 位寄存器 = 8 个 fp16）。

## 4. shared 内存布局与 bank conflict（深度）

swizzle 数学（`gemm_layouts.cc:502-520`，128B FullBank）：
```cpp
PrimExpr ts = FloorDiv(i, 8);
PrimExpr s  = FloorMod(i, 8);
PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
PrimExpr c  = FloorMod(FloorDiv(j, vector_size), 8);
PrimExpr vec = FloorMod(j, vector_size);
PrimExpr c_swizzle = xor8x8(c, s);      // 关键：c XOR s
PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
```

**bank 冲突消除验证**（fp16，stride=continuous=64，vector_size=8）：
无 swizzle：8 行同列 → 全 bank 0（8路冲突）。
有 swizzle：`index = vec + ((c⊕s)+8s)*8`：

| i | s | c⊕s | index | bank=(index/2)%32 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 72 | 4 |
| 2 | 2 | 2 | 144 | 8 |
| ... | ... | ... | ... | ... |
| 7 | 7 | 7 | 504 | 28 |

8 线程落在 **8 个不同 bank**，无冲突。

## 5. 同步语义

- block 间：无同步。
- block 内 shared：ThreadSync pass 自动插入 `__syncthreads()`。
- fragment：依赖 mma 语义保证。

## 6. 动手实验

```bash
mkdir -p /home/hpc/ghr_code/cuda_pytorch/TileLang/experiments/19_cuda_map
# 写 elementwise + matmul，get_kernel_source 对照映射表
```

## 7. 深入自测

1. m16n8k16 的 C fragment：thread 0 持有哪些元素？
2. A fragment 的线程/本地映射公式？
3. 128B swizzle 如何消除 bank conflict？手算 8 个 bank。
4. `threads=256` 对应 CUDA 什么？
5. `T.copy` 生成哪些指令？

## 8. 下一步

进入 `20_核心数据结构.md`（深度版）。
