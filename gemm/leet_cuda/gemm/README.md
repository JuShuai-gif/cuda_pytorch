# CUDA SGEMM 优化系列 - 知识点总结

本目录包含从基础到优化的多个SGEMM（单精度矩阵乘法）实现版本，展示了CUDA编程中常见的优化技术。

## 文件说明

| 文件 | 描述 | 优化层级 |
|------|------|----------|
| `sgemm_cublas.cu` | cuBLAS库实现，作为性能基准 | N/A (最优参考) |
| `sgemm_naive.cu` | 最基本的GPU实现 | Level 0 (基准) |
| `sgemm_v1.cu` | 共享内存分块版本 | Level 1 |
| `sgemm_v2.cu` | 寄存器缓存优化版本 | Level 2 |
| `sgemm_v3.cu` | 双缓冲优化版本 | Level 3 |

## 核心概念

### 1. 矩阵分块 (Block Tiling)

**原理**：将大矩阵划分为小块(Block Tile)，每个CUDA block负责计算一个输出块。

```
        C = A × B
        
    A (M×K)     B (K×N)     C (M×N)
    
    ┌─────┐             ┌─────┐
    │块1  │   ┌───┐     │块1  │
    │     │ × │   │ =   │     │
    └─────┘   └───┘     └─────┘
```

**参数说明**：
- `BM`: Block M - 每个block处理的M维度大小
- `BN`: Block N - 每个block处理的N维度大小  
- `BK`: Block K - 每次加载到共享内存的K维度大小

### 2. 共享内存 (Shared Memory)

**特点**：
- GPU上访问延迟最低的内存（~30-50周期 vs 全局内存~400周期）
- 每个SM有有限的共享内存大小（通常48KB/块）
- 需要手动管理，使用`__syncthreads()`同步

**典型使用模式**：
```cuda
__shared__ float s_a[BM][BK];  // 共享内存缓存
__shared__ float s_b[BK][BN];
```

### 3. 寄存器优化 (Register Caching)

**原理**：将共享内存数据预加载到寄存器，减少共享内存访问次数。

**寄存器分工**：
- `r_c[TM][TN]`: 累加结果（核心计算单元）
- `r_load_a/b`: 全局内存加载缓存
- `r_comp_a/b`: 共享内存读取缓存

### 4. 双缓冲 (Double Buffering)

**目的**：隐藏内存访问延迟，计算与访存并行

**原理**：
```
轮次1: 计算Buffer0 ────── 加载Buffer1
轮次2: 计算Buffer1 ────── 加载Buffer0
...
```

**实现**：
```cuda
__shared__ float s_a[2][BK][BM];  // 双缓冲
int smem_sel = (bk - 1) & 1;       // 当前使用
int smem_sel_next = bk & 1;        // 预加载目标
```

### 5. 向量化内存访问

**原理**：使用`float4`一次加载4个float，提高内存带宽利用率

**宏定义**：
```cuda
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
```

**效果**：
- 内存访问次数减少4倍
- 利用GPU的内存合并访问(coalesced access)

### 6. 循环展开 (#pragma unroll)

**作用**：让编译器展开循环，减少循环控制开销，增加指令级并行

```cuda
#pragma unroll
for (int k = 0; k < BK; k++) {
    // 计算逻辑
}
```

## 性能对比（典型值）

| 版本 | 性能(相对cuBLAS) | 主要瓶颈 |
|------|------------------|----------|
| naive | ~5-10% | 全局内存访问 |
| v1 | ~30-50% | 共享内存延迟 |
| v2 | ~50-70% | 指令调度 |
| v3 | ~70-90% | 接近硬件极限 |

## 关键优化技术总结

### 内存层次结构优化
```
全局内存 ──(加载)──> 共享内存 ──(加载)──> 寄存器 ──(计算)──> 寄存器
  (慢)                    (快)              (最快)
```

### 性能优化优先级
1. **内存访问优化** - 减少全局内存访问，使用共享内存
2. **计算密度提升** - 每个线程处理更多数据
3. **指令级并行** - 循环展开，编译器优化
4. **隐藏延迟** - 双缓冲，计算与访存重叠

## 常见参数配置

对于`BM=128, BN=128, BK=8, TM=8, TN=8`:
- 每个block的线程数: `(BN/TN) × (BM/TM) = 16 × 16 = 256` threads/block
- 每个线程计算: `TM × TN = 8 × 8 = 64` 个输出元素
- 共享内存使用: `128×8×4 + 128×8×4 = 16KB`

## 测试矩阵尺寸

```cuda
M_list = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384}
N_list = 同M_list
K_list = 1024 (固定)
```

## 编译运行

```bash
# 编译
nvcc -o sgemm_v1 sgemm_v1.cu -lcublas

# 运行
./sgemm_v1
```

## 验证正确性

每个实现都包含`testError()`函数：
1. 在CPU上运行参考实现
2. 在GPU上运行优化版本
3. 比较两者的最大误差

```cuda
float max_error = 0.0;
for (int i = 0; i < M * N; i++) {
    max_error = max(max_error, abs(h_d_c[i] - h_c[i]));
}
```

## 性能计算

```cuda
// GFLOPS = (M × N × K × 2) / 时间 / 10^9
// 乘以2是因为每次乘加操作包含一次乘法和一次加法
double Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / sec;
```

## 扩展学习

- **Warp级优化**: 利用warp内的线程协作
- **Tensor Core**: 使用混合精度矩阵乘法
- **多流并行**: 重叠多个kernel执行
- **统一内存**: 简化内存管理

## 参考资料

- CUDA C Programming Guide
- CUDA Best Practices Guide
- "Optimizing Matrix Multiply" - NVIDIA GTC Talks
