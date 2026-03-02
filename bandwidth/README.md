# CUDA 内存带宽与延迟基准测试

本项目 forked from https://github.com/Yinghan-Li/YHs_Sample/blob/master/cuda/microbenchmark

## 概述

这是一个用于测量 NVIDIA GPU 内存层次结构性能的基准测试套件。它可以帮助开发者了解不同内存类型的带宽和延迟，从而更好地优化 CUDA 程序。

## 为什么需要这些测试?

GPU 拥有复杂的内存层次结构,不同层级的内存具有显著不同的性能特征:

1. **共享内存 (Shared Memory)**: 每个 SM 上最快的内存,延迟极低(~1-2周期),但容量小(~48KB/SM)
2. **L1 缓存**: 位于每个 SM 上,延迟低(~5-10周期),用于缓存局部数据和指令
3. **L2 缓存**: 所有 SM 共享,延迟中等(~30-50周期),容量较大(1-6MB)
4. **DRAM (全局内存)**: 延迟最高(~400-800周期),但容量最大

理解这些性能差异对于优化 CUDA 程序至关重要:
- 将频繁访问的数据放入共享内存
- 合理利用 L1/L2 缓存
- 避免不必要的全局内存访问

## 测试文件说明

### 带宽测试

| 文件 | 测试内容 | 说明 |
|------|----------|------|
| `dram_bandwidth.cu` | DRAM 带宽 | 测量 GPU 到全局内存的读写带宽,包括纯读取、纯写入和同时读写三种模式 |
| `l2cache_bandwidth.cu` | L2 缓存带宽 | 测量 L2 缓存的访问带宽,数据大小(2MB)小于 L2 缓存确保缓存命中 |
| `smem_bandwidth.cu` | 共享内存带宽 | 测量共享内存的写入带宽,支持不同 GPU 架构(Kepler, Maxwell+) |

### 延迟测试

| 文件 | 测试内容 | 说明 |
|------|----------|------|
| `dram_latency.cu` | DRAM 延迟 | 测量全局内存访问延迟,使用大 stride(1024B)避免 L2 缓存命中 |
| `l2cache_latency.cu` | L2 缓存延迟 | 测量 L2 缓存访问延迟,使用小 stride(128B)确保缓存命中 |
| `l1cache_latency.cu` | L1 缓存延迟 | 测量 L1 缓存访问延迟,使用指针链式解引用访问缓存数据 |
| `smem_latency.cu` | 共享内存延迟 | 测量共享内存访问延迟,这是 GPU 上最低延迟的内存访问 |

## 技术实现要点

1. **依赖加载链**: 延迟测试使用依赖加载(每次加载地址依赖上次结果),确保延迟无法被并行访问隐藏
2. **CUDA 事件计时**: 使用 `cudaEventRecord` 和 `cudaEventElapsedTime` 获得精确的毫秒级计时
3. **内联汇编**: 使用 PTX 内联汇编确保使用特定指令(如 `ldg.cs`, `stg.cs`, `ld.shared.b32`)
4. **缓存刷清**: DRAM 延迟测试前使用大量数据刷清 L2 缓存,确保访问真实 DRAM
5. **预热循环**: 运行预热迭代使 GPU 达到稳定状态,减少测量误差

## 构建与运行

### 构建

```bash
# 带宽测试
sh build.sh dram_bandwidth.cu 90
sh build.sh l2cache_bandwidth.cu 90
sh build.sh smem_bandwidth.cu 90

# 延迟测试
sh build.sh dram_latency.cu 90
sh build.sh l1cache_latency.cu 90
sh build.sh l2cache_latency.cu 90
sh build.sh smem_latency.cu 90
```

### 运行

```bash
./a.out
```

## 预期结果

典型的现代 GPU 测试结果(数值仅供参考,实际因 GPU 架构而异):

- **共享内存带宽**: ~3000+ GB/s (每 SM)
- **共享内存延迟**: ~1-2 周期
- **L1 缓存延迟**: ~5-10 周期
- **L2 缓存延迟**: ~30-50 周期
- **L2 缓存带宽**: ~500-1500 GB/s
- **DRAM 延迟**: ~400-800 周期
- **DRAM 带宽**: ~500-1500 GB/s (取决于 GPU)

## 优化建议

基于测试结果,可以采取以下优化策略:

1. **使用共享内存**: 对于频繁访问的数据,手动管理共享内存可获得最佳性能
2. **合并访问**: 全局内存访问应尽量合并,以提高带宽利用率
3. **避免 bank 冲突**: 共享内存访问应避免同一 warp 内访问相同 bank
4. **利用缓存**: 合理的数据布局可以充分利用 L1/L2 缓存
5. **预取**: 对于可预测的访问模式,使用软件预取隐藏延迟

## 支持

支持 Kepler+(sm_30+) 及以上架构的 GPU 设备
