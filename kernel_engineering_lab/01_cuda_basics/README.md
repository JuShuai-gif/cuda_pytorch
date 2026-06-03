# 01_cuda_basics - PyTorch 的 CUDA C++ 扩展

## 工业背景

当 Triton 无法表达某些模式时，自定义 CUDA kernel 必不可少：

- **Warp 级操作**（shuffle、ballot、match），用于跨线程通信
- **动态并行**（kernel 启动 kernel）
- **跨 block 同步**，超出 cooperative groups 所能提供的范围
- **Tensor Core MMA 指令**，实现细粒度控制
- **自定义数据类型**，不受 Triton 类型系统支持

使用自定义 CUDA kernel 的生产系统：

| 系统 | 用例 |
|--------|----------|
| **vLLM** | PagedAttention、融合 MoE、自定义量化 |
| **TensorRT-LLM** | 优化 attention、MLP 融合、KV-cache 操作 |
| **FlashInfer** | GPU 加速采样、不规则 attention |
| **xFormers** | 内存高效 attention、稀疏操作 |

## 构建

```bash
# 原地构建（扩展模块写在 setup.py 旁边）
python 01_cuda_basics/setup.py build_ext --inplace

# 或设置自定义计算能力目标
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" python 01_cuda_basics/setup.py build_ext --inplace
```

构建后，按如下方式导入：
```python
import cuda_basics_kernels
result = cuda_basics_kernels.vector_add(a, b)
total = cuda_basics_kernels.reduce_sum(x)
```

## 运行测试

```bash
pytest 01_cuda_basics/test_cuda_basics.py -v
```

## 运行基准测试

```bash
python 01_cuda_basics/benchmark_cuda_basics.py
```

## Kernel 详情

### vector_add
- **Grid**：`ceil(N / 256)` 个 block，256 线程/block
- **策略**：每个线程处理一个元素，带边界检查
- **内存**：全局内存读/写，合并访问

### reduce_sum
- **Grid**：`min(ceil(N / 256), 1024)` 个 block，256 线程/block
- **共享内存**：每 block 256 个 float（1 KB）
- **策略**：在共享内存中进行 block 级顺序归约，然后使用 `atomicAdd` 进行跨 block 归约
- **局限性**：当多个 block 写入同一地址时，`atomicAdd` 会产生争用。后续模块将展示 warp-shuffle 和多级归约。

## 常见陷阱

### 计算能力不匹配
以高于 GPU 的 SM 目标构建会导致"no kernel image available"错误。
使用 `nvidia-smi --query-gpu=compute_cap --format=csv` 验证。

### Block 大小选择
- **太小**：GPU 利用不足（warp 为 32 线程）
- **太大**：超出 `maxThreadsPerBlock` 或 `maxSharedMemoryPerBlock`
- **推荐**：32 的倍数，逐元素操作通常为 128-512

### 共享内存 Bank 冲突
在归约中，当 `s` 为 2 的幂时，顺序寻址（按 `s` 步进）会导致 bank 冲突。
这个朴素 kernel 存在此问题。后续模块将解决它。

### Warp 分支发散
在归约中，当 `tid < s` 发散时，warp 内的线程走不同路径。
这会使有效 warp 利用率减半。warp-shuffle 归约可以避免此问题。
