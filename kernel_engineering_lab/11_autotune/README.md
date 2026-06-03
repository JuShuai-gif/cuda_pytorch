# 模块 11：Kernel 自动调优

## 工业背景

GPU kernel 性能严重依赖于超参数（block size、num_warps、num_stages），这些参数因 GPU 架构、问题规模和数据类型而异。**Auto-tuning** 系统地搜索参数空间以找到最优配置。

### 为什么 Auto-tune 至关重要

手动调优不切实际，原因如下：
- **组合爆炸**：block size (32-256) × warps (2-16) × stages (1-4) = 数百种配置
- **架构依赖性**：对 A100 最优的配置可能对 H100 是次优的
- **问题规模依赖性**：小矩阵需要与大矩阵不同的分块方式
- **Dtype 依赖性**：fp16、bf16、fp32 偏好不同的 block size

### 生产环境中 Auto-tune 的使用场景

| 系统 | Auto-tuning 策略 |
|--------|-------------------|
| **FlashAttention** | 按序列长度自动调优 block size |
| **xFormers** | 自动调优 attention kernel 配置 |
| **torch.inductor** | 自动调优所有生成的 Triton kernel |
| **Triton DSL** | `@triton.autotune` 装饰器 |
| **CUTLASS** | 编译时模板自动调优 |
| **cuBLAS** | NVIDIA 提供预调优的启发式算法 |

### Triton 的 Autotune 机制

```
@triton.autotune(
    configs=[...],     # 要尝试的 Config 对象列表
    key=['M', 'N'],    # 缓存 key：问题维度
)
@triton.jit
def kernel(...):
    pass
```

**流程：**
1. 对给定 `key` 的首次调用：Triton 运行所有配置，测量每个配置，选择最快的
2. 后续相同 `key` 的调用：直接使用缓存的最佳配置
3. 缓存存储在 `~/.triton/cache/` 中，key 为 kernel 哈希 + key 值
4. 不满足硬件约束的配置会被自动剔除

### Config 空间设计原则

1. **保持有界**：配置太多 = 第一次调用很慢。使用 2 的幂和合理的范围
2. **使用 key 参数**：key 应该是影响配置选择的问题维度
3. **约束剔除**：Triton 自动跳过超出硬件限制的配置
4. **预热启动**：在模型编译期间预调优已知形状

### 常见陷阱

1. **过大的配置空间**：1000+ 配置导致首次编译缓慢（几分钟）。生产环境限制在 50-200 个配置内
2. **忘记 `key` 参数**：没有 `key`，每次调用都重新 auto-tune。始终指定问题维度
3. **缺少有效配置**：如果最优配置不在搜索空间中，你永远找不到它。包含合理的默认值
4. **编译 vs 运行时权衡**：Auto-tune 开销通过多次 kernel 调用摊销。对于一次性 kernel，固定配置更好
5. **不同的 GPU 架构**：在 A100 上 auto-tune 的配置将与 H100 分开缓存（Triton 在缓存 key 中包含 GPU 架构）

## 文件

| 文件 | 用途 |
|------|---------|
| `triton_autotune_demo.py` | 使用 `@triton.autotune` 的自动调优 matmul |
| `layernorm_autotune.py` | 自动调优的 LayerNorm 和 RMSNorm kernel |
| `softmax_autotune.py` | 自动调优的在线 softmax kernel |
| `autotune_report.py` | 综合报告生成（JSON + Markdown） |
| `test_autotune.py` | 正确性和行为的 pytest 测试 |
| `benchmark_autotune.py` | vs 固定配置的性能基准测试 |

## 运行

```bash
# 演示 matmul 自动调优
python 11_autotune/triton_autotune_demo.py

# 演示 LayerNorm/RMSNorm 自动调优
python 11_autotune/layernorm_autotune.py

# 演示 softmax 自动调优
python 11_autotune/softmax_autotune.py

# 生成综合报告
python 11_autotune/autotune_report.py

# 测试
pytest 11_autotune/test_autotune.py -v

# 基准测试
python 11_autotune/benchmark_autotune.py
```
