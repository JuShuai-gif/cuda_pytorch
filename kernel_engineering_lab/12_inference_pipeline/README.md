# 模块 12：推理管线

生产级 LLM 推理管线的完整演示，包含 Transformer 模型、KV cache 管理、prefill/decode 两阶段推理流程及全面 benchmark。

## 工业背景

在生产 LLM 推理服务中（如 **vLLM**、**TGI**、**TensorRT-LLM**），推理管线是决定服务质量和成本的核心组件。与传统训练不同，推理面临以下挑战：

- **延迟敏感**：用户期望 <100ms 的首 token 延迟（TTFT, Time To First Token），每 token 生成延迟需稳定（TPOT, Time Per Output Token）
- **显存瓶颈**：KV cache 是最大的显存消费者——GPT-3 175B 的完整 KV cache 可达数十 TB
- **并发请求**：需同时服务数百个请求，资源调度复杂
- **两阶段特性**：prefill 和 decode 的计算特征截然不同，需针对性优化

### Prefill vs Decode

| 阶段 | 输入 | 计算特征 | 瓶颈 |
|------|------|----------|------|
| **Prefill** | `[batch, prompt_len, hidden]` | 计算密集型，O(L²) 注意力 | GPU 算力（SM 利用率） |
| **Decode** | `[batch, 1, hidden]` | 内存密集型，O(L) 注意力 | 显存带宽（KV cache 读取） |

**Prefill** 一次性处理 prompt 中的所有 token，需要计算整个因果注意力矩阵，是计算瓶颈。
**Decode** 自回归生成时每次只处理 1 个新 token，但需要读取完整的 KV cache，是内存瓶颈。

### KV Cache 管理

KV cache 存储所有层、所有 token 的 Key 和 Value 投影结果：

```
显存 = 2 × num_layers × batch_size × num_heads × seq_len × head_dim × dtype_size
```

例如：12 层、8 batch、32 头、2048 seq、128 dim、FP16 = 约 3.0 GB

**PagedAttention**（vLLM 提出）将 KV cache 划分为固定大小的 block（如 16 个 token/block），所有序列共享 block 池，按需分配。相比连续分配：
- 无预分配浪费
- 无内存碎片
- 利用率从 20-40% 提升到 90%+

### 批处理策略

- **Static batching**：固定 batch size，等所有请求完成后才能加入新请求
- **Continuous batching**（也叫 In-flight batching）：TGI 和 vLLM 的核心优化。当某个请求完成时立即从 batch 移除，插入新的 prefill 请求，无需等待整个 batch 完成
- **Chunked prefill**：将长 prompt 切分为多个 chunk，与 decode 交替执行，防止单个长 prompt 抢占 GPU

### 常见坑

1. **KV cache 显存过度分配**：为 max_seq_len 预分配导致大量浪费。多使用 PagedAttention 或动态分配。
2. **Prefill-decode 干扰**：一个 batch 中同时存在 prefill 和 decode 请求时，prefill 的计算密度会拖慢 decode 的延迟。使用 chunked prefill 或分离调度缓解。
3. **小 batch size 的批处理开销**：batch=1 时 GPU 利用率极低（<5%）。聚合多个请求到更大 batch 可显著提升吞吐，但需权衡延迟。
4. **dtype 一致性**：KV cache 的 dtype 必须与模型 dtype 一致。常见错误：模型用 FP16 但 KV cache 用 FP32，导致 2× 显存消耗且带宽加倍。
5. **KV cache 未对齐**：block size 不整除 seq_len 时产生浪费。选择合适的 block_size（16/32/64）。
6. **忽略 embedding 层**：在实际模型中 embedding 和 lm_head 可能占总参数量 20-30%，benchmark 中必须计入。

## 文件结构

| 文件 | 功能 |
|------|------|
| `pipeline.py` | TransformerBlock、OptimizedTransformer、InferencePipeline 实现 |
| `kv_cache.py` | KVCache（连续分配）和 PagedKVCache（分页分配） |
| `benchmark_all_ops.py` | 所有模块 kernel 的统一 benchmark |
| `test_pipeline.py` | 推理管线 pytest 测试套件 |
| `benchmark_pipeline.py` | 推理管线端到端 benchmark |
| `README.md` | 本文件 |

## 核心类

### TransformerBlock
单个 Transformer block，包含自注意力（QKV 投影 + 注意力计算 + 输出投影）和 FFN（gate + GELU + down）。支持两种模式：
- `use_fusions=True`：使用自定义 Triton kernel（tiled attention、fused residual+layernorm、fused bias+GELU）
- `use_fusions=False`：纯 PyTorch eager 实现

### OptimizedTransformer
N 层 TransformerBlock 堆叠，构成完整模型。

### InferencePipeline
封装 prefill 和 decode 两阶段推理逻辑：
- `prefill()`：处理完整 prompt，一次性填充 KV cache
- `decode_step()`：单 token 解码，读取/更新 KV cache
- `generate()`：完整自回归生成流程

### KVCache
连续分配的 KV cache，适合小 batch/短序列场景：
```python
cache = KVCache(num_layers=12, batch_size=4, num_heads=32, max_seq_len=2048, head_dim=128)
cache.update(layer_idx, batch_idx, k, v, positions)
k_cached, v_cached = cache.get(layer_idx, batch_idx, up_to=128)
```

### PagedKVCache
简化版 PagedAttention 实现，block 粒度管理：
```python
pcache = PagedKVCache(num_layers=12, num_heads=32, max_seq_len=4096, head_dim=128, block_size=16)
seq_id = pcache.allocate_sequence()
pcache.grow(seq_id, num_blocks_needed)
pcache.write(seq_id, k, v, start_pos=0)
k_out, v_out = pcache.read(seq_id, layer_idx)
pcache.free_sequence(seq_id)
```

## 使用方法

### 运行测试
```bash
# 在项目根目录
cd kernel_engineering_lab

# 运行全部测试
pytest 12_inference_pipeline/test_pipeline.py -v

# 运行特定测试类
pytest 12_inference_pipeline/test_pipeline.py::TestTransformerBlock -v

# 仅运行 KV cache 测试
pytest 12_inference_pipeline/test_pipeline.py::TestKVCache -v
```

### 运行 benchmark
```bash
# 运行全部 benchmark
python 12_inference_pipeline/benchmark_pipeline.py

# 输出报告到文件
python 12_inference_pipeline/benchmark_pipeline.py --output pipeline_report

# 跳过 torch.compile 对比（减少耗时）
python 12_inference_pipeline/benchmark_pipeline.py --skip-compile
```

### 运行独立 demo
```bash
# 运行管线 demo
python 12_inference_pipeline/pipeline.py

# 运行 KV cache demo
python 12_inference_pipeline/kv_cache.py

# 运行统一 kernel benchmark
python 12_inference_pipeline/benchmark_all_ops.py
```

## 依赖

- PyTorch >= 2.0
- Triton >= 2.0
- pytest（测试）
- tabulate（报告格式化）

## Benchmark 维度

`benchmark_pipeline.py` 覆盖以下对比维度：

1. **eager vs fused**：纯 PyTorch 实现 vs 自定义 Triton 融合 kernel
2. **torch.compile vs 自定义融合**：torch.compile 自动优化与传统手写 kernel 对比
3. **模型规模**：hidden=512/1024/4096，heads=8/16/32，layers=4/8/16
4. **Prefill 延迟**：prompt 长度 32/128/512/1024
5. **Decode 延迟**：逐 token 生成 100 步
6. **吞吐量**：batch size 1/4/8/16 下的 tokens/sec
7. **峰值显存**：不同规模模型的内存占用

## 相关模块

- [04_operator_fusion](../04_operator_fusion/)：fused_add_relu、fused_bias_gelu、fused_residual_layernorm
- [06_attention_flash_like](../06_attention_flash_like/)：tiled_attention、attention_decode
- [02_triton_basics](../02_triton_basics/)：基础 Triton kernel
