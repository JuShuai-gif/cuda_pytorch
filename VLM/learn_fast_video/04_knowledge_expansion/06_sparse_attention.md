# 稀疏注意力

> 知识点扩展：dense vs sparse、block sparse、VSA、BSA、SLA、VMoBA，回扣 FastVideo 源码。

## 1. 为什么稀疏

视频 DiT 序列长度数万，dense attention O(L²) 不可接受。观察：attention 分数高度稀疏——每个 query 只与少数 key 强相关（尤其时空邻近的）。稀疏 attention 只计算这些相关部分，降到近似 O(L·k)。

### 1.1 稀疏 attention 的三个核心问题

任何稀疏方法都要回答：
1. **粒度**：按 token 还是 block（块）稀疏？block 稀疏对硬件友好（连续访存、能用 tensor core），token 稀疏灵活但慢。FastVideo 全用 **block 稀疏**。
2. **选择**：哪些 block 保留？——按 block 间相似度/attention score 选 top-k，或按固定模式（滑窗）。
3. **实现**：如何高效跳过未选 block？——需要 gather/scatter 重排 token（tile/untile）+ 专用 kernel。

### 1.2 稀疏 vs 精确的取舍

稀疏是**近似**（丢弃了部分 attention），可能掉质量。FastVideo 的应对：
- 训练无关方法（BSA）：直接推理加速，接受轻微掉点。
- 训练配合方法（VSA + sparse distillation）：训练时就用稀疏，模型学会适应，几乎无损。

## 2. Dense vs Sparse 在源码的体现

```python
# Dense（sdpa.py）：接口简单
def forward(self, q, k, v, metadata):
    return F.scaled_dot_product_attention(q.T, k.T, v.T)

# Sparse（video_sparse_attn.py）：需 tile + metadata
def preprocess_qkv(self, qkv, metadata):
    return self.tile(qkv, metadata)         # token → block 重排
def forward(self, q, k, v, gate, metadata):
    cur_topk = (1 - VSA_sparsity) * num_kv_blocks
    return video_sparse_attn(q, k, v, ..., cur_topk)
def postprocess_output(self, output, metadata):
    return self.untile(output, ...)
```

Sparse 后端需要 `AttentionMetadata` 携带 tile 划分、block 大小、稀疏度。

## 3. Block Sparse Attention

把序列分成 block，只计算选中的 (Q block, KV block) 对：
```
稠密：所有 QᵢKⱼ
块稀疏：只算 top-k 个 KV block 对每个 Q block
```
底层 kernel：`fastvideo-kernel/csrc/attention/block_sparse_h100.cu`（ThunderKittens）。给定稀疏索引做 exact attention。

## 4. VSA（Video Sparse Attention）

```
源码：attention/backends/video_sparse_attn.py + fastvideo-kernel
论文：arXiv:2505.13389
```
双分支架构：
- **压缩分支**：block mean pooling + full attention（捕获全局）。
- **稀疏分支**：top-k block selection + block-sparse attention（捕获局部细节）。
- 用 `gate_compress` 融合两分支。

稀疏度控制：`cur_topk = (1 - VSA_sparsity) * num_kv_blocks`。`VSA_sparsity=0.9` → 只保留 10% block。VSA 是 FastVideo 稀疏蒸馏（sparse distillation）的基础。

## 5. BSA（Bidirectional Sparse Attention）

```
源码：attention/backends/bsa_attn.py（740 行）
论文：arXiv:2509.01085
```
训练无关的推理加速，对 query 和 KV 同时稀疏化：
1. **Query 剪枝**（`_prune_queries`）：block 内按 cosine similarity 保留最不相似 token。
2. **KV block 选择**（`_select_kv_blocks`）：按 block attention score + 累计阈值动态选。
3. 对选中 block 做 exact attention（flash-attn varlen 加速）。

## 6. SLA（Sparse-Linear Attention）

```
源码：attention/backends/sla.py
论文：arXiv:2509.24006
```
稀疏 + 线性 attention 混合：
- 稀疏分支：block-sparse（捕获重要交互）。
- 线性分支：`q@(kᵀ@v)`（O(L) 全局近似）。
- 融合：`output = o_sparse + proj_l(o_linear)`（proj_l 初始化为零）。

## 7. VMoBA（Video Mixture of Block Attention）

```
源码：attention/backends/vmoba.py + fastvideo-kernel/vmoba.py
```
MoE 风格：KV 分 chunk，每个 query head 通过 gate 选 top-k chunk，只对选中 chunk 做 flash-attn。按 layer index 选 temporal/spatial/spatiotemporal chunk 类型。

## 8. STA（Sliding Tile Attention）

```
源码：fastvideo-kernel/csrc/attention/st_attn_h100.cu
```
视频 3D 滑动窗口：每个 query 只 attend 时空邻域内的 KV。硬编码多种窗口大小（3×3×3 到 6×6×10）。

## 9. 各稀疏方法对比

| 方法 | 稀疏策略 | 训练需求 | 底层 |
|------|---------|---------|------|
| VSA | 双分支（压缩+top-k block） | 可蒸馏 | fastvideo-kernel |
| BSA | query 剪枝 + KV block 选择 | 训练无关 | flash-attn varlen |
| SLA | 稀疏 + 线性混合 | 需训练 proj | Triton + sage |
| VMoBA | chunk + per-head top-k | - | fastvideo-kernel |
| STA | 时空滑窗 | - | ThunderKittens |

## 10. 回扣源码
| 方法 | 后端 | kernel |
|------|------|--------|
| VSA | `video_sparse_attn.py` | `block_sparse_h100.cu` |
| BSA | `bsa_attn.py` | flash-attn |
| SLA | `sla.py` | `sla_triton.py` |
| VMoBA | `vmoba.py` | `fastvideo-kernel/vmoba.py` |
| STA | `ops.py:sliding_tile_attention` | `st_attn_h100.cu` |

## 11. 延伸
- attention 加速：[`05_attention_acceleration.md`](05_attention_acceleration.md)
- kernel：[`../02_source_by_directory/11_fastvideo_kernel.md`](../02_source_by_directory/11_fastvideo_kernel.md)
- 稀疏蒸馏：[`10_distillation_dmd_sparse_distill.md`](10_distillation_dmd_sparse_distill.md)
