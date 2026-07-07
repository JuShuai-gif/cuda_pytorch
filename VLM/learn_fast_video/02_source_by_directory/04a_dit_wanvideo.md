# DiT 精读 · WanVideo

> 源码：`/home/hpc/ghr_code/FastVideo/fastvideo/models/dits/wanvideo.py`（748 行）
> 关键类：`WanTransformer3DModel`(L561)、`WanTransformerBlock`(L242)、`WanTimeTextImageEmbedding`(L54)、`WanT2VCrossAttention`(L153)、`WanI2VCrossAttention`(L190)
>
> Wan 是 FastVideo 最典型、最推荐先读的 DiT。本文逐层拆解结构、forward、张量形状。

## 1. 架构常量（以 Wan2.1 T2V 1.3B 为例）

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `num_attention_heads` | 12 (1.3B) / 40 (14B) | 注意力头数 |
| `attention_head_dim` | 128 | 每头维度 |
| `inner_dim` = heads×head_dim | 1536 / 5120 | Transformer 宽度 |
| `in_channels` / `out_channels` | 16 / 16 | VAE latent 通道 |
| `patch_size` | `(1, 2, 2)` | 时空 patch（时间不切，空间 2×2） |
| `num_layers` | 30 / 40 | block 数 |
| `text_dim` | 4096 | UMT5 文本维度 |
| `freq_dim` | 256 | timestep 正弦编码维度 |
| `qk_norm` | `rms_norm_across_heads` | QK 归一化 |

## 2. __init__ 子模块（L570-630）

```python
inner_dim = num_attention_heads * attention_head_dim
# 1. patch embedding: Conv3d 切 patch
self.patch_embedding = PatchEmbed(in_chans=16, embed_dim=inner_dim, patch_size=(1,2,2), flatten=False)
# 2. 条件嵌入
self.condition_embedder = WanTimeTextImageEmbedding(dim=inner_dim, time_freq_dim=256, text_embed_dim=4096, image_embed_dim=...)
# 3. transformer blocks（VSA 时用 WanTransformerBlock_VSA）
transformer_block = WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" else WanTransformerBlock
self.blocks = nn.ModuleList([transformer_block(...) for i in range(num_layers)])
# 4. 输出
self.norm_out = LayerNormScaleShift(inner_dim, ...)
self.proj_out = nn.Linear(inner_dim, out_channels * prod(patch_size))   # inner_dim → 16*1*2*2=64
self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
```

**关键约束**（L584）：`num_attention_heads % sp_world_size == 0`——头数必须能被序列并行度整除（因为 SP 的 all-to-all 要按头切分）。

## 3. forward 逐步（L632-745）

```python
def forward(self, hidden_states, encoder_hidden_states, timestep, encoder_hidden_states_image=None, guidance=None):
    B, C, T, H, W = hidden_states.shape          # [1, 16, 21, 60, 104]
    p_t, p_h, p_w = self.patch_size              # (1, 2, 2)
```

### Step 1：3D RoPE（L656-667）
```python
d = hidden_size // num_attention_heads          # 每头维度 128
rope_dim_list = [d - 4*(d//6), 2*(d//6), 2*(d//6)]   # 三轴分配（时/高/宽）
freqs_cos, freqs_sin = get_rotary_pos_embed((T//p_t, H//p_h, W//p_w), ...)
```
RoPE 按 head_dim 分配给时间/高/宽三个轴，让模型感知 3D 位置。

### Step 2：patchify（L669-670）
```python
hidden_states = self.patch_embedding(hidden_states)      # [1, inner_dim, 21, 30, 52]
hidden_states = hidden_states.flatten(2).transpose(1, 2) # [1, L, inner_dim]，L=21*30*52=32760
```

### Step 3：序列并行切分（L672-677）
```python
hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
# 每个 SP rank 只持有 L/sp_size 个 token（含 padding）
```

### Step 4：条件嵌入（L687-694）
```python
temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
    timestep, encoder_hidden_states, encoder_hidden_states_image, ...)
timestep_proj = timestep_proj.unflatten(1, (6, -1))    # [1, 6, inner_dim]，6=shift/scale/gate×2
```
- `temb`：timestep 嵌入 `[B, inner_dim]`。
- `timestep_proj`：AdaLN 调制参数 `[B, 6, inner_dim]`。
- 文本经 `text_embedder`（MLP 4096→inner_dim）。

### Step 5：I2V 图像条件拼接（L696-701）
```python
if encoder_hidden_states_image is not None:
    encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
    # 图像 CLIP 特征拼在文本前面，供 cross attention
```

### Step 6：30/40 个 block（L710-719）
```python
for block in self.blocks:
    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, original_seq_len)
```

### Step 7：输出 norm + gather + unpatchify（L720-745）
```python
shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
hidden_states = self.norm_out(hidden_states, shift, scale)
hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)  # SP 汇聚
hidden_states = self.proj_out(hidden_states)              # → [1, L, 64]
# unpatchify: reshape + permute → [1, 16, 21, 60, 104]
```
输出与输入同形状 `[1, 16, 21, 60, 104]`——预测的速度（flow matching）。

## 4. WanTransformerBlock 内部（L242-404）

结构：**Self-Attn → Cross-Attn → FFN**，每个子层前 AdaLN 调制。

```python
# AdaLN 调制参数（6 组来自 scale_shift_table + temb）
e = self.scale_shift_table + temb.float()
shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=1)

# 1. Self-Attention
norm_hidden_states = self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa   # AdaLN
query = self.to_q(norm_hidden_states); key = self.to_k(...); value = self.to_v(...)
query = self.norm_q(query); key = self.norm_k(key)                                     # QK-Norm(RMS)
# reshape 到多头 [B, L, heads, head_dim]
attn_output, _ = self.attn1(query, key, value, original_seq_len, freqs_cis=freqs_cis)  # DistributedAttention(RoPE)
attn_output, _ = self.to_out(attn_output)
norm_hidden_states, hidden_states = self.self_attn_residual_norm(hidden_states, attn_output, gate_msa, ...)

# 2. Cross-Attention（文本条件）
attn_output = self.attn2(norm_hidden_states, context=encoder_hidden_states, context_lens=None)
norm_hidden_states, hidden_states = self.cross_attn_residual_norm(hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

# 3. FFN
ff_output = self.ffn(norm_hidden_states)                     # MLP(gelu_pytorch_tanh)
hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
```

要点：
- **QK-Norm**：`rms_norm_across_heads` 对整个 dim 做 RMSNorm（L279）。
- **DistributedAttention**：内部做 SP all-to-all + RoPE + attention 后端。
- **ScaleResidualLayerNormScaleShift**：融合"gate 残差 + LayerNorm + 下一层 shift/scale"。

## 5. Cross Attention：T2V vs I2V

### WanT2VCrossAttention（L153-187）
Q 来自视频 token，K/V 来自文本。支持 **KV cache**（`crossattn_cache`）——同一 prompt 各去噪步复用 K/V：
```python
q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
if crossattn_cache 已初始化: k, v = cache["k"], cache["v"]
else: k = self.norm_k(self.to_k(context)); v = self.to_v(context)   # 缓存
x = self.attn(q, k, v)
```

### WanI2VCrossAttention（L190-239）
context 前 257 个 token 是图像 CLIP 特征，其余是文本。分别对图像和文本做 attention 再相加：
```python
context_img = context[:, :257]      # 图像 CLIP（257=256 patch + 1 CLS）
context = context[:, 257:]          # 文本
img_x = self.attn(q, k_img, v_img)  # 图像分支
x = self.attn(q, k, v)              # 文本分支
x = x + img_x                       # 相加
```

## 6. 条件嵌入 WanTimeTextImageEmbedding（L54-99）

```python
temb = self.time_embedder(timestep)              # 正弦编码 + SiLU MLP → [B, inner_dim]
timestep_proj = self.time_modulation(temb)       # ModulateProjection(factor=6) → AdaLN 参数
encoder_hidden_states = self.text_embedder(encoder_hidden_states)   # MLP 4096→inner_dim
if image: encoder_hidden_states_image = self.image_embedder(...)    # WanImageEmbedding
```

## 7. VSA 变体 WanTransformerBlock_VSA（L407+）

当 `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN` 时用此 block。多一个 `to_gate_compress` 线性层（VSA 双分支融合权重），attn 用 `DistributedAttention_VSA`。见 [`../04_knowledge_expansion/06_sparse_attention.md`](../04_knowledge_expansion/06_sparse_attention.md)。

## 8. 张量形状全表（Wan2.1 T2V, 81帧 480×832）

| 位置 | 形状 |
|------|------|
| 输入 hidden_states | `[1, 16, 21, 60, 104]` |
| patchify + flatten | `[1, 32760, 1536]` |
| SP 切分（sp=2） | `[1, ~16380, 1536]` |
| timestep_proj | `[1, 6, 1536]` |
| encoder_hidden_states | `[1, 512, 1536]`（text_embedder 后） |
| 每个 block 输出 | 同输入 seq 形状 |
| proj_out | `[1, 32760, 64]` |
| unpatchify（输出） | `[1, 16, 21, 60, 104]` |

## 9. 阅读重点
1. forward 的 patchify → SP shard → blocks → unpatchify 主干。
2. block 的 Self→Cross→FFN + AdaLN 调制。
3. T2V/I2V cross attention 区别。
4. KV cache 优化。

## 10. 调试
在 forward 打印 `hidden_states.shape`（patchify 前后）、`encoder_hidden_states.shape`、`timestep_proj.shape`。
