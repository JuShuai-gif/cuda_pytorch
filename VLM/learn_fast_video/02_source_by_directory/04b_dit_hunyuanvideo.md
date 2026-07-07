# DiT 精读 · HunyuanVideo（MMDiT）

> 源码：`/home/hpc/ghr_code/FastVideo/fastvideo/models/dits/hunyuanvideo.py`（834 行）
> 关键类：`HunyuanVideoTransformer3DModel`(L431)、`MMDoubleStreamBlock`(L81)、`MMSingleStreamBlock`(L283)、`FinalLayer`(L791)
>
> HunyuanVideo 是 MMDiT（Multi-Modal DiT）架构，与 SD3/Flux 同源。核心是 double-stream + single-stream 混合。

## 1. 架构常量

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 3072 (=24×128) | Transformer 宽度 |
| `num_attention_heads` | 24 | 头数 |
| `attention_head_dim` | 128 | 每头维度 |
| `patch_size` | `[1, 2, 2]` | 时空 patch |
| `in/out_channels` | 16/16 | VAE latent 通道 |
| `mlp_ratio` | 4.0 | MLP hidden = 12288 |
| `num_layers` | 20 | double-stream block 数 |
| `num_single_layers` | 40 | single-stream block 数 |
| `num_refiner_layers` | 2 | text refiner 层 |
| `rope_axes_dim` | `(16, 56, 56)` | 3D RoPE 三轴（和=128） |
| `text_embed_dim` | 4096 | 文本维度 |
| `pooled_projection_dim` | 768 | 全局池化文本维度 |

## 2. __init__ 子模块（L431-525）

```python
self.img_in = PatchEmbed(...)              # Conv3d patchify [B,16,T,H,W]→[B,THW,3072]
self.txt_in = SingleTokenRefiner(4096, 3072, depth=2)   # 文本精炼（2 层）
self.time_in = TimestepEmbedder(3072)      # timestep → [B,3072]
self.vector_in = MLP(768, 3072, 3072)      # 全局池化文本 → [B,3072]
self.guidance_in = TimestepEmbedder(3072)  # CFG guidance（可选）
self.double_blocks = ModuleList([MMDoubleStreamBlock(...) × 20])
self.single_blocks = ModuleList([MMSingleStreamBlock(...) × 40])
self.final_layer = FinalLayer(3072, patch_size, 16)
```

## 3. forward（L529-625）

输入：
```
hidden_states:         [B, 16, T, H, W]      视频 latent
encoder_hidden_states: [B, L, 4096]           文本（含全局 token）
timestep:              [B]
```

### Step 1：拆分文本（L558-563）
```python
txt = encoder_hidden_states[:, 1:]                    # [B, L-1, 4096] 逐 token
text_states_2 = encoder_hidden_states[:, 0, :768]      # [B, 768] 全局池化
```

### Step 2：3D RoPE（L574-578）
```python
freqs_cos, freqs_sin = get_rotary_pos_embed((T, H//2, W//2), 3072, 24, [16,56,56], theta=256)
# 各 [THW, 128]
```

### Step 3：调制向量 vec（L580-587）
```python
vec = self.time_in(t) + self.vector_in(text_states_2)   # [B, 3072]（+ guidance 可选）
```
**关键**：Hunyuan 用**全局** vec（time + pooled_text + guidance），不是 per-token。

### Step 4：patchify + 文本精炼（L589-593）
```python
img = self.img_in(img)                                  # [B, THW, 3072]
img, original_seq_len = sequence_model_parallel_shard(img, dim=1)  # SP 切分
txt = self.txt_in(txt, t)                               # [B, L-1, 3072]
```

### Step 5：double-stream blocks（L598-600）
```python
for block in self.double_blocks:    # 20 个
    img, txt = block(img, txt, vec, freqs_cis, original_seq_len)
```

### Step 6：拼接 + single-stream blocks（L602-614）
```python
x = torch.cat((img, txt), 1)        # [B, S_img+S_txt, 3072]
for block in self.single_blocks:    # 40 个
    x = block(x, vec, txt_seq_len, freqs_cis, original_seq_len)
```

### Step 7：提取 img + 输出（L617-625）
```python
img = x[:, :img_seq_len]
img = sequence_model_parallel_all_gather_with_unpad(img, original_seq_len, dim=1)
img = self.final_layer(img, vec)    # [B, THW, 64]
img = unpatchify(img, ...)          # [B, 16, T, H, W]
```

## 4. MMDoubleStreamBlock（L81-280）—— 双流

img 和 txt **各自独立**调制/QKV/FFN，但在 attention 中 **joint 计算**。

```python
# 调制：img_mod(vec)→6组, txt_mod(vec)→6组
# img 分支
img_attn_input = self.img_attn_norm(img, img_attn_shift, img_attn_scale)   # AdaLN
img_qkv = self.img_attn_qkv(img_attn_input)          # [B, S_img, 9216]
img_q, img_k, img_v = split → [B, S_img, 24, 128]
img_q = self.img_attn_q_norm(img_q); img_k = ...     # QK-Norm

# txt 分支（对称）
txt_q, txt_k, txt_v = ... [B, S_txt, 24, 128]

# ★ joint attention：img 和 txt 的 QKV 一起送入
img_attn, txt_attn = self.attn(img_q, img_k, img_v, original_seq_len,
                               txt_q, txt_k, txt_v, freqs_cis=freqs_cis)

# img FFN + gate 残差 / txt FFN + gate 残差
```

**joint attention 机制**（`attention/layer.py:DistributedAttention`）：
- img 的 K/V 和 txt 的 K/V 在序列维拼接 `[img_tokens | txt_tokens]`。
- 于是 img query 能 attend 到所有 txt token，txt 也能 attend 到 img——这是 MMDiT 用一个 attention 同时做 self + cross 的精髓。

## 5. MMSingleStreamBlock（L283-405）—— 单流

img 和 txt 合并成一个序列，共享调制。QKV 和 MLP 在 `linear1` 融合：

```python
mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3)   # 共享调制
x_mod = self.input_norm_scale_shift(x, mod_shift, mod_scale)
linear1_out = self.linear1(x_mod)                    # [B, S, 21404]
qkv, mlp = split([9216, 12288])                       # QKV + MLP 输入
q, k, v = ... [B, S, 24, 128] → QK-Norm
img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]       # 靠 txt_len 拆分（txt 在末尾）
img_attn, txt_attn = self.attn(img_q,..., txt_q,...)  # joint attention
attn_output = cat(img_attn, txt_attn)
mlp_output = self.mlp_act(mlp)                        # GELU
output = self.linear2(cat(attn_output, mlp_output))   # 融合输出
return self.output_residual(x, output, mod_gate)      # gate 残差
```

## 6. Double vs Single 对比

| | Double (20层) | Single (40层) |
|--|--------------|---------------|
| img/txt 调制 | 独立 | 共享 |
| QKV 投影 | 独立 | 融合在 linear1 |
| FFN | 独立 | 与 attn 融合处理 |
| attention | joint | joint |

设计意图：early 层用 double 让两模态充分独立发展，late 层用 single 高效融合。

## 7. 调制方式：AdaLN（factor 6/3）

`ModulateProjection(factor=6)`（double）产生 shift/scale/gate × 2（attn + mlp）；`factor=3`（single）产生 shift/scale/gate × 1。QK-Norm 用自实现的 `HunyuanRMSNorm`。

## 8. 张量形状（Hunyuan T2V）

| 位置 | 形状 |
|------|------|
| hidden_states | `[1, 16, T, H, W]` |
| img（patchify） | `[1, S_img, 3072]`，S_img=T×(H/2)×(W/2) |
| txt（refine） | `[1, S_txt, 3072]` |
| vec | `[1, 3072]`（全局） |
| double 输出 | img/txt 各自不变 |
| single 输入 | `[1, S_img+S_txt, 3072]` |
| final | `[1, THW, 64]` → unpatchify → `[1, 16, T, H, W]` |

## 9. 与 Wan 对比

| | Wan | Hunyuan |
|--|-----|---------|
| 架构 | Self→Cross→FFN 单流 | MMDiT double+single |
| cross attention | 独立 cross-attn 层 | joint attention（无独立 cross） |
| 调制 | per-token timestep_proj | 全局 vec |
| block 数 | 30/40 | 20+40=60 |

## 10. 阅读重点
1. double/single 两种 block 的区别。
2. joint attention 如何用一个 attention 同时做 self+cross。
3. 全局 vec 的构成（time+pooled_text+guidance）。

## 11. 调试
打印 `img.shape`、`txt.shape`（进 double 前）、`x.shape`（single 阶段拼接后）。
