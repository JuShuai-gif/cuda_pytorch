# DiT 精读 · Cosmos

> 源码：`/home/hpc/ghr_code/FastVideo/fastvideo/models/dits/cosmos.py`（743 行）
> 关键类：`CosmosTransformer3DModel`(L536)、`CosmosTransformerBlock`(L324)、`CosmosAdaLayerNormZero`(L116)、`CosmosSelfAttention`(L160)、`CosmosCrossAttention`(L249)、`CosmosRotaryPosEmbed`(L403)
>
> Cosmos（NVIDIA）用 AdaLN-Zero 调制 + GQA + condition mask，是 world model 类视频生成的代表。

## 1. 架构常量

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 2048 (=16×128) | Transformer 宽度 |
| `num_attention_heads` | 16 | 头数 |
| `attention_head_dim` | 128 | 每头维度 |
| `patch_size` | `(1, 2, 2)` | 时空 patch |
| `in_channels` | 17 (=16 latent + 1 condition_mask) | **已含 mask 通道** |
| `out_channels` | 16 | 输出通道 |
| `num_layers` | 28 | block 数 |
| `text_embed_dim` | 1024 | 文本维度 |
| `adaln_lora_dim` | 256 | AdaLN 低秩瓶颈 |
| `rope_scale` | `(1.0, 3.0, 3.0)` | RoPE NTK 缩放 |
| `concat_padding_mask` | True | padding mask 拼入通道 |

## 2. __init__ 子模块（L543-607）

```python
self.patch_embed = CosmosPatchEmbed(...)         # reshape+permute+Linear patchify
self.rope = CosmosRotaryPosEmbed(...)            # 3D RoPE 生成器
self.learnable_pos_embed = CosmosLearnablePositionalEmbed(...)  # 可选
self.time_embed = CosmosEmbedding(...)           # timestep → temb + embedded_timestep
self.transformer_blocks = ModuleList([CosmosTransformerBlock(...) × 28])
self.norm_out = CosmosAdaLayerNorm(2048, 256)    # 输出前 AdaLN
self.proj_out = nn.Linear(2048, 64)              # → 16*1*2*2
```

## 3. forward（L609-740）

输入：
```
hidden_states:         [B, 17, T, H, W]     17 = 16 latent + 1 condition_mask
encoder_hidden_states: [B, L, 1024]
timestep:              [B] 或 [B,1,T,1,1]（per-frame）
padding_mask:          [B, 1, T, H, W]
```

### Step 1：mask 拼接（L646-660）
```python
if condition_mask is not None:
    hidden_states = torch.cat([hidden_states, condition_mask], dim=1)
if self.concat_padding_mask:
    padding_mask = resize(padding_mask, [H, W])
    hidden_states = torch.cat([hidden_states, padding_mask...], dim=1)
# → [B, 18 或 19, T, H, W]
```
**condition_mask 语义**：标记哪些帧/区域是"已知条件"（如 I2V 的首帧、world model 的历史帧），哪些是"要生成"的。

### Step 2：RoPE + 可学习位置（L667-669）
```python
image_rotary_emb = self.rope(hidden_states, fps=fps)   # (cos, sin) 各 [THW, 128]
extra_pos_emb = self.learnable_pos_embed(hidden_states) # [B, THW, 2048] 可选
```

### Step 3：patchify（L672-678）
```python
hidden_states = self.patch_embed(hidden_states)   # [B, THW, 2048]
hidden_states = hidden_states.flatten(1, 3)
```

### Step 4：时间嵌入（L681-697）
```python
if timestep.ndim == 1:                             # [B] 全局
    temb, embedded_timestep = self.time_embed(hidden_states, timestep)
    # temb [B, 3*2048], embedded_timestep [B, 2048]
elif timestep.ndim == 5:                           # [B,1,T,1,1] per-frame
    # 扩展到每个 patch token → temb [B, THW, 3*2048]
```
**per-frame timestep**：每帧可有不同噪声水平，用于 world model 的自回归生成。

### Step 5：28 个 block（L700-722）
```python
for block in self.transformer_blocks:
    hidden_states = block(hidden_states, encoder_hidden_states, embedded_timestep, temb,
                          image_rotary_emb, extra_pos_emb, attention_mask)
```

### Step 6：输出（L725-733）
```python
hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)  # AdaLN
hidden_states = self.proj_out(hidden_states)                            # [B, THW, 64]
# unpatchify → [B, 16, T, H, W]
```

## 4. CosmosTransformerBlock（L324-400）

结构：**Self-Attn → Cross-Attn → FFN**，三者都用 **AdaLN-Zero** 调制。

```python
if extra_pos_emb is not None:
    hidden_states = hidden_states + extra_pos_emb

# Self-Attention with AdaLN-Zero
norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
hidden_states = hidden_states + gate * attn_output       # ← gate 残差

# Cross-Attention with AdaLN-Zero
norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, attention_mask)
hidden_states = hidden_states + gate * attn_output

# FFN with AdaLN-Zero
norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
ff_output = self.ff(norm_hidden_states)
hidden_states = hidden_states + gate * ff_output
```

## 5. AdaLN-Zero（CosmosAdaLayerNormZero, L116-157）

```python
def forward(self, hidden_states, embedded_timestep, temb):
    # embedded_timestep → SiLU → linear_1 → linear_2 → [B, S, 3*2048]
    modulation = ... + temb                # temb 作为 bias 加入
    shift, scale, gate = modulation.chunk(3)
    normed = LayerNorm(hidden_states) * (1 + scale) + shift
    return normed, gate
```
**"Zero" 的含义**：gate 初始化为 0，训练初期 `hidden_states + 0*attn = hidden_states`（恒等），训练更稳定。这是 DiT 论文的经典技巧。

## 6. GQA + RoPE（CosmosSelfAttention, L160-246）

```python
query = to_q(x); key = to_k(x); value = to_v(x)          # [B, S, 2048]
query = norm_q(query); key = norm_k(key)                  # QK-Norm(head_dim=128)
query = apply_rotary_emb(query, image_rotary_emb, use_real_unbind_dim=-2)  # RoPE
# GQA：kv 头少于 q 头时 repeat_interleave
key = key.repeat_interleave(q_heads // kv_heads, dim=...)
output = F.scaled_dot_product_attention(query, key, value)
```
- **GQA**（Grouped Query Attention）：K/V 头数 < Q 头数，省显存。
- **RoPE unbind_dim=-2**：与 Hunyuan 的 -1 不同，Cosmos 在倒数第二维拆实部/虚部。

## 7. NTK 缩放 RoPE（CosmosRotaryPosEmbed, L403-482）

`rope_scale=(1.0, 3.0, 3.0)` 做 NTK-aware 缩放，让模型泛化到训练时未见过的分辨率。head_dim=128 分配给时间/高/宽三轴。

## 8. Cosmos 2.5

`cosmos2_5.py`（967 行）是升级版，`Cosmos25DenoisingStage` 支持更灵活的条件化（T2W/V2W 自动路由）。`gen3c.py` 继承 Cosmos 2.5，加 3D 相机条件（`condition_video_pose`）。

## 9. 张量形状（Cosmos）

| 位置 | 形状 |
|------|------|
| hidden_states | `[1, 17, T, H, W]` |
| + mask concat | `[1, 18/19, T, H, W]` |
| patchify | `[1, THW, 2048]` |
| temb | `[1, 3*2048]` 或 `[1, THW, 3*2048]` |
| block 输出 | `[1, THW, 2048]` |
| 输出 | `[1, 16, T, H, W]` |

## 10. 与 Wan/Hunyuan 对比

| | Wan | Hunyuan | Cosmos |
|--|-----|---------|--------|
| 调制 | AdaLN（scale_shift_table） | 全局 vec | AdaLN-Zero（gate=0 初始化） |
| attention | Self+Cross 分开 | joint | Self+Cross 分开，GQA |
| 额外输入 | - | - | condition_mask + padding_mask |
| RoPE | 标准 | unbind=-1 | unbind=-2 + NTK 缩放 |
| 位置编码 | RoPE | RoPE | RoPE + 可学习 |

## 11. 阅读重点
1. condition_mask/padding_mask 如何拼进通道。
2. AdaLN-Zero 的 gate 残差（三个子层都用）。
3. per-frame timestep（world model 特性）。
4. GQA。

## 12. 调试
打印拼接后 `hidden_states.shape`（看通道数）、`timestep.ndim`（判断 per-frame）、`gate` 的值（训练初期应接近 0）。
