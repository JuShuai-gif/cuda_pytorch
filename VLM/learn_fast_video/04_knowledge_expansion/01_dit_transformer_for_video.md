# 视频 DiT / Transformer

> 知识点扩展：DiT 是什么、视频 token 组织、temporal/spatial attention、3D patch embedding、timestep embedding、cross attention，回扣 FastVideo 各 DiT 实现。

## 1. DiT 是什么

DiT（Diffusion Transformer）用 Transformer 替代 U-Net 作为扩散模型的骨干网络。相比 U-Net，DiT 更易 scale、更适合长序列（视频）。

FastVideo 所有视频模型都是 DiT，基类 `BaseDiT`（`models/dits/base.py`）。

### 1.1 DiT vs U-Net

| | U-Net（老一代） | DiT（新一代） |
|--|----------------|--------------|
| 结构 | 卷积 + 下/上采样 + skip | 纯 Transformer block |
| 条件注入 | 卷积 + cross-attn | AdaLN + cross-attn / joint attn |
| scale 能力 | 有瓶颈 | 好（同 LLM scaling law） |
| 长序列 | 局部感受野 | 全局 attention |
| 代表 | SD1.5/SDXL | SD3/Flux/Wan/Hunyuan/Cosmos |

视频序列长、需全局时空建模、要 scale 到大参数量，所以视频生成几乎全用 DiT。

### 1.2 DiT 的通用骨架

任何 DiT（无论哪个模型）都是这套：
```
latent [B,C,T,H,W]
  → patchify（切 token）
  → + timestep embedding（调制）
  → + text/image/... 条件（cross-attn 或 joint attn）
  → N × TransformerBlock（attention + FFN + 归一化 + 调制）
  → unpatchify
  → 预测的噪声/速度 [B,C,T,H,W]
```
差异只在：调制方式、attention 组织、额外条件。抓住骨架，任何模型都能快速读懂。

## 2. 视频 token 如何组织

视频 latent `[B, C, T, H, W]` 通过 3D patch embedding 切成 token：
```
patch_size = (p_t, p_h, p_w)  # 如 (1, 2, 2)
token 数 L = (T/p_t) × (H/p_h) × (W/p_w)
```
Wan（81帧, 480×832, latent 21×60×104, patch 1×2×2）：
```
L = 21 × 30 × 52 = 32760 个 token
```
每个 token 是一个高维向量（inner_dim）。序列长度 32760 极大，是需要序列并行 + 稀疏 attention 的原因。

### 2.1 时空 attention 的三种组织

| 方式 | 做法 | 代价 | 用于 |
|------|------|------|------|
| **Full 3D** | 所有 T×H×W token 互相 attend | O(L²)，最贵 | Wan/Hunyuan（配稀疏加速） |
| **Factorized** | 分别做 spatial（同帧）+ temporal（同位置跨帧） | 便宜，但表达弱 | 部分老模型 |
| **Sparse** | 只 attend 时空邻域/top-k block | ~O(L·k) | VSA/STA |

FastVideo 主流用 full 3D + VSA 稀疏加速，兼顾表达力和速度。

## 3. 3D Patch Embedding

```
源码：models/dits/wanvideo.py 的 PatchEmbed
```
用 3D 卷积（Conv3d）把 latent 切成 patch 并投影到 hidden 维：
```python
self.patch_embedding = PatchEmbed(in_chans=16, embed_dim=inner_dim, patch_size=(1,2,2))
# [B,16,T,H,W] → [B, T',H',W', inner_dim] → flatten → [B, L, inner_dim]
```

**为什么时间 patch 常是 1**：视频帧间相关性强但不像空间那样冗余，时间下采样已由 VAE 做（4×），DiT 里再压时间会丢动态细节，所以 `p_t=1`（不压时间）、`p_h=p_w=2`（压空间）是常见选择。

### 3.1 RoPE（旋转位置编码）

Transformer 本身无位置概念，需位置编码。视频 DiT 用 **3D RoPE**：把 head_dim 分配给时间/高/宽三个轴，每个轴按其位置旋转 Q/K 向量。
```python
# wanvideo.py:657
rope_dim_list = [d - 4*(d//6), 2*(d//6), 2*(d//6)]   # 时/高/宽三轴分配
```
优点（相比绝对位置编码）：
- 相对位置感知（attention 分数只依赖位置差）。
- 可外推到训练未见的分辨率（Cosmos 还加 NTK 缩放增强外推）。

各模型 RoPE 细节略有不同（Hunyuan `unbind=-1`，Cosmos `unbind=-2`），见各 DiT 精读。

## 4. Timestep Embedding

timestep 是标量，需嵌入成向量注入网络：
```
正弦编码(t) → SiLU MLP → temb [B, inner_dim]
temb → 调制参数（AdaLN 的 shift/scale/gate）
```
FastVideo 各模型用不同调制方式（见下表）。

### 4.1 AdaLN 调制家族

timestep（和其他全局条件）通过调制 LayerNorm 注入，这比直接 cross-attn 更高效。变体：

| 变体 | 公式 | 用于 |
|------|------|------|
| **AdaLN** | `LN(x)·(1+scale)+shift` | Wan |
| **AdaLN-Zero** | 额外 `+ gate·sublayer(x)`，gate 初始化 0 | Cosmos, DiT 原论文 |
| **AdaLN-Single** | 全局共享一组调制（PixArt） | LTX-2 |
| **全局 vec** | time+pooled_text+guidance 相加 | Hunyuan |

`modulate(x, shift, scale) = x·(1+scale)+shift`（`models/utils.py:118`）是这一切的基础操作。AdaLN-Zero 的 gate=0 初始化让训练初期是恒等映射，是稳定训练的关键技巧。

**简单代码示例（教学用，patchify + AdaLN 最小 DiT 块）**：
```python
import torch, torch.nn as nn, math

# --- 3D patchify：latent → 平展的 token 序列 ---
def patchify(x, p_t=1, p_h=2, p_w=2):
    # x: [B, C, T, H, W] → [B, (T/p_t)*(H/p_h)*(W/p_w), C*p_t*p_h*p_w]
    B, C, T, H, W = x.shape
    Tp, Hp, Wp = T // p_t, H // p_h, W // p_w
    x = x.reshape(B, C, Tp, p_t, Hp, p_h, Wp, p_w)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, Tp * Hp * Wp, C * p_t * p_h * p_w)
    return x

# --- 正弦 timestep embedding ---
def sinusoidal_embedding(t, dim):
    # t: [B, 1], dim: 输出维度的一半（频率数）
    freq = torch.exp(-math.log(10000) * torch.arange(0, dim // 2).float() / (dim // 2))
    freq = freq.to(t.device)
    emb = t.float() * freq                                         # [B, dim//2]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)               # [B, dim]

# --- 最小 DiT block：Self-Attn + AdaLN ---
class ToyDiTBlock(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)     # 无自带 affine，由 AdaLN 代替
        self.qkv = nn.Linear(dim, 3 * dim)                           # 融合 QKV
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))   # 6 组调制：attn(shift/scale/gate) + mlp(shift/scale/gate)
    def forward(self, x, t_emb):
        # t_emb: [B, dim]（从 timestep 编码来）
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(t_emb).chunk(6, dim=-1)
        # --- Self-Attention with AdaLN ---
        h = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)   # AdaLN
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(t.shape[0], t.shape[1], -1, self.head_dim).transpose(1, 2) for t in [q, k, v]]
        attn_out = self.proj(torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
        x = x + gate_sa.unsqueeze(1) * attn_out                            # gate 残差
        # --- FFN with AdaLN ---
        h = self.norm1(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x
```
对应关系：
- `patchify` ↔ `wanvideo.py:669-670`（Wan 的 `PatchEmbed` + flatten）。
- `sinusoidal_embedding` ↔ `TimestepEmbedder` / `CosmosEmbedding`。
- `AdaLN 公式` ↔ `modulate(x, shift, scale)` / `norm * (1+scale) + shift`。
- 真实 DiT 还增加了 text cross-attention + RoPE + SP 切分 + unpatchify。

## 5. Text Condition / Cross Attention

文本条件通过 cross attention 注入：
- Query 来自视频 token。
- Key/Value 来自文本 embedding（`encoder_hidden_states`）。

```python
# Wan: WanT2VCrossAttention（wanvideo.py L153）
# Q ← 视频 latent, K/V ← text embedding
# 支持 KV cache（同 prompt 各步复用 K/V）
```

## 6. Temporal vs Spatial Attention

视频 DiT 的 attention 有几种组织方式：
- **Full 3D attention**：所有时空 token 互相 attend（Wan/Hunyuan，计算量最大）。
- **Factorized**：分别做 spatial（同帧内）和 temporal（跨帧）attention（省算力）。
- **Sparse（VSA/STA）**：只 attend 时空邻域或 top-k block。

FastVideo 主流模型用 full 3D + 稀疏加速（VSA）。

## 7. 各 DiT 结构对比

| 模型 | Block 结构 | 调制 | 位置编码 | 特殊 |
|------|-----------|------|---------|------|
| **Wan** (`wanvideo.py` L561) | Self→Cross→FFN | scale_shift_table + temb (AdaLN) | 3D RoPE | QK-Norm, KV cache |
| **Hunyuan** (`hunyuanvideo.py` L408) | Double→Single (MMDiT) | ModulateProjection (6参) | 3D RoPE | joint img+txt attention |
| **Cosmos** (`cosmos.py` L536) | Self→Cross→FFN | AdaLN-Zero (含gate) | 3D RoPE+Learnable | GQA, mask concat |
| **LTX-2** (`ltx2.py` L2757) | Self→Cross→FFN | AdaLN-Single (PixArt) | RoPE | Audio+Video 双模态 |

### MMDiT（Hunyuan）
Double-stream（图像和文本各自 self-attn+FFN，在 joint attention 融合）+ Single-stream（拼接后统一处理）。这是 SD3/Flux 家族的架构。

### AdaLN 调制
```python
# models/utils.py:modulate
x = x * (1 + scale) + shift    # timestep 控制归一化后的缩放/偏移
# AdaLN-Zero 额外有 gate: x = x + gate * attn_output
```

## 8. Wan DiT forward 流程（wanvideo.py L632）

```
hidden_states [B,16,T,H,W]
1. patch_embedding → flatten [B, L, inner_dim]
2. sequence_model_parallel_shard（SP 序列切分）
3. condition_embedder(timestep, text, image) → temb, timestep_proj, text_emb
4. for block in 40 blocks:
     Self-Attention(RoPE, QK-Norm) → Cross-Attention(text) → FFN
     每个子层前 AdaLN 调制（scale_shift_table + temb）
5. norm_out + proj_out + unpatchify → [B,16,T,H,W]（预测速度）
```

## 9. 关键张量含义

| 变量 | 形状 | 含义 |
|------|------|------|
| `hidden_states` | `[B,16,T,H,W]` → `[B,L,inner_dim]` | 视频 latent / token |
| `encoder_hidden_states` | `[B,512,4096]` | 文本条件（cross attn 的 K/V） |
| `timestep` | `[B]` | 噪声水平 |
| `timestep_proj` | `[B,6,inner_dim]` | AdaLN 调制参数 |
| output | `[B,16,T,H,W]` | 预测的速度/噪声 |

## 10. 回扣源码
- 完整 DiT：`models/dits/wanvideo.py`。
- patch embed / RoPE：`layers/visual_embedding.py`, `layers/rotary_embedding_3d.py`。
- AdaLN：`models/utils.py:modulate`, `layers/layernorm.py:ScaleResidual`。

## 11. 延伸
- attention 加速：[`05_attention_acceleration.md`](05_attention_acceleration.md)
- 序列并行：[`07_sequence_parallelism.md`](07_sequence_parallelism.md)
