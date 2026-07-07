# DiT 精读 · LTX-2（Audio+Video 双模态）

> 源码：`/home/hpc/ghr_code/FastVideo/fastvideo/models/dits/ltx2.py`（3099 行，最大的 DiT）
> 关键类：`LTX2Transformer3DModel`(L2767)、`LTXModel`(L2342)、`BasicAVTransformerBlock`(L1775)、`AdaLayerNormSingle`(L241)、`VideoLatentPatchifier`(L422)、`AudioLatentPatchifier`(L497)
>
> LTX-2 是唯一同时生成**视频 + 音频**的 DiT。结构最复杂，本文抓主干。部分细节标注"待确认"。

## 1. 三层类结构

```
LTX2Transformer3DModel (L2767)   ← 外层：patchify/unpatchify + 调 LTXModel
  └─ LTXModel (L2342)             ← 中层：preprocessor + blocks + output
      └─ BasicAVTransformerBlock (L1775) × 48   ← 每层含 video/audio/cross-modal
```

## 2. 关键维度

| | Video | Audio |
|--|-------|-------|
| inner_dim | 4096 (=32×128) | 2048 (=32×64) |
| heads | 32 | 32 |
| head_dim | 128 | 64 |
| latent 通道 | 128 | 128 |
| cross_attention_dim | 4096 | 2048 |
| 文本维度 | 3840（T5） | 3840 |

## 3. LTX2Transformer3DModel.__init__（L2767-2846）

```python
self.model = LTXModel(model_type=AudioVideo, ...)     # 核心
self.patchifier = VideoLatentPatchifier(patch_size=32)
self.audio_patchifier = AudioLatentPatchifier(patch_size=16, sample_rate=16000, ...)
```

## 4. forward（L2897-3096）

输入：
```
hidden_states:         [B, 128, T_v, H, W]    Video latent
encoder_hidden_states: [B, 256, 3840]          文本
timestep:              [B, N]                   per-token 时间步
audio_hidden_states:   [B, 128, T_a, 16]        Audio latent（16 mel bins）
audio_encoder_hidden_states: [B, K, 3840]
```

### Video 路（L2935-2999）
```python
latents = self.patchifier.patchify(hidden_states)     # [B, 128,T,H,W] → [B, N_v, 128]
video_shape = VideoLatentShape.from_torch_shape(...)
latents, video_original_seq_len = sequence_model_parallel_shard(latents, dim=1)  # SP
positions = patchifier.get_patch_grid_bounds(video_shape)  # RoPE 3D 坐标（full seq）
positions = _get_pixel_coords(positions, scale_factors=(8,32,32), fps=fps)  # → 秒/像素坐标
# 构造 video_modality（Modality dataclass）
```

### Audio 路（L3001-3050）
```python
audio_latents = self.audio_patchifier.patchify(audio_hidden_states)  # [B,128,T_a,16]→[B,T_a,2048]
# SP shard + 1D 时间 RoPE positions [B, T_a, 1, 2]
```

### 调 LTXModel（L3052-3061）
```python
video_out, audio_out = self.model(video=video_modality, audio=audio_modality, ...)
```

### 后处理（L3063-3096）
```python
video_out = _to_denoised(video.latent, video_out, video.timesteps)  # sample - velocity*sigma
video_out = sequence_model_parallel_all_gather_with_unpad(video_out, ...)  # SP gather
video_out = self.patchifier.unpatchify(video_out, video_shape)      # [B,N,128]→[B,128,T,H,W]
audio_out = self.audio_patchifier.unpatchify(audio_out, audio_shape) # [B,T_a,2048]→[B,128,T_a,16]
```

输出：`video_out [B,128,T_v,H,W]` + `audio_out [B,128,T_a,16]`。

## 5. LTXModel.forward（L2684-2754）

```python
# 1. preprocessor.prepare：patchify_proj(128→4096) + adaln(timestep) + caption_proj(3840→4096) + RoPE
# 2. 48 层 BasicAVTransformerBlock 循环
# 3. output：AdaLN + norm_out + proj_out(4096→128)
```

## 6. BasicAVTransformerBlock（L1775-2290）—— 每层三部分

```
┌ Video 分支 ─────────────────────────┐
│ Self-Attn (AdaLN rows 0-2)          │
│ Text Cross-Attn (LTX2.3 有 gate)    │
└─────────────────────────────────────┘
┌ Audio 分支（镜像对称）───────────────┐
│ Self-Attn / Text Cross-Attn         │
└─────────────────────────────────────┘
┌ Cross-Modal ────────────────────────┐
│ A→V：video += a2v_attn(audio) * gate │
│ V→A：audio += v2a_attn(video) * gate │
└─────────────────────────────────────┘
┌ FFN（两路各自，AdaLN rows 3-5）──────┐
└─────────────────────────────────────┘
```

**cross-modal attention** 是 LTX-2 的核心创新：视频和音频在每层互相 attend，实现音画同步。SP 模式下需 `all_gather` 获取完整跨模态 context。

## 7. AdaLayerNormSingle（L241-258）—— PixArt-Alpha 风格

```python
def forward(self, timestep, hidden_dtype):
    embedded_timestep = self.emb(timestep)                       # 正弦 + MLP → [B, dim]
    modulation = self.linear(self.silu(embedded_timestep))        # [B, coefficient*dim]
    return modulation, embedded_timestep
```

modulation 行数（`embedding_coefficient`）：
- 基础 6 行：shift_msa/scale_msa/gate_msa（self-attn）+ shift_mlp/scale_mlp/gate_mlp（FFN）。
- LTX-2.3 额外 3 行：cross-attn 的 shift/scale/gate（`cross_attention_adaln=True`，coefficient=9）。

调制公式：
```python
norm_vx = RMSNorm(vx) * (1 + scale_msa) + shift_msa
vx = vx + attn(norm_vx) * gate_msa
```

## 8. Video vs Audio patchify

| | VideoLatentPatchifier (L422) | AudioLatentPatchifier (L497) |
|--|------------------------------|------------------------------|
| 输入 | `[B,128,T,H,W]` | `[B,128,T_a,16]` |
| 操作 | rearrange 空间 patch | rearrange mel+channel 展平 |
| 输出 | `[B, N_v, 128]` | `[B, T_a, 2048]` |
| token 含义 | 时空 patch | 每帧 1 token |
| RoPE | 3D（时+空） | 1D（时间，秒） |

> **待确认**：`VideoLatentPatchifier._patch_size=(1,32,32)` 与 `patchify_proj=Linear(128,4096)` 的匹配——推断 LTX-2 latent 分辨率已对齐 patch 边界，patchify 输出每 token 128 维。具体需结合 LTX-2 VAE 压缩比确认。

## 9. 高级特性

| 特性 | 位置 | 作用 |
|------|------|------|
| `cross_attention_adaln` | L2293 | LTX-2.3 对 text cross-attn 加 AdaLN（K/V 也调制） |
| `apply_gated_attention` | L1596 | per-head gating：`2*sigmoid(logit)` |
| `stg_block_idx` | L2049 | STG（时空引导）：特定 block 跳过 self-attn（CFG perturbed pass） |
| `use_distributed_attention` | L1772 | SP 时用 `LTXDistributedSelfAttention` |

**RoPE 与 SP**：RoPE 在 all-to-all **之后**应用（此时每 rank 有 full sequence + 部分 head），因为 RoPE 需要完整序列的位置信息。

## 10. 数据流全貌

```
video [B,128,T,H,W]          audio [B,128,T_a,16]
  patchify                     patchify
[B,N_v,128]                  [B,T_a,2048]
  SP shard                     SP shard
  ┌─────── LTXModel ───────────────────┐
  │ patchify_proj: 128→4096 / →2048     │
  │ adaln(timestep) → 6/9 行调制         │
  │ caption_proj: 3840→4096              │
  │ 48× BasicAVTransformerBlock:         │
  │   Video: self→text-cross             │
  │   Audio: self→text-cross             │
  │   Cross-modal: A→V, V→A              │
  │   Video FFN / Audio FFN              │
  │ norm_out + proj_out: →128            │
  └──────────────────────────────────────┘
  _to_denoised → SP gather → unpatchify
[B,128,T,H,W]                [B,128,T_a,16]
```

## 11. 阅读重点
1. video/audio 双路 + cross-modal attention。
2. AdaLayerNormSingle 的 6/9 行调制含义。
3. patchify（video 3D vs audio 1D）。
4. `_to_denoised`（velocity → sample）。

## 12. 调试
LTX-2 复杂，建议先用 video-only 模式（`LTXModelType.VideoOnly`）理解主干，再看 audio + cross-modal。打印 `video_out.shape` / `audio_out.shape`。
