# 视频 VAE

> 知识点扩展：VAE encode/decode、latent space、为什么视频扩散在 latent 上做，回扣 FastVideo VAE 实现。

## 1. VAE 的作用

VAE（Variational Autoencoder）把高维像素压缩成低维 latent，扩散在 latent 空间进行，最后 decode 回像素。

### 1.1 VAE vs 普通 Autoencoder

普通 AE 只学"压缩-重建"，latent 分布杂乱。VAE 额外约束 latent 接近标准正态分布（KL 散度正则），好处：
- latent 空间平滑连续，适合扩散在上面加噪去噪。
- 采样有意义（从正态采样能 decode 出合理图像）。

视频 VAE 多是 **VAE 的变体**（有的甚至去掉 KL，如 HunyuanVideo15 的非 KL VAE），核心目的都是提供一个"扩散友好"的压缩空间。

## 2. 为什么在 latent 上做扩散

原始视频 `[1,3,81,480,832]` ≈ 9700 万值。直接扩散：
- 显存爆炸。
- attention 序列长度不可接受。

VAE 压缩到 `[1,16,21,60,104]` ≈ 200 万值（~48 倍压缩），大幅降低算力和显存。这是 Latent Diffusion 的核心优势。

### 2.1 压缩比的权衡

- **压缩比越大**：latent 越小，扩散越快越省显存，但重建质量下降（细节丢失）、VAE 更难训。
- **压缩比越小**：质量好但扩散负担重。

视频 VAE 通常时间 ×4、空间 ×8（Wan/Hunyuan），HunyuanVideo15 激进到空间 ×16。时间压缩比空间保守，因为时间维信息密度高（帧间运动细节）。图像 VAE 无时间维，只空间 ×8。

## 3. encode / decode

```
源码位置：models/vaes/common.py（ParallelTiledVAE, DiagonalGaussianDistribution）
```
```python
# encode: 像素 → latent 分布
dist = vae.encode(x)             # [B,3,T,H,W] → DiagonalGaussianDistribution
z = dist.sample()                # 或 dist.mode()   → [B,16,T',H',W']
# decode: latent → 像素
image = vae.decode(z)            # [B,16,T',H',W'] → [B,3,T,H,W]
```

`DiagonalGaussianDistribution`（common.py L476）：VAE encoder 输出 mean + logvar，`sample()` 从对角高斯采样。

## 4. 压缩比

| VAE | latent 通道 | 时间压缩 | 空间压缩 |
|-----|-----------|---------|---------|
| `AutoencoderKLWan` | 16 | ×4 | ×8 |
| `AutoencoderKLHunyuanVideo` | 16 | ×4 | ×8 |
| `AutoencoderKLHunyuanVideo15` | 32 | ×4 | ×16 |

时间压缩公式：latent 帧数 = `(num_frames-1)/时间压缩 + 1`。第一帧单独处理（因果）。

**简单代码示例（教学用，形状换算 + encode/decode 概念）**：
```python
import torch

# 压缩比换算（Wan：时间 ×4，空间 ×8，通道 3→16）
def pixel_to_latent_shape(num_frames, H, W, t_ratio=4, s_ratio=8, z=16):
    lat_T = (num_frames - 1) // t_ratio + 1
    return (z, lat_T, H // s_ratio, W // s_ratio)

print(pixel_to_latent_shape(81, 480, 832))   # (16, 21, 60, 104)

# encode/decode 概念（真实 vae 来自 pipeline.get_module("vae")）
def vae_roundtrip_demo(vae, video):           # video: [B, 3, T, H, W] ∈ [-1,1]
    dist = vae.encode(video)                  # → DiagonalGaussianDistribution
    z = dist.mode()                           # 取均值（确定性）；训练常用 dist.sample()
    z = z * vae.scaling_factor                # 归一化（让 latent 接近标准正态）
    # ... DiT 在 z 上做扩散 ...
    z = z / vae.scaling_factor                # 反归一化
    recon = vae.decode(z)                     # → [B, 3, T, H, W] ∈ [-1,1]
    recon = (recon / 2 + 0.5).clamp(0, 1)     # → [0,1] 用于保存
    return recon
```
`scaling_factor` 用错会导致黑屏/过曝——这是最常见的 VAE 调试陷阱。

## 5. 因果 3D 卷积

视频 VAE 用因果卷积（`WanCausalConv3d`），时间方向只看过去帧，支持流式/自回归生成。配合 feature cache 做跨帧连续解码。

## 6. Tiling（省显存）

```
源码位置：common.py 的 ParallelTiledVAE
```
VAE decode 是显存峰值点。tiling 把大视频分块处理：
- `tiled_decode`：时间维分块。
- `spatial_tiled_decode`：空间维分块。
- `parallel_tiled_decode`：多 GPU 并行。
- 块间 `blend_v/blend_h/blend_t` 混合重叠区避免拼接痕迹。

配置在 `VAEConfig`（`use_tiling`, `tile_sample_min_*`, `tile_sample_stride_*`）。

## 7. latent 归一化

DiT 训练/推理时 latent 需归一化（接近标准正态），decode 前反归一化：
```python
# stages/decoding.py:_denormalize_latents
z = z * latents_std + latents_mean            # 方式1
z = z / scaling_factor + shift_factor          # 方式2
```
统计量存在 VAE config（`latents_mean`, `latents_std`, `scaling_factor`）。

## 8. Wan VAE 结构

```
Encoder: conv_in → down_blocks(ResNet+Attention, downsample3d/2d) → mid_block → norm_out → conv_out
Decoder: conv_in → mid_block → up_blocks(ResNet+Attention, upsample3d/2d) → norm_out → conv_out
```

## 9. 回扣源码
| 概念 | 源码 |
|------|------|
| encode/decode API | `models/vaes/common.py` |
| Wan VAE | `models/vaes/wanvae.py:AutoencoderKLWan` |
| decode 调用 | `pipelines/stages/decoding.py` |
| 反归一化 | `stages/decoding.py:_denormalize_latents` |
| VAE 配置 | `configs/models/vaes/base.py` |

## 10. 延伸
- VAE decode 流程：[`../03_core_flows/06_vae_decode_flow.md`](../03_core_flows/06_vae_decode_flow.md)
- 显存优化：[`13_memory_optimization.md`](13_memory_optimization.md)
