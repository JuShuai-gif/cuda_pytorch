# VAE 精读 · encode/decode 内部结构

> 源码：`fastvideo/models/vaes/`（common.py 基类 + 各模型 VAE）
> 关键类：`ParallelTiledVAE`(common.py:17)、`DiagonalGaussianDistribution`(common.py:476)、`AutoencoderKLWan`(wanvae.py:1103)
>
> VAE 把像素视频压成 latent（DiT 的工作空间），最后再 decode 回像素。本文拆解 encode/decode 内部与 tiling。

## 1. 基类 ParallelTiledVAE（common.py:17）

所有视频 VAE 继承它，统一 encode/decode API + tiling 逻辑。

```python
class ParallelTiledVAE(ABC):
    @property
    def temporal_compression_ratio(self): return config.temporal_compression_ratio  # 4
    @property
    def spatial_compression_ratio(self): return config.spatial_compression_ratio    # 8
    @property
    def scaling_factor(self): return config.scaling_factor                          # 归一化因子

    @abstractmethod
    def _encode(self, x): ...   # 子类实现真正的编码
    @abstractmethod
    def _decode(self, z): ...   # 子类实现真正的解码
```

## 2. encode（common.py:65-77）

```python
def encode(self, x):   # x: [B, C, T, H, W] 像素 [-1,1]
    latent_num_frames = (num_frames - 1) // temporal_compression_ratio + 1
    if use_temporal_tiling and num_frames > tile_min:
        latents = self.tiled_encode(x)          # 时间维分块
    elif use_tiling and (W>tile or H>tile):
        latents = self.spatial_tiled_encode(x)  # 空间维分块
    else:
        latents = self._encode(x)               # 直接编码
    return DiagonalGaussianDistribution(latents)
```

- 输入 `[B, 3, 81, 720, 1280]` → 输出分布，`sample()`/`mode()` → `[B, 16, 21, 90, 160]`。
- 时间 `(81-1)/4+1=21`，空间 `720/8=90, 1280/8=160`。

## 3. decode（common.py:79-）

```python
def decode(self, z):   # z: [B, 16, T', H', W'] latent
    num_sample_frames = (num_frames - 1) * temporal_compression_ratio + 1
    if use_parallel_tiling and sp_world_size > 1:
        return self.parallel_tiled_decode(z)     # 多 GPU 并行
    if use_temporal_tiling and num_frames > tile:
        return self.tiled_decode(z)              # 时间分块
    elif use_tiling and (...):
        return self.spatial_tiled_decode(z)      # 空间分块
    else:
        return self._decode(z)                   # 直接解码
```

- 输入 `[B, 16, 21, 90, 160]` → 输出 `[B, 3, 81, 720, 1280]`。
- 时间 `(21-1)*4+1=81`，空间 `90*8=720, 160*8=1280`。

## 4. DiagonalGaussianDistribution（common.py:476）

VAE encoder 输出 mean + logvar（拼在通道维），构成对角高斯分布：
```python
class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)   # [B, 2*z, ...] → 各 [B, z, ...]
        self.std = exp(0.5 * logvar)
    def sample(self, generator): return self.mean + self.std * randn(...)   # 采样
    def mode(self): return self.mean                                        # 取众数（推理常用）
```

推理时 I2V/V2V 编码条件多用 `mode()`（确定性）；训练加噪多用 `sample()`。

## 5. Tiling（省显存核心）

VAE decode 是显存峰值点（要在像素空间重建大视频）。tiling 分块处理：

| 方法 | 切分维度 | 用途 |
|------|---------|------|
| `tiled_decode` | 时间 T | 长视频 |
| `spatial_tiled_decode` | 空间 H/W | 高分辨率 |
| `parallel_tiled_decode` | 多 GPU | SP 场景 |

块之间有重叠区，用 `blend_v`（垂直）/`blend_h`（水平）/`blend_t`（时间）线性混合，避免拼接痕迹：
```python
# 相邻块重叠区做加权平均，边界平滑过渡
```

配置在 `VAEConfig`：`use_tiling`, `tile_sample_min_height/width/num_frames`, `tile_sample_stride_*`, `blend_num_frames`。

## 6. AutoencoderKLWan 内部（wanvae.py:1103）

```python
self.encoder = WanEncoder3d(in_channels=3, dim, z_dim*2, dim_mult, ...)
self.decoder = WanDecoder3d(dim, z_dim=16, dim_mult, ...)
self.quant_conv = WanCausalConv3d(z_dim*2, z_dim*2, 1)
self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

def _encode(self, x):
    out = self.encoder(x)               # 卷积下采样
    enc = self.quant_conv(out)          # → [B, 2*z, T', H', W']（mean+logvar）
    return enc

def _decode(self, z):
    x = self.post_quant_conv(z)
    return self.decoder(x)              # 卷积上采样 → clamp[-1,1]
```

### Encoder/Decoder 结构
```
Encoder: conv_in → down_blocks(ResNet+Attention, downsample3d/2d) → mid_block → norm_out → conv_out
Decoder: conv_in → mid_block → up_blocks(ResNet+Attention, upsample3d/2d) → norm_out → conv_out
```

### 因果卷积 WanCausalConv3d
时间方向只向后看（padding 只在过去侧），支持流式/自回归生成 + feature cache（跨帧缓存中间特征，避免重复计算）。

## 7. latent 归一化

DiT 在归一化的 latent 空间工作。VAE encode 后乘 `scaling_factor`，decode 前逆操作：
```python
# encode 后（image_encoding.py）
latent = vae.encode(x) * vae.scaling_factor            # Wan: 0.476986
# decode 前（decoding.py:_denormalize_latents）
latents = latents / scaling_factor + shift_factor       # 或 latents * std + mean
```
不同 VAE 用不同统计量（存 config），目的是让 latent 分布接近标准正态，扩散更稳定。

## 8. 各 VAE 规格

| VAE | 文件 | z 通道 | 时间压缩 | 空间压缩 | 特点 |
|-----|------|--------|---------|---------|------|
| `AutoencoderKLWan` | wanvae.py (1379行) | 16 | ×4 | ×8 | 因果卷积, feature cache |
| `AutoencoderKLHunyuanVideo` | hunyuanvae.py (852行) | 16 | ×4 | ×8 | 因果卷积+自注意力 |
| `AutoencoderKLHunyuanVideo15` | hunyuan15vae.py (703行) | 32 | ×4 | ×16 | 非 KL VAE |
| `CausalVideoAutoencoder` | ltx2vae.py (1849行) | 128 | - | - | LTX-2 |
| `AutoencoderKLGen3CTokenizer` | gen3c_tokenizer_vae.py | - | - | - | GEN3C |
| `AutoencoderKLFlux2` | flux2vae.py (533行) | - | - | ×8 | Flux2 |
| Oobleck | oobleck.py (376行) | - | - | - | Stable Audio |

## 9. 音频 VAE（LTX-2 / Stable Audio）

- `audio/ltx2_audio_vae.py`（1955行）：mel-spectrogram ↔ audio latent + vocoder（→ waveform）。
- `oobleck.py`：Stable Audio 的音频 autoencoder。

## 10. 阅读重点
1. `common.py` 的 encode/decode 自动 tiling 分派。
2. `DiagonalGaussianDistribution` 的 sample vs mode。
3. `WanCausalConv3d` 的因果性 + feature cache。
4. 归一化/反归一化。

## 11. 调试
在 `decode` 打印输入 latent 和输出像素形状。OOM 时确认 `use_tiling=True`。观察 `scaling_factor` 是否正确应用（错了会导致黑屏/过曝）。

## 12. 相关
- VAE decode 流程：[`../03_core_flows/06_vae_decode_flow.md`](../03_core_flows/06_vae_decode_flow.md)
- VAE 知识：[`../04_knowledge_expansion/03_vae_for_video.md`](../04_knowledge_expansion/03_vae_for_video.md)
