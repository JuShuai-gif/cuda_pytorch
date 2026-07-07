# models —— 模型层

> 模块作用：DiT / VAE / TextEncoder / Scheduler 的具体实现 + 注册 + 加载。
>
> **本文是总览。各模型的层级精读见**：
> - [`04a_dit_wanvideo.md`](04a_dit_wanvideo.md) —— Wan DiT（推荐先读）
> - [`04b_dit_hunyuanvideo.md`](04b_dit_hunyuanvideo.md) —— Hunyuan MMDiT
> - [`04c_dit_cosmos.md`](04c_dit_cosmos.md) —— Cosmos（AdaLN-Zero / world model）
> - [`04d_dit_ltx2.md`](04d_dit_ltx2.md) —— LTX-2（Audio+Video 双模态）
> - [`04e_vae_detailed.md`](04e_vae_detailed.md) —— VAE encode/decode 内部
> - [`04f_other_models_overview.md`](04f_other_models_overview.md) —— 其余模型族概览
> - 数据在模型中如何流动：[`../../03_core_flows/10_data_input_flow_and_shapes.md`](../03_core_flows/10_data_input_flow_and_shapes.md)

## 1. 模块结构

```
models/
├── registry.py        # 模型注册（AST 扫描 EntryClass + 延迟导入）
├── loader/            # 模型/权重加载器（含 FSDP）
├── dits/              # DiT/Transformer
├── encoders/          # text/image encoder
├── vaes/              # VAE
├── schedulers/        # 采样调度器
├── audio/ camera/ upsamplers/   # 音频/相机/超分
├── parameter.py       # vLLM 风格参数类
└── mask_utils.py utils.py
```

## 2. 模型注册（registry.py）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/registry.py
```

- 硬编码注册表：`_TEXT_TO_VIDEO_DIT_MODELS`（14 种）、`_TEXT_ENCODER_MODELS`（11 种）、`_VAE_MODELS`（9 种）、`_SCHEDULERS`（5 种）等。
- `_discover_and_register_models()`（L145）：`os.walk` + `ast.parse` 扫描每个 `.py` 找 `EntryClass = ClassName` 赋值。
- `_LazyRegisteredModel`（L309）：只在 `load_model_cls()` 时 `importlib.import_module`，避免过早 CUDA init。
- `ModelRegistry.resolve_model_cls(architectures)`（L448）：候选架构 → `(model_cls, arch)`。

**为什么延迟导入**：视频模型 import 时会触发 CUDA 初始化，主进程要避免（fork 冲突），所以用 lazy import + 子进程检查（`inspect_model_cls`）。

## 3. 模型加载器（loader/）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/loader/component_loader.py
```

- `ComponentLoader.for_module_type()`（L63）：module_type → 具体 Loader（`TransformerLoader`/`VAELoader`/`TextEncoderLoader`/`SchedulerLoader`...）。
- `TransformerLoader.load()`（L919）：解析 diffusers config → `resolve_model_cls` → `maybe_load_fsdp_model` FSDP 加载 → 可选量化/compile。
- `VAELoader.load()`（L670）：处理各模型特殊 config（GEN3C/Cosmos2.5/LTX-2）。
- `SchedulerLoader.load()`（L1076）：从 config 建 scheduler，应用 `flow_shift`。

权重迭代（`loader/weight_utils.py`）：`safetensors_weights_iterator`（L163，支持 `dist.broadcast`）、`default_weight_loader`。

FSDP 加载（`loader/fsdp_load.py`）：见 [`07_distributed.md`](07_distributed.md)。

## 4. DiT / Transformer（dits/）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/dits/
基类：base.py 的 BaseDiT (L?)
```

### BaseDiT 抽象 forward

```python
def forward(self,
    hidden_states,          # [B, C, T, H, W] 视频 latent
    encoder_hidden_states,  # [B, L, D] 文本嵌入
    timestep,               # [B] 扩散时间步
    encoder_hidden_states_image=None,  # 图像条件
    guidance=None, **kwargs) -> torch.Tensor:  # → [B, C, T', H', W']
```

关键变量含义：
- `hidden_states`：待去噪的视频 latent。
- `encoder_hidden_states`：text encoder 输出的文本条件（cross attention 的 K/V）。
- `timestep`：当前噪声水平，用于 timestep embedding。

### 典型 DiT 结构对比

| 模型 | Block 结构 | 调制 | 位置编码 | 特殊 |
|------|-----------|------|---------|------|
| **WanVideo** (`wanvideo.py` L561) | Self→Cross→FFN | scale_shift_table+temb (AdaLN) | 3D RoPE | QK-Norm, KV cache |
| **HunyuanVideo** (`hunyuanvideo.py` L408) | Double→Single (MMDiT) | ModulateProjection (6参) | 3D RoPE | joint img+txt attention |
| **Cosmos** (`cosmos.py` L536) | Self→Cross→FFN | AdaLN-Zero (含gate) | 3D RoPE+Learnable | GQA, condition mask concat |
| **LTX-2** (`ltx2.py` L2757) | Self→Cross→FFN | AdaLN-Single (PixArt) | RoPE | Audio+Video 双模态 |

### WanTransformer3DModel forward 流程（wanvideo.py L632）

```
输入 hidden_states [B,C,T,H,W], encoder_hidden_states [B,512,4096], timestep [B]
1. patch_embedding → [B, T',H',W', inner_dim] → flatten [B, L_img, inner_dim]
2. sequence_model_parallel_shard（SP 切分序列）
3. condition_embedder(timestep, text, image) → temb, timestep_proj, text_emb
4. for block in blocks: hidden_states = block(...)   # Self→Cross→FFN
5. norm_out + proj_out + unpatchify → [B, C, T, H, W]（预测速度/噪声）
```

张量形状详解见 [`04_knowledge_expansion/01_dit_transformer_for_video.md`](../04_knowledge_expansion/01_dit_transformer_for_video.md)。

## 5. Text Encoders（encoders/）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/encoders/
基类：base.py 的 TextEncoder / ImageEncoder
```

| 文件 | 模型 | 输出 |
|------|------|------|
| `t5.py` | T5/UMT5 (L542/L634) | `[B, L, d_model]`（无 RoPE，相对位置） |
| `clip.py` | CLIP text/vision | `last_hidden_state` |
| `llama.py` | Llama (RoPE, GQA, SwiGLU) | `[B, L, hidden]` |
| `qwen2_5.py`/`qwen3.py`/`gemma.py`/`mistral3.py` | 各家 LLM | 同上 |
| `siglip.py` | SigLIP vision | 图像特征 |

输出统一为 `BaseEncoderOutput(last_hidden_state, attention_mask)`。支持 TP（`QKVParallelLinear`/`RowParallelLinear`）。

## 6. VAEs（vaes/）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/vaes/
基类：common.py 的 ParallelTiledVAE (L17), DiagonalGaussianDistribution (L476)
```

| VAE | latent 通道 | 时间压缩 | 空间压缩 | 特点 |
|-----|-----------|---------|---------|------|
| `AutoencoderKLWan` (wanvae.py) | 16 | ×4 | ×8 | 因果卷积, feature cache |
| `AutoencoderKLHunyuanVideo` | 16 | ×4 | ×8 | 因果卷积+自注意力 |
| `AutoencoderKLHunyuanVideo15` | 32 | ×4 | ×16 | 非 KL VAE |
| `CausalVideoAutoencoder` (ltx2vae) | - | - | - | LTX-2 |

统一 API：
- `encode(x)` → `DiagonalGaussianDistribution`（`[B, 2*z_dim, T', H', W']` mean+logvar）。
- `decode(z)` → 像素 `[B, C, T, H, W]`。
- 自动选择 tiled / spatial_tiled / parallel_tiled（分块处理大视频，省显存）。

VAE 知识见 [`04_knowledge_expansion/03_vae_for_video.md`](../04_knowledge_expansion/03_vae_for_video.md)。

## 7. Schedulers（schedulers/）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/models/schedulers/
基类：base.py 的 BaseScheduler (L?)
```

| Scheduler | 文件 | 说明 |
|-----------|------|------|
| `FlowMatchEulerDiscreteScheduler` | `scheduling_flow_match_euler_discrete.py` | 主力，flow matching |
| `FlowUniPCMultistepScheduler` | `scheduling_flow_unipc_multistep.py` | Wan 默认，高阶多步 |
| `SelfForcingFlowMatchScheduler` | `scheduling_self_forcing_flow_match.py` | 因果 |
| `RCMScheduler` | `scheduling_rcm.py` | - |

`FlowMatchEulerDiscreteScheduler` 三个方法：
- `scale_noise(sample, t, noise)`：加噪 `σ·noise + (1-σ)·sample`。
- `set_timesteps(N)`：构建时间步序列。
- `step(model_output, t, sample)`：Euler 步 `sample + dt·v_pred`。

Scheduler 知识见 [`04_knowledge_expansion/04_scheduler_sampling_solver.md`](../04_knowledge_expansion/04_scheduler_sampling_solver.md)。

## 8. 其他

- `parameter.py`：vLLM 风格参数类（`ColumnvLLMParameter`/`QKVParameter`），支持 TP 加载。
- `utils.py`：`modulate(x, shift, scale)`（AdaLN）、`pred_noise_to_pred_video`（flow matching x0）。
- `mask_utils.py`：`causal_mask_function`、`sdpa_mask`。
- `audio/ltx2_audio_vae.py`：LTX-2 音频 encoder/decoder/vocoder。
- `camera/trajectory.py`：GameCraft 相机 Plücker 坐标。

## 9. 源码阅读重点
1. `dits/wanvideo.py` 的 `WanTransformer3DModel.forward`（一个完整 DiT）。
2. `vaes/common.py` 的 `encode`/`decode` API。
3. `schedulers/scheduling_flow_match_euler_discrete.py` 的 `step`。

## 10. 调试入口
在 DiT `forward` 开头打印 `hidden_states.shape`, `encoder_hidden_states.shape`, `timestep`，理解输入张量。
