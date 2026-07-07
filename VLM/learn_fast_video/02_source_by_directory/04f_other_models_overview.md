# DiT 精读 · 其余模型族概览

> 除 Wan/Hunyuan/Cosmos/LTX-2 外，FastVideo 还支持多个模型族。本文概览它们的定位、结构特点、源码位置。
> 精读了主力模型后，这些可按需查阅。

## 1. 模型族总表

| 模型 | 文件 | 行数 | 类型 | 特点 |
|------|------|------|------|------|
| WanVideo | `wanvideo.py` | 748 | T2V/I2V | 主力，Self→Cross→FFN，见 [04a](04a_dit_wanvideo.md) |
| Causal Wan | `causal_wanvideo.py` | 759 | 因果 T2V/I2V | 自回归流式（Self-Forcing） |
| HunyuanVideo | `hunyuanvideo.py` | 834 | T2V | MMDiT，见 [04b](04b_dit_hunyuanvideo.md) |
| HunyuanVideo1.5 | `hunyuanvideo15.py` | 766 | I2V + SR | 超分，SigLIP encoder |
| HunyuanGameCraft | `hunyuangamecraft.py` | 363 | 游戏条件 | 相机/动作控制 |
| Cosmos | `cosmos.py` | 743 | world model | AdaLN-Zero，见 [04c](04c_dit_cosmos.md) |
| Cosmos 2.5 | `cosmos2_5.py` | 967 | world model | T2W/V2W 自动路由 |
| GEN3C | `gen3c.py` | 1021 | 3D 相机 | 继承 Cosmos2.5 + Plücker 坐标 |
| LTX-2 | `ltx2.py` | 3099 | Audio+Video | 双模态，见 [04d](04d_dit_ltx2.md) |
| SD3.5 | `sd3.py` | 1078 | T2I | MMDiT（图像） |
| Flux2 | `flux_2.py` | 1094 | T2I | packed latent，Mistral3 encoder |
| LongCat | `longcat.py` | 1149 | T2V/I2V | refine 多阶段 |
| Kandinsky5 | `kandinsky5.py` | 761 | T2V | - |
| Magi | `magi_human.py` | 867 | 人物视频 | - |
| Stable Audio | `stable_audio.py` | 389 | 音频 | Oobleck VAE + T5 conditioner |

多文件模型族（子目录）：`hyworld/`、`lingbotworld/`、`matrixgame2/`、`matrixgame3/`。

## 2. Causal Wan（causal_wanvideo.py）

因果视频 DiT，用于自回归/流式生成（CausalWan2.2）。与标准 Wan 区别：
- **因果 attention**：token 只 attend 过去帧（time-causal mask）。
- **KV cache 传播**：逐块生成，历史块的 KV 缓存传给后续块（`predict_noise_streaming`）。
- 配合 `CausalDenoisingStage` + Self-Forcing 蒸馏（见 [`../03_core_flows/09_distillation_flow.md`](../03_core_flows/09_distillation_flow.md)）。
- `_relative_rope.py` 的 `relativistic_window_offsets` 将 KV-cache 窗口重映射到训练位置范围。

## 3. SD3.5 / Flux2（图像 MMDiT）

虽然 FastVideo 主打视频，但也支持图像模型：
- **SD3.5**（sd3.py）：Stable Diffusion 3.5，MMDiT 架构（同 Hunyuan 的 double/single 思想，但用于图像）。
- **Flux2**（flux_2.py）：黑森林 Flux，用 **packed latent**（token 打包）+ Mistral3 文本编码器。`cast_prompt_embeds_to_dit_dtype=True`。VAE 是 `AutoencoderKLFlux2`。

图像模型 `workload_type=T2I`，latent 无时间维（或 T=1）。

## 4. Cosmos 2.5 / GEN3C（world model）

- **Cosmos 2.5**（cosmos2_5.py）：升级版 Cosmos，`Cosmos25DenoisingStage` 根据是否有条件输入自动路由 T2W（text-to-world）或 V2W（video-to-world）。
- **GEN3C**（gen3c.py）：继承 Cosmos 2.5，加 3D 相机条件。在 latent（16ch）上 concat `condition_video_input_mask`(1ch) + `condition_video_pose`(frame_buffer_max×32ch)。用 `camera/trajectory.py` 生成 Plücker 坐标。

## 5. HunyuanVideo 1.5 + 超分

- `hunyuanvideo15.py`：I2V，用 SigLIP 图像编码器。
- `upsamplers/hunyuan15.py`：`SRTo720pUpsampler` / `SRTo1080pUpsampler`，3D 因果卷积残差块超分。
- pipeline 分两阶段：先低分辨率生成，再超分。

## 6. 游戏/交互模型

- **HunyuanGameCraft**（hunyuangamecraft.py）：相机轨迹 + 动作（WASD）条件，`GameCraftDenoisingStage`。
- **MatrixGame2/3**（matrixgame2/3/ 子目录）：动作控制（mouse_cond/keyboard_cond），用于交互式游戏视频生成（dreamverse 类应用）。
- **HYWorld**（hyworld/ 子目录）：相机控制世界模型。

这些模型的条件输入（camera_states/mouse_cond/keyboard_cond/grid_sizes）在 `SamplingParam` 和 `ForwardBatch` 里都有对应字段。

## 7. Stable Audio（纯音频）

- `stable_audio.py`：音频扩散 DiT。
- VAE 是 `oobleck.py`（Oobleck autoencoder）。
- 文本条件用 `stable_audio_conditioner.py`（T5 + NumberConditioners，控制音频时长等）。
- `SamplingParam` 有 `audio_start_in_s`、`init_audio`、`inpaint_audio` 等字段。

## 8. 共同模式（读任何 DiT 的套路）

无论哪个模型，DiT 都遵循：
```
1. patchify（latent → token 序列）
2. condition embedding（timestep + text + 可选 image/camera/audio）
3. N 个 transformer block（attention + FFN + 调制）
4. unpatchify（token → latent）
```
差异主要在：
- **调制方式**：AdaLN / AdaLN-Zero / AdaLN-Single / 全局 vec。
- **attention 组织**：Self+Cross 分开 / joint（MMDiT）/ 因果 / 双模态 cross。
- **额外条件**：图像 / 相机 / 音频 / mask。

抓住这个套路，任何新模型都能快速上手。

## 9. 每个模型的入口

所有 DiT 文件末尾都有 `EntryClass = XxxTransformer3DModel`，供 `models/registry.py` 的 AST 扫描自动注册。找模型实现就搜 `EntryClass`。

## 10. 阅读建议
1. 先精读 Wan（[04a](04a_dit_wanvideo.md)），掌握套路。
2. 想理解 MMDiT 看 Hunyuan（[04b](04b_dit_hunyuanvideo.md)）。
3. 想理解 world model 看 Cosmos（[04c](04c_dit_cosmos.md)）。
4. 想理解多模态看 LTX-2（[04d](04d_dit_ltx2.md)）。
5. 其余按需查本文。
