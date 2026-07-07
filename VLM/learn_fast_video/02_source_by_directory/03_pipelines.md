# pipelines —— 管线层

> 模块作用：把"prompt → 视频"的流程拆成可组合的 stage。这是理解 FastVideo 推理的核心。

## 1. 模块作用

`fastvideo/pipelines/` 实现"Stage 组合"设计模式：每个 pipeline = 一串 `PipelineStage`，每个 stage 一个动词（validate/encode/schedule/denoise/decode）。添加新模型 = 组装 stages。

设计原则（`pipelines/AGENTS.md`）：
- Stage 必须确定性（相同输入→相同输出）。
- Stage 通过重新赋值 `ForwardBatch` 字段来 mutate，新字段先加到 dataclass。
- 禁止在 stage 里读 `os.getenv`；配置从 `FastVideoArgs`/`PipelineConfig` 读。
- 仅当去噪循环有结构性差异（causal / refine / 多流）时才 fork stage。

## 2. 目录结构

```
pipelines/
├── composed_pipeline_base.py   # ComposedPipelineBase 基类（523 行）
├── pipeline_registry.py        # pipeline 注册与选择
├── pipeline_batch_info.py      # ForwardBatch 数据载体（335 行）
├── lora_pipeline.py            # LoRAPipeline 混入（449 行）
├── stages/                     # ~30 个可复用 stage
├── basic/                      # 各模型 pipeline
├── preprocess/                 # 预处理 pipeline
└── training/                   # 训练 pipeline（预留）
```

## 3. ComposedPipelineBase（基类）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/composed_pipeline_base.py
关键类：ComposedPipelineBase (L31)
```

### 核心属性与方法

| 成员 | 行号 | 作用 |
|------|------|------|
| `modules` | L44 | 已加载模块字典（vae/transformer/text_encoder/scheduler） |
| `_required_config_modules` | L48 | 必须加载的模块列表（子类声明） |
| `load_modules()` | L357 | 读 `model_index.json`，逐模块 `PipelineComponentLoader.load_module` |
| `post_init()` | L155 | 推理：`initialize_pipeline` + `create_pipeline_stages` + torch.compile |
| `add_stage()` | L466 | 注册 stage 到 `_stages` 列表 |
| **`forward()`** | L488 | **核心**：`for stage in stages: batch = stage(batch, args)` |

```python
@torch.no_grad()
def forward(self, batch, fastvideo_args):
    if not self.post_init_called:
        self.post_init()
    for stage in self.stages:
        batch = stage(batch, fastvideo_args)
    return batch
```

**为什么这样设计**：stateless pipeline——pipeline 自身不存 batch 状态，所有状态在 `ForwardBatch` 里流动。这让 pipeline 可复用、可测试、可组合。

## 4. PipelineStage（stage 基类）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/stages/base.py
关键类：PipelineStage (L29)
```

- 抽象方法 `forward(batch, args) -> ForwardBatch`（子类必实现）。
- `__call__`（L114）包装：`verify_input` → `forward`（记录耗时到 `batch.logging_info`）→ `verify_output`。

## 5. 各 Stage 一览

| Stage | 文件 | 读 → 写 |
|-------|------|---------|
| `InputValidationStage` | `input_validation.py` | prompt/seed → generator, pil_image |
| `TextEncodingStage` | `text_encoding.py` | prompt → prompt_embeds `[B,L,D]` |
| `ConditioningStage` | `conditioning.py` | guidance_scale → do_cfg（no-op forward） |
| `TimestepPreparationStage` | `timestep_preparation.py` | num_steps → timesteps |
| `LatentPreparationStage` | `latent_preparation.py` | 尺寸 → latents 噪声 |
| **`DenoisingStage`** | `denoising.py` | latents+embeds → 去噪后 latents |
| `DecodingStage` | `decoding.py` | latents → output 像素 |
| `ImageEncodingStage` | `image_encoding.py` | image → image_embeds（CLIP） |
| `ImageVAEEncodingStage` | `image_encoding.py` | image → image_latent（I2V 条件） |
| `EncodingStage` | `encoding.py` | 像素 → latent（预处理用） |

变体（fork）：`CosmosDenoisingStage`、`DmdDenoisingStage`、`CausalDenoisingStage`、`SD35DenoisingStage`、`HYWorldDenoisingStage` 等。

### DenoisingStage 细节

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/stages/denoising.py
关键：DenoisingStage.forward (L72-635)
```

去噪循环核心（见 [`03_core_flows/05_scheduler_and_sampling_flow.md`](../03_core_flows/05_scheduler_and_sampling_flow.md)）：
```python
for i, t in enumerate(timesteps):
    latent_model_input = scheduler.scale_model_input(latents, t)
    noise_pred = current_model(latent_model_input, prompt_embeds, t_expand, ...)  # DiT
    # CFG
    noise_pred = uncond + guidance_scale * (cond - uncond)
    latents = scheduler.step(noise_pred, t, latents)[0]   # L567
```

### DecodingStage 细节

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/stages/decoding.py
关键：DecodingStage.decode (L122-182)
```
```python
latents = self._denormalize_latents(latents)   # 逆 scaling/shift
image = self.vae.decode(latents)                # L170
image = (image / 2 + 0.5).clamp(0, 1)           # → [0,1]
```

## 6. ForwardBatch（数据载体）

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/pipeline_batch_info.py
关键类：ForwardBatch (L62)
```

贯穿所有 stage 的 dataclass，200+ 字段。关键：

| 字段 | 由谁写 | 含义 |
|------|--------|------|
| `prompt_embeds` | TextEncoding | 文本嵌入 |
| `latents` | LatentPrep → Denoising | 当前 latent `[B,C,T,H,W]` |
| `timesteps` | TimestepPrep | 去噪时间步 |
| `do_classifier_free_guidance` | `__post_init__` | 自动根据 guidance_scale>1 计算 |
| `output` | Decoding | 最终像素输出 |
| `logging_info` | 每个 stage | 各 stage 耗时 |
| `extra` | 各 stage | 扩展字段 |

## 7. Pipeline 注册与选择

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/pipeline_registry.py
```

- `import_pipeline_classes()`（L99）：`pkgutil.walk_packages` 扫描 `basic/<model>/`，找 `EntryClass` 属性。
- 约定：每个模型 pipeline 文件导出 `EntryClass = WanPipeline`。
- `resolve_pipeline_cls()`（L79）：根据 pipeline_type + workload_type 返回 pipeline 类。

## 8. 各模型 pipeline

`basic/` 下每个模型一个子目录：`wan/`, `hunyuan/`, `hunyuan15/`, `cosmos/`, `ltx2/`, `sd35/`, `flux_2/`, `longcat/`, `gen3c/`, `gamecraft/`, `hyworld/`, `matrixgame2/3/`, `turbodiffusion/`, `stable_audio/`。

以 Wan 为例（`basic/wan/`）：`wan_pipeline.py`（T2V）、`wan_i2v_pipeline.py`、`wan_causal_pipeline.py`、`wan_v2v_pipeline.py`、`wan_dmd_pipeline.py`。

## 9. LoRAPipeline

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/pipelines/lora_pipeline.py
关键类：LoRAPipeline (L95) 继承 ComposedPipelineBase
```
- `convert_to_lora_layers()`（L231）：把线性层替换为 `BaseLayerWithLoRA`。
- `set_lora_adapter()`（L296）：加载 `.safetensors` LoRA 权重。
- `merge/unmerge_lora_weights()`（L425）：推理时合并/取消合并。

具体 pipeline 如 `WanPipeline(LoRAPipeline, ComposedPipelineBase)` 多重继承获得 LoRA 能力。

## 10. 源码阅读重点
1. `composed_pipeline_base.py:488` 的 `forward`。
2. `stages/denoising.py` 的去噪循环。
3. `pipeline_batch_info.py` 的 `ForwardBatch` 字段。

## 11. 调试入口
在 `ComposedPipelineBase.forward` 循环里打印 `type(stage).__name__` 和 `batch.latents.shape`，观察每个 stage 后张量的变化。
