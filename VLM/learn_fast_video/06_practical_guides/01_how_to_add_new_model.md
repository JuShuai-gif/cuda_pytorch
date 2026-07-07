# 如何添加新模型

> 基于 FastVideo 的 registry + stage 架构，添加新 DiT 模型的步骤。参考 `docs` 的 add_pipeline 指南和 `pipelines/AGENTS.md`。

> 注意：以下为基于源码结构推断的通用步骤，具体细节以官方文档 `hao-ai-lab.github.io/FastVideo/contributing/add_pipeline.html` 为准（**部分步骤待官方文档确认**）。

## 1. 涉及的三层

添加模型要动三处：
1. **模型实现**（`models/dits/`, `models/vaes/` 等）。
2. **配置**（`configs/`）。
3. **pipeline**（`pipelines/basic/<model>/`）。

## 2. Step 1：实现 DiT

在 `fastvideo/models/dits/mymodel.py`：
```python
from fastvideo.models.dits.base import BaseDiT

class MyTransformer3DModel(BaseDiT):
    _supported_attention_backends = (...)
    _fsdp_shard_conditions = [...]
    param_names_mapping = {...}   # HF 权重名 → 内部名 regex

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        # patch embed → blocks → unpatchify
        return output

EntryClass = MyTransformer3DModel   # 关键：供 registry AST 扫描
```
参考现有 `wanvideo.py`。`EntryClass` 让 `models/registry.py:_discover_and_register_models` 自动发现。

## 3. Step 2：配置

在 `configs/models/dits/mymodel.py` 定义 `MyDiTConfig` / `MyDiTArchConfig`（`num_attention_heads` 等）。
在 `configs/pipelines/` 定义 `MyPipelineConfig(PipelineConfig)`。

在 `fastvideo/registry.py` 注册：
```python
register_configs(
    pipeline_config_cls=MyPipelineConfig,
    workload_types=(WorkloadType.T2V,),
    hf_model_paths=["Org/MyModel-Diffusers"],
    model_detectors=[lambda p: "mymodel" in p.lower()],
    model_family="mymodel", default_preset="mymodel_t2v")
```

## 4. Step 3：Pipeline

在 `pipelines/basic/mymodel/mymodel_pipeline.py`：
```python
class MyPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def create_pipeline_stages(self, fastvideo_args):
        self.add_stage("input_validation_stage", InputValidationStage())
        self.add_stage("prompt_encoding_stage", TextEncodingStage(...))
        self.add_stage("timestep_preparation_stage", TimestepPreparationStage(scheduler))
        self.add_stage("latent_preparation_stage", LatentPreparationStage(scheduler, transformer))
        self.add_stage("denoising_stage", DenoisingStage(transformer, scheduler, vae=vae))
        self.add_stage("decoding_stage", DecodingStage(vae))

EntryClass = MyPipeline   # 供 pipeline_registry 扫描
```

**尽量复用已有 stage**。只有去噪循环结构性不同（causal / refine / 多流）才 fork 新 stage（参考 `CosmosDenoisingStage`）。

## 5. Step 4：权重转换（如需要）

若权重命名与 diffusers 不同，写 `scripts/checkpoint_conversion/mymodel_to_diffusers.py`，用 `param_names_mapping` regex 映射（参考 `wan_to_diffusers.py`）。

## 6. Step 5：测试

```python
g = VideoGenerator.from_pretrained("Org/MyModel-Diffusers", num_gpus=1)
g.generate_video(prompt="test", num_frames=17, height=256, width=256)
```

## 7. 检查清单

- [ ] DiT 文件有 `EntryClass`。
- [ ] pipeline 文件有 `EntryClass`。
- [ ] `register_configs` 已调用。
- [ ] `param_names_mapping` 正确（权重能加载）。
- [ ] 复用 stage 优先，必要才 fork。

## 8. 关键源码参考
- `models/dits/wanvideo.py`（DiT 模板）。
- `pipelines/basic/wan/wan_pipeline.py`（pipeline 模板）。
- `models/registry.py`（发现机制）。
- `pipelines/AGENTS.md`（设计约定）。

## 9. 相关
- 添加 pipeline：[`02_how_to_add_new_pipeline.md`](02_how_to_add_new_pipeline.md)
