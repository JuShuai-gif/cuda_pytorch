# 如何添加新 Pipeline

> 当已有模型但需要新的生成流程（如新的 workload 或去噪结构）时。

## 1. 何时需要新 pipeline

- 新模型（见 [`01_how_to_add_new_model.md`](01_how_to_add_new_model.md)）。
- 同模型新 workload：T2V → I2V/V2V（如 `wan_pipeline.py` vs `wan_i2v_pipeline.py`）。
- 去噪结构性不同：causal（`wan_causal_pipeline.py`）、蒸馏（`wan_dmd_pipeline.py`）。

## 2. Pipeline 骨架

```python
# pipelines/basic/mymodel/my_variant_pipeline.py
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

class MyVariantPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def initialize_pipeline(self, fastvideo_args):
        # 可选：覆盖 scheduler 等
        pass

    def create_pipeline_stages(self, fastvideo_args):
        # 组装 stages
        ...

EntryClass = MyVariantPipeline
```

## 3. 复用 vs Fork stage（关键决策）

参考 `pipelines/AGENTS.md`：
- **优先复用**已有 stage（validate/encode/timestep/latent/denoise/decode）。
- **仅当** ForwardBatch 形状不同或去噪循环结构性差异时才 fork。

Fork 示例（去噪不同）：
```python
# pipelines/stages/my_denoising.py
class MyDenoisingStage(PipelineStage):
    def forward(self, batch, fastvideo_args):
        # 自定义去噪循环
        return batch
```

## 4. I2V pipeline 需要的额外 stage

I2V 比 T2V 多图像条件：
```python
self.add_stage("image_encoding_stage", ImageEncodingStage())       # CLIP 图像特征
self.add_stage("image_vae_encoding_stage", ImageVAEEncodingStage()) # 图像→latent 条件
```

## 5. 注册

pipeline 通过 `EntryClass` 被 `pipeline_registry.py:import_pipeline_classes` 扫描。若需新 workload_type，在 `register_configs` 指定 `workload_types`。

## 6. ForwardBatch 新字段

若 stage 需要新数据，先加到 `ForwardBatch`（`pipeline_batch_info.py`），不要用 module-level dict 传状态（反模式）。

## 7. 测试

```python
g = VideoGenerator.from_pretrained(model_path, num_gpus=1)  # workload 匹配到新 pipeline
g.generate_video(...)
```

## 8. 检查清单

- [ ] 有 `EntryClass`。
- [ ] `_required_config_modules` 正确。
- [ ] 优先复用 stage。
- [ ] 新字段加到 ForwardBatch。
- [ ] stage 确定性（相同输入→相同输出）。

## 9. 参考
- `pipelines/basic/wan/`（T2V/I2V/V2V/Causal/DMD 多变体）。
- `pipelines/stages/`（现有 stage）。
- `pipelines/AGENTS.md`（设计约定）。
