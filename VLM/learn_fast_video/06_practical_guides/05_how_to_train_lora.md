# 如何训练 LoRA

> 用新训练框架（`fastvideo/train/`）做 LoRA 微调的完整步骤。

## 1. 前置：准备数据

先把视频预处理成 Parquet（见 [`04_how_to_add_dataset.md`](04_how_to_add_dataset.md)）：
```bash
torchrun ... fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --data_merge_path index.txt \
    --output_dir ./processed --preprocess_task "t2v" --num_frames 81 --train_fps 16
```

## 2. 编写 YAML 配置

```yaml
# my_lora.yaml
models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    lora:
      enable: true
      rank: 16
      alpha: 32
      target_modules: [to_q, to_k, to_v, to_out]

method:
  _target_: fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod

training:
  distributed:
    num_gpus: 8
    sp_size: 1
    hsdp_replicate_dim: 8
    hsdp_shard_dim: 1
  data:
    data_path: ./processed
    train_batch_size: 1
    num_frames: 81
  optimizer:
    learning_rate: 1e-4
    betas: [0.9, 0.999]
    weight_decay: 0.01
    lr_scheduler: constant
    lr_warmup_steps: 100
  loop:
    max_train_steps: 5000
    gradient_accumulation_steps: 1
  checkpoint:
    output_dir: ./lora_output
    training_state_checkpointing_steps: 1000

callbacks:
  grad_clip:
    _target_: fastvideo.train.callbacks.grad_clip.GradNormClipCallback
```

## 3. 启动训练

```bash
torchrun --nproc_per_node 8 -m fastvideo.train.entrypoint.train --config my_lora.yaml
# 或用 wrapper
bash examples/train/run.sh my_lora.yaml
```

## 4. 背后发生什么

```
run.sh → run_training_from_config (train/entrypoint/train.py)
  → build_from_config → WanModel(lora.enable=true)
    → _enable_lora_if_configured → enable_lora_training (lora.py:192)
        冻结全部 → 注入 LoRA 层 → 只 lora_A/B 可训练
  → Trainer.run
    → FineTuneMethod.single_train_step → MSE flow matching loss
    → 只更新 LoRA 参数
```

详见 [`../03_core_flows/08_lora_finetune_flow.md`](../03_core_flows/08_lora_finetune_flow.md)。

## 5. 关键参数调节

| 参数 | 建议 |
|------|------|
| `rank` | 16-64，越大容量越强但越慢 |
| `alpha` | 通常 2×rank |
| `target_modules` | attention 投影为主 |
| `learning_rate` | LoRA 可比全量高（1e-4 级） |
| `hsdp_replicate_dim` | 小模型全复制 |

## 6. LoRA vs 全量微调

去掉 YAML 的 `lora` 块即全量微调（`apply_trainable` 所有参数）。全量更强但需更多显存、更小学习率（1e-6 级）。

## 7. Checkpoint

保存在 `output_dir/checkpoint-N/`（DCP 格式）。恢复：
```yaml
checkpoint:
  resume_from_checkpoint: ./lora_output/checkpoint-1000
```

## 8. 推理时用 LoRA

```python
generator = VideoGenerator.from_pretrained(base_model, lora_path="path/to/lora.safetensors", num_gpus=1)
```
（`pipelines/lora_pipeline.py:set_lora_adapter` 加载）

## 9. 旧栈方式

```bash
torchrun ... fastvideo/training/wan_training_pipeline.py \
    --lora_training True --lora_rank 16 --lora_alpha 32 ...
```

## 10. 参考
- `examples/train/configs/fine_tuning/wan/`（现成 YAML）。
- `train/methods/fine_tuning/finetune.py`（loss）。
- `train/utils/lora.py`（注入）。
- 知识：[`../04_knowledge_expansion/09_lora_finetuning.md`](../04_knowledge_expansion/09_lora_finetuning.md)
