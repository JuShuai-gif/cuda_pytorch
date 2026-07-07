# train / training —— 训练层

> 模块作用：模型后训练。**两套并存**：`train/`（新，组合式 YAML）和 `training/`（旧，单体 pipeline）。

## 1. 双栈对照

| | `fastvideo/train/`（新） | `fastvideo/training/`（旧） |
|--|------------------------|--------------------------|
| 范式 | Method × Model × Callback | 单体 pipeline |
| 配置 | YAML + `TrainingConfig` | argparse + `TrainingArgs` |
| 入口 | `torchrun -m fastvideo.train.entrypoint.train --config x.yaml` | `torchrun fastvideo/training/wan_training_pipeline.py --flag` |

## 2. 新栈结构

```
train/
├── entrypoint/train.py    # run_training_from_config（入口）
├── trainer.py             # Trainer.run（主循环）
├── methods/               # 训练算法
│   ├── base.py            #   TrainingMethod (ABC)
│   ├── fine_tuning/       #   FineTuneMethod, DiffusionForcingSFTMethod
│   ├── distribution_matching/  # DMD2Method, SelfForcingMethod
│   └── knowledge_distillation/ # KDMethod, KDCausalMethod
├── models/                # ModelBase, WanModel...
├── callbacks/             # EMA, validation, grad_clip
└── utils/                 # config/lora/optimizer/checkpoint/builder
```

### 入口（entrypoint/train.py L32）

```python
def run_training_from_config(config_path, dry_run, overrides):
    cfg = load_run_config(config_path, overrides)               # YAML → RunConfig
    maybe_init_distributed_environment_and_model_parallel(tp, sp)
    _, method, dataloader, start_step = build_from_config(cfg)  # 实例化 Model×Method×DataLoader
    trainer = Trainer(tc, config, callbacks)
    checkpoint_manager = CheckpointManager(method, dataloader, ...)
    trainer.run(method, dataloader, max_steps, start_step, checkpoint_manager)
```

### 主循环（trainer.py L101 Trainer.run）

```python
for step in progress:
    for accum_iter in range(grad_accum):
        batch = next(data_stream)
        loss_map, outputs, metrics = method.single_train_step(batch, step)
        method.backward(loss_map, outputs, grad_accum_rounds=grad_accum)
    callbacks.on_before_optimizer_step(method, step)   # grad clip
    method.optimizers_schedulers_step(step)            # optimizer + scheduler
    method.optimizers_zero_grad(step)
    checkpoint_manager.maybe_save(step)
    self._run_method_validation(method, step)
```

### Method 层次

```
TrainingMethod (base.py L26, ABC)
├── FineTuneMethod (fine_tuning/finetune.py)     # MSE flow matching
├── DiffusionForcingSFTMethod (dfsft.py)          # 非均匀时步
├── DMD2Method (distribution_matching/dmd2.py)    # student/teacher/critic
├── SelfForcingMethod (self_forcing.py)           # 因果流式 rollout
└── KDMethod (knowledge_distillation/kd.py)       # ODE 轨迹缓存
```

核心接口：`single_train_step(batch, iter) → (loss_map, outputs, metrics)`。

### FineTuneMethod（最简单）

```python
# fine_tuning/finetune.py:95
pred = student.predict_noise(noisy_latents, timesteps, batch)
if precondition_outputs:
    pred_x0 = noisy_latents - pred * sigmas
    loss = F.mse_loss(pred_x0, clean_latents)
else:
    target = noise - clean_latents          # flow matching v-prediction
    loss = F.mse_loss(pred, target)
```

## 3. LoRA 微调（train/utils/lora.py）

```
关键函数：enable_lora_training (L192)
```
```python
def enable_lora_training(transformer, lora_rank, lora_alpha, lora_target_modules):
    transformer.requires_grad_(False)                      # 冻结全部
    for name, module in transformer.named_modules():
        if _is_target_layer(name, target_modules):
            lora_layer = get_lora_layer(module, rank, alpha)
            replace_submodule(transformer, name, lora_layer)  # 注入 LoRA
    _replicate_lora_parameters(transformer)                # DTensor Replicate 包装
```
默认目标层：`q_proj/k_proj/v_proj/o_proj/to_q/to_k/to_v/to_out/to_qkv`。

全量微调路径：`apply_trainable(transformer, trainable=True)`（所有参数 requires_grad=True）。

## 4. 蒸馏（DMD2Method）

```
源码位置：train/methods/distribution_matching/dmd2.py
三角色：student(generator)/teacher(real score)/critic(fake score)
```

```python
# _dmd_loss (L600)
real_cfg_x0 = teacher_uncond + w*(teacher_cond - teacher_uncond)  # teacher CFG
faker_x0 = critic.predict_x0(...)
grad = (faker_x0 - real_cfg_x0) / denom       # 分布匹配梯度
loss = 0.5 * MSE(gen_x0, (gen_x0 - grad).detach())
# _critic_flow_matching_loss: MSE(critic_noise_pred, noise - gen_x0)
```

详见 [`03_core_flows/09_distillation_flow.md`](../03_core_flows/09_distillation_flow.md)。

## 5. Optimizer / Scheduler

```
源码位置：train/utils/optimizer.py L21
```
`AdamW(params, lr, betas, weight_decay, eps=1e-8)` + `get_scheduler`（支持 constant/linear/cosine/polynomial 等，`training/training_utils.py:1472`）。

蒸馏中 student 和 critic 各有独立 optimizer 和学习率。

## 6. Checkpoint（DCP）

```
新栈：train/utils/checkpoint.py 的 CheckpointManager (L156)
旧栈：training/training_utils.py 的 save_checkpoint (L109)
```
```
checkpoint-1000/
├── metadata.json       # step + config
├── dcp/               # torch.distributed.checkpoint 分片
└── rng_state_rank0.pt # per-rank RNG
```
`checkpoint_state()`（base.py L112）返回 DCP 就绪的 `{roles.*, optimizers.*, schedulers.*}`。

## 7. EMA

```
源码位置：training/training_utils.py L1570 的 EMA_FSDP + train/callbacks/ema.py
```
- `mode="local_shard"`：每 rank 维护 local shard float32 EMA。
- `ema_context(transformer)`：验证时临时交换 EMA 权重。

## 8. 旧栈

- `TrainingPipeline`（`training/training_pipeline.py` L58，继承 `LoRAPipeline`）：全量/LoRA。
- `DistillationPipeline`（`training/distillation_pipeline.py` L47，1514 行）：DMD/DMD2，含 MoE 双模型。

## 9. TrainingArgs（旧栈参数）

```
源码位置：fastvideo_args.py L849
```
`data_path`, `train_batch_size`, `learning_rate`, `lr_scheduler`, `max_grad_norm`, `weighting_scheme`, `lora_rank/alpha/training`, 蒸馏参数（`real_score_model_path`, `dmd_denoising_steps`）, Self-forcing 参数（`num_frame_per_block`, `context_noise`）。

## 10. 源码阅读重点
1. `trainer.py` 的 `run` 主循环。
2. `fine_tuning/finetune.py` 的 loss。
3. `utils/lora.py` 的 `enable_lora_training`。
4. `dmd2.py` 的三角色 loss。

## 11. 调试入口
```bash
bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml
```
在 `Trainer.run` 的循环里打印 `loss_map`、`metrics`。
