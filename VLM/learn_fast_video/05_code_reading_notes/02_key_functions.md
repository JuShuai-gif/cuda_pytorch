# 关键函数索引

> 核心函数速查，按调用链顺序组织。路径相对 `/home/hpc/ghr_code/FastVideo/`。

## 推理链函数

| 函数 | 文件:行 | 输入 | 输出 | 作用 |
|------|---------|------|------|------|
| `VideoGenerator.from_pretrained` | `entrypoints/video_generator.py:144` | model_path, kwargs | VideoGenerator | 加载模型 |
| `VideoGenerator.from_config` | 同上:204 | GeneratorConfig | VideoGenerator | 从配置构建 |
| `VideoGenerator.generate_video` | 同上:359 | prompt, sampling_param | dict | 生成（legacy） |
| `VideoGenerator._generate_single_video` | 同上:658 | prompt, param | dict | 单视频实现 |
| `build_pipeline` | `pipelines/__init__.py:27` | fastvideo_args | pipeline | 构建 pipeline |
| `Executor.execute_forward` | `worker/executor.py:44` | batch, args | ForwardBatch | 跨进程 forward |
| `MultiprocExecutor.collective_rpc` | `worker/multiproc_executor.py:273` | method, kwargs | list[result] | Pipe 广播 |
| `Worker.execute_forward` | `worker/gpu_worker.py:74` | batch, args | ForwardBatch | 调 pipeline.forward |
| `ComposedPipelineBase.forward` | `pipelines/composed_pipeline_base.py:488` | batch, args | ForwardBatch | 顺序执行 stages |

## Stage forward 函数

| 函数 | 文件 | 读取 | 写入 |
|------|------|------|------|
| `InputValidationStage.forward` | `stages/input_validation.py` | prompt, seed | generator, pil_image |
| `TextEncodingStage.encode_text` | `stages/text_encoding.py:117` | prompt | prompt_embeds |
| `TimestepPreparationStage.forward` | `stages/timestep_preparation.py:49` | num_steps | timesteps |
| `LatentPreparationStage.forward` | `stages/latent_preparation.py:104` | 尺寸 | latents(噪声) |
| `DenoisingStage.forward` | `stages/denoising.py:72` | latents,embeds | latents(去噪) |
| `DecodingStage.decode` | `stages/decoding.py:122` | latents | image |

## 模型 forward 函数

| 函数 | 文件:行 | 输入形状 | 输出形状 |
|------|---------|---------|---------|
| `WanTransformer3DModel.forward` | `models/dits/wanvideo.py:632` | `[B,16,T,H,W]`,`[B,512,4096]`,`[B]` | `[B,16,T,H,W]` |
| `AutoencoderKLWan.decode` | `models/vaes/wanvae.py` | `[B,16,T',H',W']` | `[B,3,T,H,W]` |
| `AutoencoderKLWan.encode` | 同上 | `[B,3,T,H,W]` | DiagonalGaussianDistribution |
| `T5EncoderModel.forward` | `models/encoders/t5.py:542` | input_ids `[B,L]` | `[B,L,4096]` |

## Scheduler 函数

| 函数 | 文件:行 | 作用 |
|------|---------|------|
| `FlowMatchEulerDiscreteScheduler.scale_noise` | `scheduling_flow_match_euler_discrete.py:198` | 加噪 `σ·ε+(1-σ)·x` |
| `.set_timesteps` | 同上:285 | 构建时间步 |
| `.step` | 同上:450 | Euler 一步 `x+dt·v` |

## Attention 函数

| 函数 | 文件:行 | 作用 |
|------|---------|------|
| `DistributedAttention.forward` | `attention/layer.py:38` | SP attention（all_to_all+impl） |
| `_cached_get_attn_backend` | `attention/selector.py:92` | 选后端 |
| `sequence_model_parallel_all_to_all_4D` | `distributed/communication_op.py:28` | SP all-to-all |

## 加载函数

| 函数 | 文件:行 | 作用 |
|------|---------|------|
| `ComponentLoader.for_module_type` | `models/loader/component_loader.py:63` | 选 Loader |
| `TransformerLoader.load` | 同上:919 | 加载 DiT |
| `maybe_load_fsdp_model` | `models/loader/fsdp_load.py:100` | FSDP 加载 |
| `shard_model` | 同上:219 | FSDP2 分片 |
| `ModelRegistry.resolve_model_cls` | `models/registry.py:448` | 类解析 |
| `safetensors_weights_iterator` | `models/loader/weight_utils.py:163` | 权重迭代 |

## 分布式函数

| 函数 | 文件:行 | 作用 |
|------|---------|------|
| `maybe_init_distributed_environment_and_model_parallel` | `distributed/parallel_state.py:893` | 初始化分布式 |
| `initialize_model_parallel` | 同上:789 | 建 TP/SP/DP 组 |
| `sequence_model_parallel_shard` | `distributed/communication_op.py:64` | SP 序列切分 |

## 训练函数

| 函数 | 文件:行 | 作用 |
|------|---------|------|
| `run_training_from_config` | `train/entrypoint/train.py:32` | 训练入口 |
| `Trainer.run` | `train/trainer.py:101` | 主循环 |
| `FineTuneMethod.single_train_step` | `train/methods/fine_tuning/finetune.py:49` | 单步（MSE loss） |
| `DMD2Method._dmd_loss` | `train/methods/distribution_matching/dmd2.py:600` | DMD loss |
| `enable_lora_training` | `train/utils/lora.py:192` | LoRA 注入 |
| `build_optimizer_and_scheduler` | `train/utils/optimizer.py:21` | 优化器 |
| `save_checkpoint` | `training/training_utils.py:109` | 旧栈 checkpoint |
| `get_scheduler` | `training/training_utils.py:1472` | LR scheduler |

## 工具函数

| 函数 | 文件:行 | 作用 |
|------|---------|------|
| `modulate` | `models/utils.py:118` | AdaLN 调制 `x*(1+scale)+shift` |
| `pred_noise_to_pred_video` | `models/utils.py:142` | flow matching x0 |
| `compute_density_for_timestep_sampling` | `training/training_utils.py:63` | 时步采样密度 |
