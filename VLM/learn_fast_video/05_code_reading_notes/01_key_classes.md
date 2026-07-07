# 关键类索引

> 全项目核心类的速查表。字段：类/函数 | 文件路径 | 作用 | 上游调用 | 下游调用 | 优先级。
>
> 路径均相对 `/home/hpc/ghr_code/FastVideo/`。优先级：P0 必看 / P1 重要 / P2 了解。

## 入口层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `VideoGenerator` | `fastvideo/entrypoints/video_generator.py` | 用户门面 | 用户代码 | Executor | P0 |
| `VideoGenerator.from_pretrained` | 同上 L144 | 加载模型 | 用户 | from_config→Executor | P0 |
| `VideoGenerator._generate_single_video` | 同上 L658 | 单视频生成 | generate_video | executor.execute_forward | P0 |
| `StreamingVideoGenerator` | `entrypoints/streaming_generator.py` L73 | 流式生成 | dreamverse | VideoGenerator | P2 |
| `SamplingParam` | `api/sampling_param.py` L17 | 运行时采样参数 | 用户 | ForwardBatch | P0 |
| `FastVideoArgs` | `fastvideo_args.py` L82 | 全局参数 | from_config | Executor/Worker | P1 |
| `TrainingArgs` | `fastvideo_args.py` L849 | 旧栈训练参数 | 训练脚本 | TrainingPipeline | P1 |

## 编排/执行层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `Executor` | `worker/executor.py` L16 | 编排抽象基类 | VideoGenerator | Worker | P1 |
| `MultiprocExecutor` | `worker/multiproc_executor.py` L76 | 多进程编排 | Executor.get_class | Worker 子进程 | P1 |
| `MultiprocExecutor.collective_rpc` | 同上 L273 | Pipe RPC 广播 | execute_forward | Worker | P1 |
| `Worker` | `worker/gpu_worker.py` L16 | GPU 执行 | 子进程 | pipeline.forward | P1 |
| `Worker.init_device` | 同上 L35 | 初始化 CUDA+分布式+模型 | worker_main | build_pipeline | P1 |

## 管线层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `ComposedPipelineBase` | `pipelines/composed_pipeline_base.py` L31 | pipeline 基类 | build_pipeline | stages | P0 |
| `ComposedPipelineBase.forward` | 同上 L488 | 顺序执行 stages | Worker | 各 stage | P0 |
| `ComposedPipelineBase.load_modules` | 同上 L357 | 加载模块 | __init__ | ComponentLoader | P1 |
| `ForwardBatch` | `pipelines/pipeline_batch_info.py` L62 | 数据载体 | 全 pipeline | 各 stage | P0 |
| `PipelineStage` | `pipelines/stages/base.py` L29 | stage 基类 | pipeline | 各具体 stage | P0 |
| `DenoisingStage` | `pipelines/stages/denoising.py` L47 | 去噪循环 | pipeline | DiT+scheduler | P0 |
| `DecodingStage` | `pipelines/stages/decoding.py` L24 | VAE decode | pipeline | vae.decode | P0 |
| `TextEncodingStage` | `pipelines/stages/text_encoding.py` L20 | prompt 编码 | pipeline | text_encoder | P1 |
| `LatentPreparationStage` | `pipelines/stages/latent_preparation.py` L25 | 初始化噪声 | pipeline | randn_tensor | P1 |
| `WanPipeline` | `pipelines/basic/wan/wan_pipeline.py` L19 | Wan T2V pipeline | registry | stages | P1 |
| `LoRAPipeline` | `pipelines/lora_pipeline.py` L95 | LoRA 混入 | 具体 pipeline | lora 层 | P1 |

## 模型层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `BaseDiT` | `models/dits/base.py` | DiT 抽象基类 | - | 各 DiT | P1 |
| `WanTransformer3DModel` | `models/dits/wanvideo.py` L561 | Wan DiT | DenoisingStage | blocks+attention | P0 |
| `HunyuanVideoTransformer3DModel` | `models/dits/hunyuanvideo.py` L408 | Hunyuan MMDiT | DenoisingStage | double/single blocks | P2 |
| `CosmosTransformer3DModel` | `models/dits/cosmos.py` L536 | Cosmos DiT | CosmosDenoisingStage | blocks | P2 |
| `LTX2Transformer3DModel` | `models/dits/ltx2.py` L2757 | LTX-2 双模态 | DenoisingStage | LTXModel | P2 |
| `AutoencoderKLWan` | `models/vaes/wanvae.py` L1103 | Wan VAE | DecodingStage | encoder/decoder | P1 |
| `ParallelTiledVAE` | `models/vaes/common.py` L17 | VAE 基类/tiling | 各 VAE | encode/decode | P1 |
| `T5EncoderModel` | `models/encoders/t5.py` L542 | T5 文本编码 | TextEncodingStage | - | P1 |
| `FlowMatchEulerDiscreteScheduler` | `models/schedulers/scheduling_flow_match_euler_discrete.py` | flow matching | DenoisingStage | step | P0 |
| `ModelRegistry` | `models/registry.py` L462 | 模型注册 | loader | _LazyRegisteredModel | P1 |
| `TransformerLoader` | `models/loader/component_loader.py` L919 | DiT 加载 | load_module | fsdp_load | P1 |

## 注意力层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `DistributedAttention` | `attention/layer.py` L38 | SP 注意力 | DiT block | attn_impl+all_to_all | P1 |
| `AttentionImpl` | `attention/backends/abstract.py` L113 | 后端抽象 | layer | 各后端 | P1 |
| `FlashAttentionImpl` | `attention/backends/flash_attn.py` | FlashAttention | layer | flash-attn 库 | P1 |
| `SDPAImpl` | `attention/backends/sdpa.py` L? | SDPA fallback | layer | F.sdpa | P1 |
| `VideoSparseAttentionImpl` | `attention/backends/video_sparse_attn.py` | VSA | layer | kernel | P2 |
| `_cached_get_attn_backend` | `attention/selector.py` L92 | 后端选择 | layer | 平台 | P1 |

## 分布式层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `GroupCoordinator` | `distributed/parallel_state.py` L117 | 通信组 | init | ProcessGroup | P1 |
| `initialize_model_parallel` | 同上 L789 | 建 TP/SP/DP 组 | maybe_init | GroupCoordinator | P1 |
| `DistributedAutograd.AllToAll4D` | `distributed/device_communicators/base_device_communicator.py` L123 | SP all-to-all | DistributedAttention | NCCL | P1 |
| `maybe_load_fsdp_model` | `models/loader/fsdp_load.py` L100 | FSDP 加载 | TransformerLoader | shard_model | P1 |
| `ColumnParallelLinear` / `QKVParallelLinear` | `layers/linear.py` L344/L674 | TP 线性层 | DiT/encoder | all-gather/reduce | P1 |

## 训练层

| 类/函数 | 文件路径 | 作用 | 上游 | 下游 | 优先级 |
|---------|---------|------|------|------|--------|
| `run_training_from_config` | `train/entrypoint/train.py` L32 | 训练入口 | torchrun | Trainer | P1 |
| `Trainer.run` | `train/trainer.py` L101 | 主循环 | 入口 | method | P1 |
| `TrainingMethod` | `train/methods/base.py` L26 | 方法基类 | Trainer | Model | P1 |
| `FineTuneMethod` | `train/methods/fine_tuning/finetune.py` L17 | 微调 | Trainer | student | P1 |
| `DMD2Method` | `train/methods/distribution_matching/dmd2.py` L22 | DMD2 蒸馏 | Trainer | student/teacher/critic | P2 |
| `WanModel` | `train/models/wan/wan.py` L56 | Wan 训练包装 | builder | transformer | P1 |
| `enable_lora_training` | `train/utils/lora.py` L192 | LoRA 注入 | Model | replace_submodule | P1 |
| `CheckpointManager` | `train/utils/checkpoint.py` L156 | checkpoint | Trainer | DCP | P2 |
| `TrainingPipeline` | `training/training_pipeline.py` L58 | 旧栈微调 | 脚本 | - | P2 |
| `DistillationPipeline` | `training/distillation_pipeline.py` L47 | 旧栈蒸馏 | 脚本 | - | P2 |

## 配置层

| 类/函数 | 文件路径 | 作用 | 优先级 |
|---------|---------|------|--------|
| `PipelineConfig` | `configs/pipelines/base.py` L28 | 管线配置根 | P1 |
| `DiTConfig` | `configs/models/dits/base.py` L44 | DiT 架构配置 | P1 |
| `VAEConfig` | `configs/models/vaes/base.py` L22 | VAE 配置 | P1 |
| `register_configs` | `registry.py` L132 | 配置注册 | P1 |
