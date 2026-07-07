# 目录树

> FastVideo 完整目录结构 + 每个目录的一句话职责。作为源码导航的地图。

## 顶层

```
FastVideo/
├── fastvideo/          # 主 Python 包（861 个 .py）
├── fastvideo-kernel/   # 独立 CUDA extension 包（71 个 kernel/py 文件）
├── apps/               # 应用（dreamverse / fastvideo_studio / performance_dashboard）
├── examples/           # 推理/训练示例
├── scripts/            # 预处理/微调/蒸馏/转换脚本
├── docs/               # mkdocs 文档
├── comfyui/            # ComfyUI 集成
├── tests/              # 顶层测试
├── pyproject.toml      # 依赖与构建
├── README.md
└── mkdocs.yml
```

## fastvideo/ 主包（核心）

```
fastvideo/
├── __init__.py                # 4 个对外导出
├── fastvideo_args.py          # FastVideoArgs / TrainingArgs（全局参数）
├── envs.py                    # 环境变量定义
├── registry.py                # pipeline config 注册表
├── forward_context.py         # 前向上下文（timesteps/attn_metadata 传递）
├── logger.py / profiler.py    # 日志与性能
│
├── entrypoints/               # 【入口层】
│   ├── video_generator.py     #   VideoGenerator 核心门面类
│   ├── streaming_generator.py #   StreamingVideoGenerator 流式
│   ├── cli/                   #   CLI 子命令（generate/serve/bench/eval）
│   ├── openai/                #   OpenAI 兼容 REST API
│   └── streaming/             #   WebSocket 流式服务
│
├── api/                       # 【API 数据结构】
│   ├── sampling_param.py      #   SamplingParam
│   ├── compat.py              #   legacy → GeneratorConfig 转换
│   └── schema.py              #   GeneratorConfig / GenerationRequest 等
│
├── worker/                    # 【编排/执行层】
│   ├── executor.py            #   Executor 抽象基类
│   ├── multiproc_executor.py  #   MultiprocExecutor（多进程）
│   └── gpu_worker.py          #   Worker（每 GPU 加载+运行 pipeline）
│
├── pipelines/                 # 【管线层】
│   ├── composed_pipeline_base.py  # ComposedPipelineBase 基类
│   ├── pipeline_registry.py       # pipeline 注册与选择
│   ├── pipeline_batch_info.py     # ForwardBatch 数据载体
│   ├── lora_pipeline.py           # LoRAPipeline 混入
│   ├── stages/                    # 各 stage（validate/encode/denoise/decode...）
│   ├── basic/                     # 各模型 pipeline（wan/hunyuan/cosmos/ltx2...）
│   ├── preprocess/                # 预处理 pipeline
│   └── training/                  # 训练 pipeline 粘合（预留）
│
├── models/                    # 【模型层】
│   ├── registry.py            #   模型注册（AST 扫描 EntryClass）
│   ├── loader/                #   模型/权重加载器（含 FSDP）
│   ├── dits/                  #   DiT/Transformer（wanvideo/hunyuanvideo/cosmos/ltx2...）
│   ├── encoders/              #   text/image encoder（t5/clip/llama/qwen/gemma...）
│   ├── vaes/                  #   VAE（wanvae/hunyuanvae/ltx2vae...）
│   ├── schedulers/            #   采样调度器（flow match/unipc/rcm...）
│   ├── audio/                 #   LTX-2 音频 VAE/vocoder
│   ├── camera/                #   相机轨迹（Plücker 坐标）
│   ├── upsamplers/            #   超分上采样器
│   ├── parameter.py           #   vLLM 风格参数类（TP 加载）
│   └── mask_utils.py          #   attention mask 工具
│
├── attention/                 # 【注意力层】
│   ├── selector.py            #   后端选择逻辑
│   ├── layer.py               #   DistributedAttention / LocalAttention
│   ├── backends/              #   flash/sage/sdpa/vsa/bsa/sla/vmoba/qat
│   └── utils/                 #   flash_attn cute / no_pad
│
├── layers/                    # 【基础层】
│   ├── linear.py              #   TP 线性层全家桶
│   ├── layernorm.py           #   RMSNorm / FP32LayerNorm / ScaleResidual
│   ├── rotary_embedding*.py   #   RoPE（含 3D）
│   ├── mlp.py / activation.py #   MLP / 激活
│   ├── lora/                  #   LoRA 层
│   └── quantization/          #   FP8/FP4/NVFP4 量化配置
│
├── distributed/               # 【分布式层】
│   ├── parallel_state.py      #   GroupCoordinator + 组管理
│   ├── communication_op.py    #   SP/TP 通信操作
│   ├── utils.py               #   padding/shard 工具
│   └── device_communicators/  #   CUDA/NPU/CPU 通信器 + PyNccl
│
├── dataset/                   # 【数据层】
│   ├── parquet_dataset_*.py   #   Parquet 数据集（iterable/map）
│   ├── latent_datasets.py     #   预计算 latent 数据集
│   ├── preprocessing_datasets.py  # 原始视频/文本处理
│   ├── transform.py           #   视频变换（crop/resize/normalize）
│   ├── dataloader/            #   Parquet schema / writer / collation
│   └── validation_dataset.py  #   验证集
│
├── train/                     # 【新训练框架】
│   ├── entrypoint/train.py    #   run_training_from_config
│   ├── trainer.py             #   Trainer.run 主循环
│   ├── methods/               #   FineTune/DMD2/SelfForcing/KD
│   ├── models/                #   ModelBase / WanModel...
│   ├── callbacks/             #   EMA/validation/grad_clip
│   └── utils/                 #   config/lora/optimizer/checkpoint/builder
│
├── training/                  # 【旧训练框架】
│   ├── training_pipeline.py   #   TrainingPipeline
│   ├── distillation_pipeline.py   # DistillationPipeline
│   ├── training_utils.py      #   save_checkpoint / get_scheduler / EMA_FSDP
│   └── *_training_pipeline.py #   各模型单体训练 pipeline
│
├── eval/                      # 【评测层】
│   ├── evaluator.py / api.py  #   Evaluator / evaluate()
│   ├── metrics/               #   psnr/ssim/lpips/fvd/vbench/...
│   ├── datasets/              #   VBench prompt 数据集
│   └── io/                    #   视频解码/输入标准化
│
├── configs/                   # 【配置层】
│   ├── pipelines/             #   PipelineConfig + 各模型 config
│   ├── models/                #   DiTConfig/VAEConfig/EncoderConfig
│   └── backend/               #   后端配置
│
├── platforms/                 # 【平台抽象】CUDA/NPU/CPU/MPS
├── hooks/ / logging_utils/    # 钩子与日志
├── workflow/                  # 工作流（含 preprocess）
└── third_party/               # 第三方（eval / longcat_video）
```

## fastvideo-kernel/ CUDA 包

```
fastvideo-kernel/
├── csrc/                      # C++/CUDA 源
│   ├── common_extension.cpp   # pybind11 入口
│   ├── attention/*.cu         # STA / block sparse（Hopper）
│   └── turbodiffusion/*.cu    # INT8 GEMM/RMSNorm/LayerNorm/quant
├── python/fastvideo_kernel/   # Python 封装 + Triton fallback
├── attn_qat_infer/            # Blackwell FP4 kernel
├── include/{cutlass,tk}/      # submodule
├── CMakeLists.txt / build.sh  # 构建
└── tests/ / benchmarks/
```

## 导航建议

- 想读推理：`entrypoints/` → `pipelines/` → `models/`。
- 想读训练：`train/`（新）或 `training/`（旧）。
- 想读并行：`distributed/` + `layers/linear.py` + `models/loader/fsdp_load.py`。
- 想读加速：`attention/` + `fastvideo-kernel/`。

各目录详解见本文件夹其余 `.md`。
