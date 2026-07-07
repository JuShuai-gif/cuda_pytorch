# fastvideo 包总览与 __init__

> 模块作用、关键文件、对外接口。这是所有其他目录笔记的入口。

## 1. 模块作用

`fastvideo/` 是主 Python 包，聚合了推理、训练、数据、评测、分布式、模型的所有实现。对外只暴露 4 个符号。

## 2. 关键源码文件

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/__init__.py
```

```python
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.version import __version__

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]
```

**这段代码做了什么**：定义包的公开 API。`__init__.py` 故意保持极简，只 re-export 三个核心类，让用户心智模型简单。

**为什么这样设计**：隐藏内部复杂度。用户不需要知道 `Executor`、`Worker`、`ComposedPipeline`、`ForwardBatch` 的存在，只操作 `VideoGenerator`。

## 3. 三个核心类的职责边界

### VideoGenerator（门面）
- 定义：`entrypoints/video_generator.py`（1308 行）
- 职责：接收用户请求 → 规范化配置 → 通过 Executor 分发到 worker → 收集结果 → 保存视频。
- **不做**：不直接加载模型（模型在 worker 里），不直接跑 forward。

### PipelineConfig（加载时配置）
- 定义：`configs/pipelines/base.py`（L28）
- 职责：聚合 DiT / VAE / TextEncoder 的架构配置 + 精度设置。
- 关键字段：`dit_config`, `vae_config`, `text_encoder_configs`, `embedded_cfg_scale`, `flow_shift`。

### SamplingParam（运行时配置）
- 定义：`api/sampling_param.py`（411 行）
- 职责：每次生成可变的参数。
- 关键字段：`prompt`, `num_frames`, `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed`, `negative_prompt`。

## 4. 全局参数：FastVideoArgs

```
源码位置：/home/hpc/ghr_code/FastVideo/fastvideo/fastvideo_args.py
关键类：FastVideoArgs (L82), TrainingArgs (L849)
```

`FastVideoArgs` 是推理和训练共享的全局参数 dataclass，比 `PipelineConfig` 更外层：
- 模式：`mode`（INFERENCE/PREPROCESS/FINETUNING/DISTILLATION）、`workload_type`（T2V/I2V/T2I/I2I）。
- 并行：`num_gpus`, `tp_size`, `sp_size`, `hsdp_replicate_dim`, `hsdp_shard_dim`。
- 后端：`distributed_executor_backend`（mp/ray）。
- Offload：`dit_cpu_offload`, `vae_cpu_offload`, `text_encoder_cpu_offload`, `pin_cpu_memory`。
- Compile：`enable_torch_compile`, `torch_compile_kwargs`。
- 配置：`pipeline_config: PipelineConfig`。

关键方法：
- `from_cli_args(args)`（L663）：从 argparse 构建。
- `from_kwargs(**kwargs)`（L714）：从 dict 构建。
- `check_fastvideo_args()`（L731）：验证一致性（offload 互斥、并行度可整除）。

`TrainingArgs(FastVideoArgs)`（L849）：旧训练栈专用，增加 `data_path`、`learning_rate`、蒸馏参数等。

## 5. 其他顶层文件

| 文件 | 作用 |
|------|------|
| `envs.py` | 环境变量集中定义（`FASTVIDEO_ATTENTION_BACKEND` 等） |
| `registry.py` | pipeline config 注册表（model_path → ConfigInfo） |
| `forward_context.py` | 前向上下文管理器（把 timesteps/attn_metadata 传给 DiT） |
| `platforms/` | CUDA/NPU/CPU/MPS 平台抽象 |
| `logger.py`, `profiler.py` | 日志与性能分析 |

## 6. 源码阅读重点

1. 先读 `__init__.py`（1 分钟）。
2. 再读 `fastvideo_args.py` 的 `FastVideoArgs` 字段（了解全局有哪些开关）。
3. 然后进入 `entrypoints/video_generator.py`（见下一篇 `02_entrypoints.md`）。

## 7. 调试入口

```python
import fastvideo
print(fastvideo.__version__)
from fastvideo.fastvideo_args import FastVideoArgs
args = FastVideoArgs(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
print(args)   # 观察所有默认值
```
