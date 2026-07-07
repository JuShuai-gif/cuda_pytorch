# 源码阅读顺序

> 给定 FastVideo 861 个 py 文件，从哪读起？本文给出分阶段的文件优先级清单。配合 [`01_key_classes.md`](01_key_classes.md) 的优先级标注。

## 优先级定义
- **P0**：必须先看，理解主干。
- **P1**：重要，深入时看。
- **P2**：了解即可，用到再看。

## 阶段 1：推理主干（P0）

按此顺序读，每个文件只抓主干不纠结细节：

1. `fastvideo/__init__.py` — 4 个导出（1 分钟）。
2. `fastvideo/entrypoints/video_generator.py` — `from_pretrained` / `generate_video` / `_generate_single_video`。
3. `fastvideo/pipelines/composed_pipeline_base.py` — `forward` 的 `for stage in stages`。
4. `fastvideo/pipelines/pipeline_batch_info.py` — `ForwardBatch` 字段。
5. `fastvideo/pipelines/stages/denoising.py` — 去噪循环。
6. `fastvideo/pipelines/stages/decoding.py` — VAE decode。

读完能回答："prompt 怎么变成视频？"

## 阶段 2：模型内部（P0-P1）

7. `fastvideo/models/dits/wanvideo.py` — 一个完整 DiT 的 forward。
8. `fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py` — flow matching。
9. `fastvideo/models/vaes/wanvae.py` + `vaes/common.py` — VAE。
10. `fastvideo/pipelines/stages/text_encoding.py` — prompt 编码。
11. `fastvideo/attention/layer.py` — DistributedAttention。
12. `fastvideo/attention/selector.py` — 后端选择。

## 阶段 3：加载与配置（P1）

13. `fastvideo/pipelines/__init__.py` — `build_pipeline`。
14. `fastvideo/models/loader/component_loader.py` — 组件加载。
15. `fastvideo/models/loader/fsdp_load.py` — FSDP 加载。
16. `fastvideo/configs/pipelines/base.py` — PipelineConfig。
17. `fastvideo/registry.py` + `models/registry.py` — 注册机制。

## 阶段 4：并行与执行（P1）

18. `fastvideo/worker/multiproc_executor.py` — 多进程编排。
19. `fastvideo/worker/gpu_worker.py` — worker。
20. `fastvideo/distributed/parallel_state.py` — 通信组。
21. `fastvideo/distributed/device_communicators/base_device_communicator.py` — AllToAll4D。
22. `fastvideo/layers/linear.py` — TP 线性层。

## 阶段 5：训练（P1-P2）

23. `fastvideo/train/entrypoint/train.py` — 训练入口。
24. `fastvideo/train/trainer.py` — 主循环。
25. `fastvideo/train/methods/fine_tuning/finetune.py` — 微调。
26. `fastvideo/train/utils/lora.py` — LoRA。
27. `fastvideo/train/methods/distribution_matching/dmd2.py` — 蒸馏。

## 阶段 6：Kernel（P2）

28. `fastvideo/attention/backends/video_sparse_attn.py` — VSA 后端。
29. `fastvideo-kernel/python/fastvideo_kernel/ops.py` — Python 封装。
30. `fastvideo-kernel/csrc/common_extension.cpp` — pybind 注册。
31. `fastvideo-kernel/csrc/turbodiffusion/norm/rmsnorm.cu` — 最简单 kernel。
32. `fastvideo-kernel/csrc/attention/block_sparse_h100.cu` — 复杂 kernel（进阶）。

## 阶段 7：数据与评测（P2）

33. `fastvideo/dataset/parquet_dataset_iterable_style.py`
34. `fastvideo/dataset/dataloader/schema.py`
35. `fastvideo/eval/evaluator.py` + `metrics/common/fvd/metric.py`

## 读码技巧

1. **先读 AGENTS.md**：`fastvideo/pipelines/AGENTS.md`、`fastvideo/attention/AGENTS.md`、`fastvideo/models/AGENTS.md` 有设计意图说明。
2. **顺着 ForwardBatch 走**：在 `ComposedPipelineBase.forward` 循环里打印每个 stage 后的 batch 字段。
3. **用小配置调试**：`num_frames=17, height=256, width=256` 快速跑通。
4. **善用 registry**：不知道 model_path 对应哪个类，看 `registry.py` 的注册表。
5. **区分双训练栈**：`train/`（新）vs `training/`（旧），别混。

## 相关
- 关键类索引：[`01_key_classes.md`](01_key_classes.md)
- 关键函数：[`02_key_functions.md`](02_key_functions.md)
- 调用图：[`03_call_graphs.md`](03_call_graphs.md)
