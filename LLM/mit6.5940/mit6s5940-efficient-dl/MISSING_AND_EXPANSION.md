# MIT 6.S5940 Efficient Deep Learning Project 补齐总控文档

本文档用于把当前项目从“由 AI 生成的课程资料集合”补成“可以系统学习 MIT 6.S5940 / Efficient Deep Learning / TinyML，并能迁移到工业实战的工程化学习项目”。

## 如何驱动 Codex 继续补齐

后续你可以直接复制下面的 prompt 给 Codex：

```text
请根据 /home/hpc/ghr_code/cuda_pytorch/mit6.5940/mit6s5940-efficient-dl/MISSING_AND_EXPANSION.md 补齐其中的 P0-M1。
要求：
1. 先阅读该模块说明、相关 note/src/lab/project 文件；
2. 按现有项目风格修改，不做无关重构；
3. 必须补工业界常用方法、实战坑、验收指标和可运行验证；
4. 修改后说明改了哪些文件、如何运行、还剩什么缺口。
```

常用 prompt 示例：

```text
请根据 MISSING_AND_EXPANSION.md 补齐 P0-M2：把 lecture-03/04 剪枝部分改成工业实战级内容，并同步修改 src/lecture-03、src/lecture-04 和 labs/lab-01。
```

```text
请根据 MISSING_AND_EXPANSION.md 补齐 P0-M3：把 lecture-05/06 量化部分补成 PTQ/QAT/LLM weight-only quantization 的实战教程，并补可运行代码。
```

```text
请根据 MISSING_AND_EXPANSION.md 补齐 P1-M8：重构最终 project，使 full_pipeline 可运行、可配置、可生成工业级 benchmark report。
```

如果只想先分析方案、不改代码：

```text
请根据 MISSING_AND_EXPANSION.md 分析 P1-M8 最终项目重构方案，先不要改代码。
```

## 当前项目判断

当前项目已经有 23 篇 lecture note、lecture 代码、labs、papers、final project 和模型压缩脚本。问题不是“完全空”，而是：

- 笔记偏讲义化，缺少工业决策树、真实部署约束、失败案例和验收标准。
- `note/`、`src/`、`labs/`、`project/` 之间没有严格闭环：读完一讲后不知道该跑哪个实验、看什么指标、怎样判断合格。
- 很多代码使用 synthetic data，适合 smoke test，但不能证明真实模型压缩收益。
- 最终项目 README 描述了完整工业流水线，但需要更强的配置、报告、错误处理和验收门槛。
- 课程内容和工业常用工具链还没有充分连接：TensorRT、ONNX Runtime、OpenVINO、TFLite Micro、MNN/ncnn、vLLM、llama.cpp、ExecuTorch 等。
- 不能承诺“看完就完美掌握”。合理目标应是：看完并跑完本项目后，能够独立完成一个模型从 baseline 到压缩、导出、部署和 benchmark 的工程闭环。

## 补齐目标

补完后，本项目应该做到：

1. 每讲都有“课程知识点 -> 工业场景 -> 实验代码 -> 验收指标 -> 常见坑”。
2. 每个核心主题都有可运行实验：剪枝、量化、蒸馏、NAS、TinyML、LLM 推理、ViT/Diffusion 加速、分布式训练、端侧训练。
3. 最终 project 可以执行一条完整 pipeline：baseline -> compression -> export -> runtime benchmark -> report。
4. 报告不只写参数量和模型大小，还包括 latency P50/P95/P99、throughput、memory、accuracy delta、output error、部署失败原因。
5. README 明确区分“课程学习”、“工业实践”、“当前可运行能力”、“后续待补”。

## P0-M1. 顶层 README 与学习路径重构

### 现状

README 有课程地图和模型压缩说明，但没有明确说明：

- 哪些内容已经可运行。
- 哪些内容只是概念演示。
- 每一阶段如何验收。
- 如何按工业项目方式学习。

### 需要补齐

- 在 README 开头加入“真实学习目标”和“不要误解”的说明。
- 加入 4 条路径：课程通读、压缩实战、TinyML/嵌入式、LLM 推理优化。
- 加入每阶段验收命令。
- 加入项目成熟度表：notes、src、labs、project、reports。

### 验收标准

- 新手打开 README 能知道先读什么、跑什么、看什么指标。
- README 不再给出无法验证的“完整掌握”承诺。
- 每个阶段都能落到具体文件和命令。

## P0-M2. 剪枝模块补齐：Lecture 03/04 + src + Lab 1

### 现状

`lecture-03.md` 和 `lecture-04.md` 已经讲了非结构化剪枝、结构化通道剪枝、敏感度分析和工业指标；`src/lecture-03`、`src/lecture-04` 有 CPU synthetic demo。

### 主要缺口

- 需要区分“把权重置零”和“真正减少网络结构”。很多生成代码只 mask weight，不会加速。
- 需要加入 N:M sparsity、block sparsity、TensorRT 2:4 sparse 的实战条件。
- 需要加入 BN gamma pruning、dependency graph、residual connection channel alignment。
- 需要加入剪枝策略选择：unstructured / structured / channel / head / MLP hidden / LLM sparsity。
- 需要把 Lab 1 从单纯 VGG/CIFAR 描述改为：sensitivity scan -> pruning policy -> finetune -> export -> latency validation。

### 工业实战必须讲清楚

- 非结构化稀疏只有在 sparse kernel 或 2:4 sparse tensor core 支持下才可能加速。
- 结构化剪枝必须真的改模型 shape，不能只把 channel 置零。
- ResNet/UNet/Transformer 中有 skip connection，剪一个分支的 channel 会影响另一个分支。
- 剪枝后的模型必须重新导出 ONNX/TensorRT/OpenVINO 并在目标硬件实测。
- 评价标准至少包括：accuracy delta、latency P50/P99、model size、peak memory、FLOPs、真实 runtime speedup。

### 推荐新增内容

- `note/pruning_industry_playbook.md`
- `src/model_compression/pruning_policy.py`
- `tests/test_pruning_policy.py`
- `labs/lab-01/industrial_requirements.md`

## P0-M3. 量化模块补齐：Lecture 05/06 + Lab 2

### 现状

已有量化笔记和部分代码，但需要更明确区分 PyTorch eager quant、FX graph mode、ONNX Runtime quant、TensorRT PTQ、LLM weight-only quant。

### 主要缺口

- 校准数据集选择和 representative data 的要求。
- per-tensor、per-channel、per-group、per-token 的取舍。
- symmetric/asymmetric、static/dynamic、weight-only、activation quant 的区别。
- QAT 的 fake quant、observer、batchnorm folding、prepare/convert 流程。
- LLM 常用方法：AWQ、GPTQ、SmoothQuant、ZeroQuant、QLoRA、KV cache quant。
- 工业验收：layerwise SQNR、cosine similarity、task metric、latency、memory。

### 工业实战必须讲清楚

- INT8 PTQ 不是只把 tensor cast 成 int8，而是 scale/zp、calibration、kernel 支持、graph rewrite。
- W4A16 LLM 量化重点是权重和 activation outlier，不能照搬 CNN INT8 PTQ。
- TensorRT INT8 需要 calibration cache 或 Q/DQ graph。
- ONNX Runtime quantization 对 opset、动态 shape、Conv/MatMul fusion 有要求。
- 量化失败通常来自 outlier、LayerNorm、Softmax、最后分类头、回归头或 action head。

### 推荐新增内容

- `note/quantization_industry_playbook.md`
- `src/model_compression/quantization_observers.py`
- `src/model_compression/quantization_report.py`
- `labs/lab-02/calibration_checklist.md`

## P0-M4. 统一 benchmark 与指标体系

### 现状

`src/model_compression/benchmark_compression.py` 已经比较完整，能生成报告。但课程其他 lecture 代码未统一使用同一套 metrics。

### 主要缺口

- latency 需要统一 P50/P90/P95/P99 和 warmup/repeat。
- GPU 测量需要 `torch.cuda.synchronize()`。
- CPU 测量需要固定线程数、记录 BLAS 后端、避免数据加载干扰。
- memory 需要区分 model size、activation memory、peak RSS、CUDA allocated/reserved。
- 对 LLM 需要加 tokens/s、prefill latency、decode latency、KV cache memory。
- 对机器人/VLA 需要 action MSE、max action deviation、control-loop deadline miss rate。

### 推荐新增内容

- `src/model_compression/benchmarking.py`
- `src/model_compression/reporting.py`
- `note/benchmarking_and_profiling.md`
- 所有 lecture code 尽量复用统一 benchmark 工具。

## P0-M5. src 与 note 的闭环

### 现状

每讲都有 `note/lecture-XX.md` 和 `src/lecture-XX/main.py`，但 `src/README.md` 只列了 09-23，和实际 01-23 不一致。

### 需要补齐

- `src/README.md` 覆盖 01-23。
- 每个 lecture src README 写清楚：对应 note、运行命令、输出指标、常见失败。
- 每篇 note 末尾加“对应代码实验”和“工业验收问题”。

### 验收标准

- 读 `note/lecture-XX.md` 后能直接运行 `src/lecture-XX/main.py`。
- 代码输出能和笔记中的概念对应。

## P1-M6. NAS 与硬件感知搜索补齐

### 必须覆盖

- Search space: depth/width/kernel/expand ratio/resolution。
- Search strategy: random/evolution/RL/differentiable/weight-sharing。
- Hardware-aware objective: accuracy-latency Pareto frontier。
- Once-for-All / MCUNet 的核心思想。
- 实测 latency lookup table，而不是只用 FLOPs。

### 工业实战

- NAS 的收益来自目标硬件上的真实 latency table。
- 同一个模型在 CPU/GPU/NPU 上 Pareto frontier 不同。
- 搜索出来的模型必须经过 export/runtime 验证。

## P1-M7. TinyML / TinyEngine / MCU 部署补齐

### 必须覆盖

- SRAM/Flash/activation memory 的硬约束。
- im2col memory blowup。
- operator scheduling 与 in-place buffer reuse。
- CMSIS-NN、TFLite Micro、TinyEngine 的角色。
- int8 kernel、per-channel scale、requantization。

### 工业实战

- MCU 上最大问题常是 activation memory，不是参数量。
- 算子是否支持比模型结构更重要。
- 需要 memory planner 和 arena allocator。
- 需要静态内存规划，避免 malloc/free。

## P1-M8. 最终 project 工业化重构

### 现状

`project/edge_ai_compression_deployment` 有模块化结构，但需要增强可运行性、配置、报告和验收。

### 需要补齐

- `--quick` 模式，保证 CPU 上 1-3 分钟内跑通。
- `--real-data` 或 `--dataset cifar10` 选项，允许真实数据集。
- pipeline 每阶段产物明确：checkpoint、pruned model、quantized model、onnx、benchmark json、report md。
- 失败不静默跳过：缺少 onnxruntime/TensorRT 时报告 skipped reason。
- report 增加工业指标：P50/P95/P99、throughput、peak memory、accuracy delta、output MSE、export status、runtime status。
- 加入 experiment matrix：baseline、pruned、quantized、pruned+quantized、distilled。

### 验收标准

- `python main.py --mode full_pipeline --quick` 能在 CPU 跑通。
- 输出 `reports/comparison_report.md` 和 `reports/experiment_results.json`。
- report 中每个优化项都有收益、损失和是否达标。

## P1-M9. LLM 推理优化补齐

### 必须覆盖

- Prefill vs decode。
- KV cache memory。
- PagedAttention / continuous batching。
- FlashAttention / GQA / MQA。
- AWQ/GPTQ/SmoothQuant/QLoRA。
- Speculative decoding。
- llama.cpp、vLLM、TensorRT-LLM 的定位。

### 工业实战

- LLM decode 通常 memory-bound，weight-only quant 的主要收益是减少权重带宽。
- batch size、context length、KV cache 决定吞吐和显存。
- tokens/s 要分 prefill 和 decode，不要只报平均。

## P1-M10. ViT / Diffusion / 多模态/VLA 补齐

### 必须覆盖

- ViT token pruning / token merging / efficient attention。
- Diffusion step reduction、distillation、UNet quantization、attention memory。
- VLA action head 压缩时，不能只看分类 accuracy，要看 action MSE、trajectory deviation、success rate。

### 工业实战

- 视觉模型通常受 resize/preprocess、memory layout、runtime kernel 影响很大。
- Diffusion 加速常常来自减少采样步数，而不是单步 kernel 小优化。
- 机器人控制关注 P99 deadline 和稳定性，平均 latency 不够。

## P1-M11. 分布式训练与端侧训练补齐

### 必须覆盖

- DDP、FSDP、ZeRO、tensor/pipeline parallel。
- Communication/computation overlap。
- Gradient checkpointing、activation offload。
- Federated learning、TinyTL、adapter-only training。

### 工业实战

- 分布式训练优化的瓶颈通常是通信、显存和数据 pipeline。
- 端侧训练关注 activation memory 和更新参数规模。

## P2-M12. Papers 目录重构

### 现状

`papers/` 有 20 篇导读，但需要统一模板。

### 推荐模板

每篇 paper 应包含：

- 论文解决什么工业问题。
- 核心方法一句话。
- 关键公式或系统设计。
- 对应 MIT 6.S5940 哪一讲。
- 代码中哪里体现。
- 工业落地限制。
- 面试可讲版本。

## P2-M13. Labs 重构

### 现状

Lab README 有目标、内容、提交要求，但缺少工业验收 rubric。

### 需要补齐

- 每个 Lab 增加 `industrial_requirements.md`。
- 每个 Lab 增加 `grading_rubric.md`。
- starter code 的 TODO 应覆盖关键算法，而不是只填小函数。
- solution code 应输出 report。

## 统一验收指标

| 类别 | 必须指标 | 工业解释 |
|---|---|---|
| 精度 | accuracy / mAP / perplexity / MSE | 模型是否还能用 |
| 延迟 | P50/P95/P99 | 端侧和机器人要看尾延迟 |
| 吞吐 | QPS / images/s / tokens/s | 服务成本 |
| 内存 | model size / peak RSS / CUDA peak / KV cache | 能否部署和并发 |
| 计算 | FLOPs / MACs / arithmetic intensity | 只解释趋势，不替代实测 |
| 导出 | ONNX/TensorRT/OpenVINO/TFLite status | 能否进入部署链路 |
| 鲁棒性 | calibration drift / outlier / deadline miss | 生产环境稳定性 |

## 推荐补齐顺序

1. P0-M1 README 与学习路径重构。
2. P0-M5 src 与 note 闭环。
3. P0-M4 统一 benchmark 与指标体系。
4. P0-M2 剪枝模块补齐。
5. P0-M3 量化模块补齐。
6. P1-M8 最终 project 工业化重构。
7. P1-M7 TinyML 部署补齐。
8. P1-M9 LLM 推理优化补齐。
9. P1-M10 ViT/Diffusion/VLA 补齐。
10. P2-M12/P2-M13 papers 和 labs 重构。

## “学完是否掌握”的现实标准

不要用“看完所有 Markdown”作为掌握标准。应该用下面的输出判断：

- 能独立解释 pruning/quantization/distillation/NAS/TinyML/LLM serving 的适用条件和失败模式。
- 能对一个 PyTorch 模型建立 FP32 baseline。
- 能做至少两种压缩策略，并量化 accuracy-latency-memory trade-off。
- 能导出 ONNX，并说明 runtime 不兼容时如何定位。
- 能生成一份包含 P50/P95/P99、throughput、memory、accuracy delta 的 benchmark report。
- 能根据目标硬件选择 TensorRT、ONNX Runtime、OpenVINO、TFLite Micro、MNN/ncnn、llama.cpp 或 vLLM。
- 能讲清楚为什么 FLOPs 降了但 latency 没降。
- 能把实验结果整理成面试/项目汇报。
