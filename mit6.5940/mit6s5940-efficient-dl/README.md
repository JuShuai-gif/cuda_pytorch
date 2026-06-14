# MIT 6.5940 高效深度学习与 TinyML 中文学习工程

> **核心理念**: 高效深度学习不是单纯让模型变小，而是在准确率、延迟、吞吐、内存、能耗、硬件约束之间做工程权衡。

## 课程简介

MIT 6.5940 (Efficient Deep Learning Computing / TinyML) 由 MIT HAN Lab 的 **Prof. Song Han** 主讲，系统覆盖从模型压缩到边缘部署的全栈技术。本工程将课程转化为完整的中文学习路线，包含笔记、代码、实验和工业项目。

## 学习前置知识

- **基础**: Python, PyTorch, 线性代数, 概率论
- **深度学习**: CNN, Transformer, 训练/推理流程
- **系统基础**: 基本的 CPU/GPU 内存层次理解, latency vs throughput 概念
- **推荐前置课程**: MIT 6.S191 (Intro to DL), CS231n

## 课程知识地图

```
第 1-2 讲: 引言与基础        → 为什么需要高效深度学习
第 3-4 讲: 剪枝 (Pruning)    → 去掉不重要的权重/通道
第 5-6 讲: 量化 (Quantization)→ 用低位宽表示权重和激活
第 7-8 讲: 神经架构搜索 (NAS) → 自动搜索最优网络结构
第 9 讲:   知识蒸馏 (KD)     → 大模型教小模型
第 10-11 讲: TinyML 系统      → MCUNet + TinyEngine 端侧部署
第 12-15 讲: LLM 优化        → Transformer/LLM 的压缩与部署
第 16-18 讲: 高级模型优化    → ViT, GAN, Diffusion 加速
第 19-21 讲: 分布式与端侧训练 → 大规模训练 + 设备端学习
第 22-23 讲: 课程总结与前沿  → 量子 ML 等前沿方向
```

## 学习路线总览

```
入门 (1-2周)     → Lecture 01-02 + Lab 0
核心压缩 (2-4周) → Lecture 03-09 + Lab 1-3  （剪枝/量化/NAS/知识蒸馏）
系统部署 (1-2周) → Lecture 10-11            （MCUNet + TinyEngine 端侧部署）
LLM专项 (2-3周)  → Lecture 12-15 + Lab 4-5  （Transformer/LLM 压缩与部署）
高级专题 (2-3周) → Lecture 16-21            （ViT/Diffusion/分布式/端侧训练）
工业项目 (2-4周) → project/edge_ai_compression_deployment
```

## 目录结构

```
mit6s5940-efficient-dl/
├── note/           # 中文讲解笔记 (23讲 + 附录A: 算力-板卡-模型效率)
├── src/            # 可运行的 PyTorch 代码示例
├── labs/           # 重构的实验 (含 starter/solution/report)
├── papers/         # 论文中文导读 (20篇, 全部来自 MIT HAN Lab)
├── project/        # 端侧 AI 模型压缩与部署完整项目
├── diagrams/       # Mermaid 架构图
├── README.md       # 本文件
└── requirements.txt# Python 依赖
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 阅读笔记 - 从 note/lecture-01.md 开始
# 3. 运行代码示例
cd src/lecture-01 && python main.py

# 4. 运行 Labs
cd labs/lab-01 && jupyter notebook

# 5. 运行最终项目
cd project/edge_ai_compression_deployment
python main.py --mode full_pipeline
```

## 各讲入口

| 讲次 | 主题 | 笔记 | 代码 | Lab |
|------|------|------|------|-----|
| 01 | 引言 | [note/lecture-01.md](note/lecture-01.md) | [src/lecture-01/](src/lecture-01/) | - |
| 02 | 基础 | [note/lecture-02.md](note/lecture-02.md) | [src/lecture-02/](src/lecture-02/) | Lab 0 |
| 03 | 剪枝 I | [note/lecture-03.md](note/lecture-03.md) | [src/lecture-03/](src/lecture-03/) | - |
| 04 | 剪枝 II | [note/lecture-04.md](note/lecture-04.md) | [src/lecture-04/](src/lecture-04/) | Lab 1 |
| 05 | 量化 I | [note/lecture-05.md](note/lecture-05.md) | [src/lecture-05/](src/lecture-05/) | - |
| 06 | 量化 II | [note/lecture-06.md](note/lecture-06.md) | [src/lecture-06/](src/lecture-06/) | Lab 2 |
| 07 | NAS I | [note/lecture-07.md](note/lecture-07.md) | [src/lecture-07/](src/lecture-07/) | - |
| 08 | NAS II | [note/lecture-08.md](note/lecture-08.md) | [src/lecture-08/](src/lecture-08/) | Lab 3 |
| 09 | 知识蒸馏 | [note/lecture-09.md](note/lecture-09.md) | [src/lecture-09/](src/lecture-09/) | - |
| 10 | MCUNet | [note/lecture-10.md](note/lecture-10.md) | [src/lecture-10/](src/lecture-10/) | - |
| 11 | TinyEngine | [note/lecture-11.md](note/lecture-11.md) | [src/lecture-11/](src/lecture-11/) | - |
| 12 | Transformer & LLM | [note/lecture-12.md](note/lecture-12.md) | [src/lecture-12/](src/lecture-12/) | - |
| 13 | LLM 部署 | [note/lecture-13.md](note/lecture-13.md) | [src/lecture-13/](src/lecture-13/) | - |
| 14 | LLM 后训练 | [note/lecture-14.md](note/lecture-14.md) | [src/lecture-14/](src/lecture-14/) | Lab 4 |
| 15 | 长上下文 LLM | [note/lecture-15.md](note/lecture-15.md) | [src/lecture-15/](src/lecture-15/) | - |
| 16 | ViT | [note/lecture-16.md](note/lecture-16.md) | [src/lecture-16/](src/lecture-16/) | - |
| 17 | 高效 GAN/视频/点云 | [note/lecture-17.md](note/lecture-17.md) | [src/lecture-17/](src/lecture-17/) | - |
| 18 | Diffusion 模型 | [note/lecture-18.md](note/lecture-18.md) | [src/lecture-18/](src/lecture-18/) | - |
| 19 | 分布式训练 I | [note/lecture-19.md](note/lecture-19.md) | [src/lecture-19/](src/lecture-19/) | - |
| 20 | 分布式训练 II | [note/lecture-20.md](note/lecture-20.md) | [src/lecture-20/](src/lecture-20/) | - |
| 21 | 端侧训练 | [note/lecture-21.md](note/lecture-21.md) | [src/lecture-21/](src/lecture-21/) | - |
| 22 | 课程总结 | [note/lecture-22.md](note/lecture-22.md) | [src/lecture-22/](src/lecture-22/) | - |
| 23 | 量子 ML | [note/lecture-23.md](note/lecture-23.md) | [src/lecture-23/](src/lecture-23/) | - |

## 从课程过渡到工业界

### 技能树映射

| 课程内容 | 工业界应用 | 对应岗位 |
|----------|-----------|----------|
| Pruning + Quantization | TensorRT 部署优化 | 推理优化工程师 |
| NAS | AutoML 平台开发 | ML Platform 工程师 |
| KD | 小模型生产化 | MLE (模型生产) |
| TinyML | MCU/嵌入式 AI | 嵌入式 AI 工程师 |
| LLM 量化 (AWQ) | LLM 本地部署 | LLM 推理工程师 |
| 分布式训练 | 大规模模型训练 | 分布式系统工程师 |
| 端侧训练 | 联邦学习/隐私计算 | 隐私 AI 工程师 |

### 推荐学习路径

```
入门 (1-2周)     → Lecture 01-02 + Lab 0
核心压缩 (2-4周) → Lecture 03-09 + Lab 1-3
系统部署 (1-2周) → Lecture 10-11
LLM专项 (2-3周)  → Lecture 12-15 + Lab 4-5
高级专题 (2-3周) → Lecture 16-21
工业项目 (2-4周) → project/edge_ai_compression_deployment
```

### 应用到实际场景

- **VLA (Vision-Language-Action)**: 使用量化+蒸馏压缩视觉编码器 → 部署到机器人端侧
- **机器人**: TinyML 模型在 MCU 上运行实时控制 → MCUNet + TinyEngine
- **端侧 AI**: 手机端运行 LLM → AWQ/W4A16 量化 + ONNX/TensorRT
- **TensorRT 部署**: pruning + quantization → ONNX export → TensorRT engine → latency benchmark

## 模型压缩实验

> 详细理论与工业实践参考：[note/model_compression_industry.md](note/model_compression_industry.md)

本项目新增统一脚本 `src/model_compression/benchmark_compression.py`，用于真实测量剪枝、量化、ONNX Runtime 和 TensorRT 可选部署路径。脚本默认使用 synthetic input，不依赖外部数据下载；没有 GPU 时自动运行 CPU baseline。

### 环境依赖

基础依赖：

```bash
pip install -r requirements.txt
```

关键测量框架：

| 框架 | 用途 |
|---|---|
| PyTorch | 参数量、推理延迟、torch.profiler、CUDA memory |
| fvcore / thop | FLOPs / MACs 统计 |
| onnxruntime | ONNX 推理延迟 |
| psutil | CPU 内存占用 |
| pandas | Markdown 表格生成 |
| TensorRT / trtexec | FP16 / INT8 engine benchmark，可选 |

TensorRT 依赖 NVIDIA Driver、CUDA、TensorRT SDK 和 `trtexec`。如果当前环境没有 TensorRT，脚本会跳过 TensorRT 并在报告中写明原因。

### 运行命令

```bash
python src/model_compression/benchmark_compression.py \
  --batch-size 8 \
  --seq-len 64 \
  --hidden-size 128 \
  --runs 30 \
  --warmup 10
```

可选 GPU：

```bash
python src/model_compression/benchmark_compression.py --device cuda --batch-size 16
```

快速 smoke test：

```bash
python src/model_compression/benchmark_compression.py --runs 3 --warmup 1 --train-steps 1
```

### 输出文件

| 文件 | 内容 |
|---|---|
| `reports/model_compression_report.md` | 压缩前后参数量、模型大小、压缩率、延迟、吞吐、显存/内存、误差对比 |
| `reports/model_compression_report.json` | 原始结构化 benchmark 数据 |
| `reports/artifacts/model_compression/*.pt` | PyTorch baseline / pruned / quantized 权重 |
| `reports/artifacts/model_compression/*.onnx` | ONNX 导出模型 |

### 覆盖案例

- SmallCNN：PyTorch 非结构化剪枝、结构化通道剪枝、动态 INT8 量化、ONNX 导出、ONNX Runtime 推理。
- Transformer AttentionBlock：Linear-heavy attention/FFN 的动态量化和低精度推理。
- VLAActionHead：面向机器人 action chunk 输出的 MLP action head 压缩，使用 output MSE 衡量动作偏移。

### 如何解读指标

- **压缩率** = baseline model size / compressed model size。压缩率高不代表一定更快，非结构化稀疏如果没有 sparse kernel，latency 可能不降反升。
- **延迟** 看 P50/P95/P99，而不是只看平均值。机器人和自动驾驶更关注 P99 是否满足控制周期。
- **吞吐** = batch size / average latency。数据中心推理看 throughput，端侧交互更看 batch=1 latency。
- **显存/内存** 决定能否提高 batch、并发数或上下文长度。LLM/VLA 场景还要额外关注 KV cache 和 action buffer。
- **精度/误差** 本脚本用 compressed output 与 FP32 baseline output 的 MSE。真实项目应替换为 task metric，例如 accuracy、mAP、perplexity、success rate 或 trajectory error。

### 工业界部署建议

1. 先建立 FP32 baseline，再逐步尝试 FP16/BF16、INT8 PTQ、剪枝、QAT、蒸馏。
2. CNN/ViT 端侧部署优先尝试结构化通道剪枝 + INT8 PTQ。
3. Transformer/LLM CPU 推理优先尝试 dynamic quantization 或 weight-only quantization；GPU 推理优先使用 TensorRT/vLLM 等 runtime。
4. VLA/机器人 action head 可以优先量化 MLP 中间层，最后 action projection 层谨慎量化，并用 action MSE、rollout success rate 和 P99 latency 一起验收。
5. TensorRT engine 必须用目标硬件真实构建和 benchmark，不能只凭 PyTorch eager latency 判断上线收益。

## 工作中注意事项

### 指标解读（别只看表面数字）

- **压缩率高不代表更快**: 非结构化稀疏如果没有 sparse kernel，latency 可能不降反升
- **延迟看 P50/P95/P99，不看平均值**: 机器人和自动驾驶更关注 P99 是否满足控制周期
- **吞吐 vs 延迟场景不同**: 数据中心推理看 throughput，端侧交互看 batch=1 latency
- **显存/内存决定上限**: 影响 batch size、并发数、上下文长度；LLM/VLA 额外关注 KV cache 和 action buffer
- **精度用 task metric 衡量**: 不要只用 MSE，替换为 accuracy、mAP、perplexity、success rate 等真实业务指标

### 部署铁律

1. 先建 FP32 baseline，再逐步尝试 FP16/BF16 → INT8 PTQ → 剪枝 → QAT → 蒸馏
2. CNN/ViT 端侧部署优先尝试结构化通道剪枝 + INT8 PTQ
3. Transformer/LLM CPU 推理优先 dynamic/weight-only quantization；GPU 推理优先 TensorRT/vLLM 等 runtime
4. VLA/机器人 action head 优先量化 MLP 中间层，action projection 层谨慎量化，用 action MSE + rollout success rate + P99 latency 一起验收
5. TensorRT engine 必须在目标硬件上真实 build + benchmark，不能只凭 PyTorch eager latency 判断上线收益

## 论文导读目录

| # | 论文 | 发表 | 与课程关联 |
|---|------|------|-----------|
| 01 | Deep Compression | ICLR 2016 | 剪枝+量化+霍夫曼编码三阶段流水线 |
| 02 | MCUNet | NeurIPS 2020 | TinyML 系统：模型-引擎协同设计 |
| 03 | Once-for-All (OFA) | ICLR 2020 | 渐进收缩：训练一次，部署多处 |
| 04 | AWQ | MLSys 2024 | 激活感知权重量化，LLM INT4 |
| 05 | SmoothQuant | ICML 2023 | 平滑异常值，W8A8 LLM 量化 |
| 06 | LoRA | ICLR 2022 | 低秩适应，高效 LLM 微调 |
| 07 | EIE | ISCA 2016 | 稀疏+压缩神经网络的专用硬件加速器 |
| 08 | Learning Weights & Connections | NIPS 2015 | 剪枝开山之作，三步法 |
| 09 | ProxylessNAS | ICLR 2019 | 直接目标硬件上 NAS，无代理 |
| 10 | HAQ | CVPR 2019 | 硬件感知自动混合精度量化 |
| 11 | MCUNetV2 | NeurIPS 2021 | 分块推理，突破 MCU 内存瓶颈 |
| 12 | MCUNetV3 (On-Device Training) | NeurIPS 2022 | 256KB MCU 上完整训练系统 |
| 13 | GAN Compression | CVPR 2020 | GAN 生成器 9-21x 压缩 |
| 14 | TSM | ICCV 2019 | 时序位移模块，零参数量视频理解 |
| 15 | StreamingLLM | ICLR 2024 | 注意力沉降，无限长流式 LLM |
| 16 | LongLoRA | ICLR 2024 | S²-Attn + LoRA，高效扩展上下文 |
| 17 | QServe | MLSys 2025 | W4A8KV4 量化+系统协同，LLM 服务 |
| 18 | EfficientViT | ICCV 2023 | 高效视觉 Transformer，线性注意力 |
| 19 | SVDQuant | ICLR 2025 | SVD 低秩吸收异常值，4-bit Diffusion |
| 20 | BEVFusion | ICRA 2023 | 多传感器融合，高效自动驾驶感知 |

> 以上 20 篇论文均来自 MIT HAN Lab (https://hanlab.mit.edu/publications)，与课程内容直接关联。

## 参考资料

- 课程官网: https://efficientml.ai
- HAN Lab: https://hanlab.mit.edu
- HAN Lab 论文集: https://hanlab.mit.edu/publications
- 参考书籍: "Efficient Deep Learning" (即将出版)
