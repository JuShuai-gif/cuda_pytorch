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

## 目录结构

```
mit6s5940-efficient-dl/
├── note/           # 中文讲解笔记 (按 lecture 一一对应)
├── src/            # 可运行的 PyTorch 代码示例
├── labs/           # 重构的实验 (含 starter/solution/report)
├── papers/         # 论文中文导读
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

## 参考资料

- 课程官网: https://efficientml.ai
- HAN Lab: https://hanlab.mit.edu
- 参考书籍: "Efficient Deep Learning" (即将出版)
