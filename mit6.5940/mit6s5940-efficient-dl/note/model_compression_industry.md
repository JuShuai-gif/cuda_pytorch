# 模型压缩工业落地实战指南

> 从理论到生产：剪枝、量化、蒸馏、部署的完整工业级工作流

---

## 1. 概述

模型压缩不是单纯的"让模型变小"，而是在**准确率、延迟、吞吐、内存、能耗、硬件约束**之间做工程权衡。一个优秀的压缩方案需要同时回答：

- 压缩了多少？（参数量、模型大小、FLOPs）
- 快了没有？（延迟、吞吐）
- 精度掉了多少？（Accuracy / MSE / Perplexity 变化）
- 能在目标设备上跑吗？（内存/显存上限、算力约束）
- 部署成本多少？（是否需要专用硬件/推理引擎）

---

## 2. 剪枝 (Pruning)

### 2.1 非结构化剪枝 (Unstructured Pruning)

**原理**：将绝对值最小的权重逐个置零，生成稀疏权重矩阵。

**公式**：对权重矩阵 $W$，设定阈值 $\theta$：
$$\text{mask}[i,j] = \begin{cases} 1 & |W[i,j]| > \theta \\ 0 & |W[i,j]| \leq \theta \end{cases}$$

**优点**：
- 压缩率极高（可达 90%+ 稀疏度）
- 精度损失小（细粒度控制）
- 实现简单

**缺点**：
- 稀疏权重矩阵没有硬件加速优势（无稀疏 BLAS 支持时延迟可能不降反升）
- 存储格式需要 CSR/CSC 等稀疏格式，增加开销
- 需要专用稀疏 kernel（如 NVIDIA cuSPARSE、Intel MKL Sparse BLAS）

**工业实践**：
- 适用于模型存储受限场景（如移动端模型分发）
- 需要配合稀疏推理引擎（如 DeepSparse、Neural Magic）
- 参考：Deep Compression (Han et al., ICLR 2016), Lottery Ticket Hypothesis (Frankle & Carbin, ICLR 2019)

### 2.2 结构化剪枝 (Structured Pruning)

**原理**：移除整个结构单元（神经元、通道、注意力头），保持权重矩阵密集。

**方法分类**：

| 粒度 | 移除单元 | 硬件友好度 | 精度影响 |
|------|----------|----------|---------|
| 权重级 | 单个权重 | 低 | 低 |
| 向量级 | 行/列 | 中 | 中 |
| 核级 | 2D卷积核 | 中 | 中 |
| 通道级 | 整个输出通道 | 高 | 较高 |

**优点**：
- 硬件友好：剪枝后依然是密集矩阵，标准 GEMM 可加速
- 直接减少计算量（FLOPs）和参数量
- 无需特殊硬件/软件支持

**缺点**：
- 压缩率不如非结构化剪枝
- 通道剪枝需要调整网络结构（修改后续层输入通道数）

**工业实践**：
- CNN 推理优化首选结构化通道剪枝
- 配合 TensorRT/ONNX Runtime 可直接获得推理加速
- 参考：Learning Efficient Convolutional Networks through Network Slimming (Liu et al., ICCV 2017)

### 2.3 通道剪枝 (Channel Pruning)

**重要性度量**：

| 度量标准 | 公式 | 适用场景 |
|---------|------|---------|
| L1 范数 | $\|W_{c,:,:,:}\|_1$ | 通用 |
| L2/Frobenius 范数 | $\|W_{c,:,:,:}\|_2$ | 对大权重更敏感 |
| BN $\gamma$ | $\|\gamma_c\|$ | 配合 BatchNorm |
| 梯度敏感度 | $\mathbb{E}\|\frac{\partial \mathcal{L}}{\partial W_c}\|$ | 精度优先 |

**流程**：
1. 计算每个通道的重要性分数
2. 按分数排序，保留 top-k 通道
3. 剪枝当前层输出通道
4. 调整下一层输入通道（通道对齐）
5. 微调恢复精度

### 2.4 权重稀疏化 (Weight Sparsification)

**稀疏模式分类**：

| 模式 | 描述 | 硬件支持 |
|------|------|---------|
| 非结构化 | 任意位置稀疏 | cuSPARSE, SparseML |
| 2:4 结构化 | 每 4 个连续权重保留 2 个 | NVIDIA Ampere+ Sparse Tensor Core |
| 块稀疏 | 固定大小的块内稀疏 | CPU Sparse BLAS |
| N:M 稀疏 | 每 M 个元素保留 N 个 | VNNI (Intel Sapphire Rapids) |

**关键决策树**：
```
需要硬件加速？ → Yes → 结构化剪枝 / 2:4稀疏
               → No  → 非结构化剪枝（最大压缩）
```

---

## 3. 量化 (Quantization)

### 3.1 精度格式对比

| 格式 | 比特数 | 符号位 | 指数位 | 尾数位 | 动态范围 | 典型场景 |
|------|--------|--------|--------|--------|---------|---------|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | 训练 |
| FP16 | 16 | 1 | 5 | 10 | ±65504 | 推理(有溢出风险) |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | 训练/推理(安全) |
| INT8 | 8 | 1 | - | 7 | ±127 | PTQ 推理 |
| INT4 | 4 | 1 | - | 3 | ±7 | 权重量化(LLM) |
| FP8 | 8 | 1 | 4/5 | 3/2 | ±448 / ±57344 | H100+ 推理 |

### 3.2 线性量化公式

**对称量化**：
$$q = \text{round}\left(\frac{r}{S}\right)$$
$$S = \frac{\max|r|}{2^{b-1} - 1}$$

**非对称量化**：
$$q = \text{round}\left(\frac{r - r_{min}}{S}\right) + Z$$
$$S = \frac{r_{max} - r_{min}}{2^b - 1}$$

其中 $r$ 为浮点值，$q$ 为量化整数值，$S$ 为 scale（缩放因子），$Z$ 为零点。

### 3.3 量化粒度

| 粒度 | Scale 数 | 精度 | 开销 |
|------|---------|------|------|
| Per-Tensor | 1 tensor 1个 | 较低 | 最低 |
| Per-Channel | 每个输出通道 1 个 | 较高 | 较低 |
| Per-Group | 每 N 个元素 1 个 | 高 | 中等 |
| Per-Token (激活) | 每个 token 1 个 | 最高 | 较高 |

**工业实践**：CNN 权重用 Per-Channel，LLM 权重用 Per-Group (Group=128)。

### 3.4 PTQ vs QAT

| 方法 | 需要训练 | 需要校准数据 | 精度 | 部署成本 |
|------|---------|------------|------|---------|
| PTQ (Post-Training Quantization) | 否 | 少量(100-1000样本) | 中 | 极低 |
| QAT (Quantization-Aware Training) | 是 | 需要全量训练数据 | 高 | 高 |

**选择建议**：
- INT8 通常 PTQ 足够
- INT4 建议 QAT（尤其 CNN 低比特）
- LLM 可用 GPTQ/AWQ 等专用 PTQ 方法实现 W4A16

### 3.5 工业界量化方案

| 方案 | 精度 | 场景 | 工具 |
|------|------|------|------|
| W8A8 PTQ | INT8 权重 + INT8 激活 | CNN/ViT 端侧 | TensorRT, ONNX Runtime |
| W4A16 | INT4 权重 + FP16 激活 | LLM 部署 | AWQ, GPTQ, llama.cpp |
| W8A8 QAT | INT8 + FakeQuant 训练 | 精度敏感场景 | PyTorch QAT, TensorRT QAT |
| Dynamic Quant | INT8 权重 + FP32 激活 | Transformer CPU 推理 | torch.quantize_dynamic |

---

## 4. 蒸馏 (Knowledge Distillation)

### 4.1 Teacher-Student 框架

**基本原理**：用大模型（Teacher）的"软标签"指导小模型（Student）训练。

**损失函数**：
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{KL}(p_s^T, p_t^T) + (1-\alpha) \cdot \mathcal{L}_{CE}(p_s, y_{true})$$

其中 $T$ 为温度参数，温度越高 Teacher 输出越"软"（类间差异缩小）：
$$p_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### 4.2 Logits Distillation

**仅用 Teacher 输出的 logits**：
- 最简单、应用最广泛
- 适合分类任务
- 学生模型学习 Teacher 的"不确定性边界"

### 4.3 Feature Distillation

**对齐中间层特征**：
- FitNet: 使用 Teacher 中间层 feature map 作为回归目标
- Attention Transfer: 对齐注意力图
- 适合同构/异构架构间的知识迁移

**损失**：
$$\mathcal{L}_{feat} = \|F_T - \phi(F_S)\|_2^2$$

其中 $\phi$ 为适配层（当 Teacher/Student 特征维度不匹配时）。

### 4.4 工业实践

| 场景 | 蒸馏策略 |
|------|---------|
| BERT → TinyBERT | Hidden states + Attention + Logits 三层蒸馏 |
| ViT-L → ViT-S | Feature + Logits 蒸馏 |
| LLM (GPT-4 → GPT-3.5) | 数据飞轮 + Logits 蒸馏 |
| VLA 视觉编码器 | Feature 蒸馏压缩视觉 backbone |

---

## 5. 部署 (Deployment)

### 5.1 ONNX (Open Neural Network Exchange)

**主要用途**：跨框架模型交换格式

**工作流**：
```
PyTorch Model → torch.onnx.export() → ONNX → onnxruntime / TensorRT / OpenVINO
```

**最佳实践**：
- 使用 `opset_version >= 13`（支持更多算子）
- 启用 `do_constant_folding=True`（静态图折叠优化）
- 设置 `dynamic_axes` 支持动态 batch size
- 导出后用 `onnx.checker.check_model()` 验证

**性能**：onnxruntime 通常比 PyTorch eager 快 1.2-2x (CPU)，配合 INT8 量化可达 3-5x。

### 5.2 TensorRT

**NVIDIA 的推理优化器**：图优化 + 内核自动调优 + 精度降级

**工作流**：
```
ONNX → trtexec → TensorRT Engine (.engine / .plan)
```
或
```
PyTorch → torch2trt / Torch-TensorRT → TensorRT Engine
```

**优化技术**：
| 技术 | 效果 |
|------|------|
| Layer Fusion | 合并 Conv+BN+ReLU 为单 kernel |
| Kernel Auto-Tuning | 针对具体 GPU 架构选择最优实现 |
| Precision Calibration | FP16/INT8 精度校准 |
| Dynamic Shapes | 支持可变输入尺寸（性能略有损失） |
| Memory Planning | 预分配+复用显存，减少碎片 |

**INT8 校准要求**：需要真实校准数据集（通常 500-1000 样本），校准过程收集每一层的 min/max 确定量化范围。

### 5.3 TorchScript

**PyTorch 原生序列化格式**：
- `torch.jit.script`: 代码级转换（支持控制流）
- `torch.jit.trace`: 追踪转换（仅支持静态图）

**适用场景**：
- PyTorch Serve 生产部署
- C++ 环境 (libtorch) 推理
- 移动端 (PyTorch Mobile)

### 5.4 OpenVINO

**Intel 的推理优化框架**：
- 针对 Intel CPU/iGPU/VPU 优化
- 支持 ONNX / TensorFlow / PaddlePaddle 模型导入
- INT8 校准工具链完整

**适用场景**：Intel 平台（Xeon, Core, Atom, Arc GPU）的边缘推理。

### 5.5 llama.cpp / GGUF

**LLM CPU 推理的事实标准**：
- GGUF 格式：量化的 LLM 权重格式
- 支持 Q4_0, Q4_K_M, Q5_K_M, Q8_0 等多种量化类型
- 纯 CPU 推理，内存映射加载

**量化类型**（llama.cpp）：
| 类型 | 每参数比特 | 质量 |
|------|-----------|------|
| Q4_0 | 4.5 bits | 低 |
| Q4_K_M | 4.8 bits | 中(推荐) |
| Q5_K_M | 5.5 bits | 较高 |
| Q8_0 | 8.5 bits | 高(几乎无损) |

---

## 6. 评估指标 (Metrics)

### 6.1 模型效率指标

| 指标 | 含义 | 工具 | 单位 |
|------|------|------|------|
| 参数量 | 可学习参数总数 | `sum(p.numel())` | M (百万) |
| 模型大小 | 磁盘存储大小 | `os.path.getsize()` | MB |
| FLOPs | 单次推理浮点运算数 | fvcore / thop / 手动计算 | MFLOPs / GFLOPs |
| MACs | 乘加运算数 (约 FLOPs/2) | fvcore | MMACs |
| 显存占用 | GPU 推理峰值显存 | `torch.cuda.max_memory_allocated()` | MB |
| 内存占用 | CPU 推理峰值内存 | psutil / 参数估算 | MB |

### 6.2 推理性能指标

| 指标 | 含义 | 关注场景 |
|------|------|---------|
| 延迟 (Latency) | 单次推理时间 | 实时交互、机器人、自动驾驶 |
| P50/P95/P99 延迟 | 延迟分位数 | SLA 保障、尾延迟分析 |
| 吞吐 (Throughput) | 单位时间处理样本数 | 数据中心、批处理 |
| 能效 | 每焦耳处理样本数 | 移动端、嵌入式 |

### 6.3 精度指标

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| Top-1 Accuracy | 分类准确率 | 图像分类 |
| MSE / MAE | 输出误差 | 回归、压缩质量评估 |
| Cosine Similarity | 输出方向一致性 | embedding 质量 |
| Perplexity | 语言模型困惑度 | LLM |
| Rollout Success Rate | 机器人任务成功率 | VLA/机器人 |

### 6.4 压缩率计算

$$\text{压缩率} = \frac{\text{原始模型大小 (MB)}}{\text{压缩后模型大小 (MB)}}$$

$$\text{参数压缩比} = \frac{\text{原始参数量}}{\text{压缩后参数量}}$$

$$\text{推理加速比} = \frac{\text{原始延迟 (ms)}}{\text{压缩后延迟 (ms)}}$$

---

## 7. 端侧部署决策框架

### 7.1 设备约束分级

| 设备等级 | 典型设备 | 算力 | 内存 | 推荐方案 |
|---------|---------|------|------|---------|
| MCU (微控制器) | Arduino, STM32 | <1 GOPs | <1 MB | TinyML, MCUNet, TF Lite Micro |
| 嵌入式 | Raspberry Pi, Jetson Nano | 1-100 GOPs | 1-8 GB | ONNX Runtime + INT8 |
| 移动端 | 手机 SoC (A17, SD8Gen3) | 1-35 TOPS | 2-8 GB | Core ML / TFLite / QNN |
| 边缘服务器 | Jetson Orin, 工控机 | 10-275 TOPS | 8-64 GB | TensorRT FP16/INT8 |
| 桌面/服务器 | RTX 4090, A100 | 100-1000+ TFLOPS | 24-80 GB | TensorRT / vLLM |

### 7.2 部署选择决策树

```
1. 目标设备是什么？
   - MCU (<1MB) → TinyML, TFLite Micro
   - 嵌入式/移动 → ONNX Runtime + INT8 / TFLite
   - GPU 边缘 → TensorRT FP16/INT8
   - GPU 服务器 → TensorRT / vLLM

2. 模型类型是什么？
   - CNN → 通道剪枝 + INT8 PTQ → ONNX/TensorRT
   - ViT → 结构化剪枝 + INT8 PTQ → TensorRT
   - LLM → GPTQ/AWQ INT4 → llama.cpp / vLLM
   - VLA Action Head → INT8 PTQ → ONNX Runtime

3. 精度要求？
   - 精度敏感 (≤0.5%) → QAT + 低剪枝率
   - 精度容忍 (1-3%) → PTQ + 中剪枝率
   - 极致压缩 (3-10%) → 非结构化剪枝 + INT4

4. 延迟要求？
   - <1ms (实时控制) → 结构化压缩 + TensorRT + GPU
   - <10ms (交互式) → INT8 PTQ + ONNX Runtime + CPU/GPU
   - <100ms (离线推理) → 任何可行方案
```

---

## 8. 参考工程实践

### 8.1 本项目代码示例

```
src/model_compression/
├── __init__.py
├── benchmark_compression.py   # 统一压缩基准测试脚本
├── models.py                  # SmallCNN, Transformer, VLAActionHead
└── metrics.py                 # 参数量/延迟/内存/FLOPs/吞吐测量工具
```

运行方式：
```bash
python src/model_compression/benchmark_compression.py
```

### 8.2 关键参考论文

| 论文 | 技术 | 发表 | 实践价值 |
|------|------|------|---------|
| Deep Compression | 剪枝+量化+霍夫曼编码 | ICLR 2016 | 三阶段压缩流水线原型 |
| MCUNet | TinyNAS + TinyEngine | NeurIPS 2020 | MCU 端侧推理系统 |
| Once-for-All (OFA) | 渐进式 NAS | ICLR 2020 | 一次训练多设备部署 |
| AWQ | 激活感知权重量化 | MLSys 2024 | LLM INT4 量化 |
| SmoothQuant | W8A8 平滑量化 | ICML 2023 | LLM INT8 量化 |
| GPTQ | 逐层权重量化 | ICLR 2023 | LLM 高效 INT4 |
| 2:4 Sparsity | NVIDIA 结构化稀疏 | NVIDIA 2020 | Ampere+ 硬件加速 |

---

## 9. 常见坑与对策

| 陷阱 | 现象 | 对策 |
|------|------|------|
| 非结构化剪枝不加速 | 稀疏度高但延迟不变 | 使用结构化剪枝或用稀疏推理引擎 |
| INT8 精度崩塌 | 某几层量化后精度暴跌 | 敏感层回退 FP16，使用混合精度 |
| ONNX 算子不支持 | 导出报错 | 降低 opset，替换不支持的算子 |
| TensorRT 校准数据不匹配 | INT8 精度差 | 确保校准数据与推理数据同分布 |
| BatchNorm 折叠错误 | 精度异常 | 导出前 fuse Conv+BN |
| 动态轴性能退化 | 推理变慢 | 固定 batch size 或用 profile 最优 shape |

---

## 10. 开源工具生态 (Open-Source Tooling Ecosystem)

### 10.1 FLOPs / MACs 计算库

| 库 | 安装 | 特点 | 精度 | 局限 |
|----|------|------|------|------|
| **fvcore** | `pip install fvcore` | Meta 出品，基于 PyTorch JIT trace | 高 | 不支持动态控制流 |
| **thop** | `pip install thop` | 轻量，<1000 行代码 | 中 | 更新慢，部分算子缺失 |
| **ptflops** | `pip install ptflops` | 社区活跃，支持自定义算子 | 高 | 需要 pip 安装 |
| **calflops** | `pip install calflops` | 支持 HuggingFace Transformers | 高 | 较新，社区较小 |
| **torchprofile** | `pip install torchprofile` | 基于 forward hook | 高 | 对 inplace 算子敏感 |
| **torchstat** | `pip install torchstat` | 类似 torchsummary + FLOPs | 中 | 维护不活跃 |
| **torchinfo** | `pip install torchinfo` | model summary + 参数量(非FLOPs) | - | 仅参数量/内存估算 |

**选用建议**：
- 学术论文：fvcore (被 ICCV/CVPR 广泛接受)
- 快速原型：thop (一行代码)
- HuggingFace 模型：calflops
- 需要 tensor-level 追踪：torchprofile

### 10.2 模型结构分析/可视化

| 工具 | 安装 | 用途 |
|------|------|------|
| **torchinfo** | `pip install torchinfo` | 参数量/每层输出shape/内存估算 |
| **torchsummary** | `pip install torchsummary` | Keras-style model summary |
| **netron** | `pip install netron` | 交互式模型结构可视化 (支持 ONNX/PyTorch/TF) |
| **tensorboard** | `pip install tensorboard` | PyTorch/TF 训练曲线 + 计算图 |
| **torchview** | `pip install torchview` | PyTorch 计算图可视化 |

### 10.3 模型压缩/量化工具链

| 工具 | 组织 | 功能 | 适用场景 |
|------|------|------|---------|
| **torch.ao.quantization** | Meta | 原生 PTQ/QAT | PyTorch 内置首选 |
| **pytorch-quantization** | NVIDIA | TensorRT 对齐的量化工具包 | NVIDIA GPU 部署 |
| **nvidia-modelopt** | NVIDIA | 剪枝+量化+蒸馏统一工具 | NVIDIA 生态 |
| **AIMET** | Qualcomm | PTQ/QAT + 压缩率分析 | Snapdragon / Qualcomm 平台 |
| **Intel Neural Compressor** | Intel | 自动混合精度量化/剪枝/蒸馏 | Intel CPU/GPU |
| **SparseML** | Neural Magic | 稀疏训练+SparseZoo | 通用 CPU/GPU 稀疏推理 |
| **llm-compressor** | Neural Magic | LLM 专用压缩 (GPTQ/SparseGPT) | LLM 量化 |
| **bitsandbytes** | Tim Dettmers | QLoRA/LoRA + 8-bit 量化 | LLM 微调/推理 |
| **AutoGPTQ** | PanQiWei | GPTQ INT4 量化 | LLM 量化 |
| **AutoAWQ** | MIT HAN Lab | AWQ INT4 量化 | LLM 量化 (推荐) |
| **llama.cpp** | ggerganov | GGML/GGUF 量化+CPU推理 | LLM CPU 部署 |

### 10.4 推理引擎

| 引擎 | 平台 | 精度 | 延迟优化 | 吞吐优化 |
|------|------|------|---------|---------|
| **PyTorch eager** | CPU/GPU | FP32/FP16 | 低 | 低 |
| **torch.compile** | CPU/GPU | FP32/FP16/BF16 | 中 (1.2-2x) | 中 |
| **ONNX Runtime** | CPU/GPU/Edge | FP32/FP16/INT8 | 中 (1.5-3x) | 中 |
| **TensorRT** | NVIDIA GPU | FP32/FP16/INT8/FP8 | 高 (2-5x) | 高 |
| **OpenVINO** | Intel CPU/GPU/VPU | FP32/FP16/INT8 | 中-高 | 中 |
| **TensorFlow Lite** | 移动端/MCU | FP32/FP16/INT8 | 高 (移动端) | 低 |
| **ExecuTorch** | 移动端/嵌入式 | FP32/INT8 | 中 (新) | 低 |
| **vLLM** | NVIDIA GPU | FP16/INT4(W4A16) | 低 (throughput oriented) | 极高 (PagedAttention) |
| **MNN** (阿里) | 移动端/嵌入式 | FP32/FP16/INT8 | 高 | 中 |
| **NCNN** (腾讯) | 移动端/嵌入式 | FP32/FP16/INT8 | 极高 | 中 |
| **TNN** (腾讯) | 移动端/GPU | FP32/FP16 | 高 | 高 |

### 10.5 Profiling 工具

| 工具 | 用途 | 输出 |
|------|------|------|
| **torch.profiler** | PyTorch 原生 kernel-level profiling | Chrome trace + FLOPs + 显存时间线 |
| **torch.utils.benchmark** | 微基准测试 (比 time.perf_counter 更可靠) | 延迟统计 (mean/median/IQR) |
| **torch.autograd.profiler** | 算子级耗时 + FLOPs | 表格/JSON |
| **nvidia-smi** | GPU 利用率/显存/温度/功耗 | 实时 CLI |
| **pynvml** (nvidia-ml-py) | nvidia-smi 的 Python 接口 | GPU 指标 API |
| **py-spy** | CPU 采样 profiler (无需 instrumentation) | flamegraph |
| **viztracer** | 多线程 Python trace | Perfetto/Chrome trace |
| **snakeviz** | cProfile 可视化 | sunburst / icicle 图 |
| **line_profiler** | 逐行性能分析 | 每行耗时统计 |
| **memory_profiler** | Python 内存分析 | 每行内存变化 |

### 10.6 实验管理与追踪

| 工具 | 安装 | 用途 |
|------|------|------|
| **TensorBoard** | `pip install tensorboard` | 标量/直方图/图/embedding 可视化 |
| **Weights & Biases (wandb)** | `pip install wandb` | 实验追踪 + 超参搜索 + 模型注册 |
| **MLflow** | `pip install mlflow` | 开源实验管理 + 模型注册 |
| **Neptune.ai** | `pip install neptune` | 实验追踪 (协作友好) |
| **Aim** | `pip install aim` | 开源实验追踪 (类似 wandb) |

### 10.7 GPU / 硬件监控

| 工具 | 用途 | 安装 |
|------|------|------|
| **nvidia-smi** | NVIDIA GPU 状态 (利用/显存/温度/功耗) | 随 NVIDIA 驱动安装 |
| **pynvml** | nvidia-smi Python API | `pip install nvidia-ml-py` |
| **gpustat** | GPU 状态一行展示 | `pip install gpustat` |
| **nvtop** | GPU 类 htop TUI | `apt install nvtop` |
| **dcgm** | NVIDIA Data Center GPU Manager | NVIDIA DCGM 包 |
| **rocm-smi** | AMD GPU 状态 | 随 ROCm 安装 |
| **intel_gpu_top** | Intel GPU 状态 | `apt install intel-gpu-tools` |
| **perf** | Linux CPU 性能计数器 | 系统自带 |

### 10.8 一键对比：同一指标、不同工具

下表汇总**计算 FLOPs** 的多种库在同一模型上的调用方式：

```python
import torch
from models import SmallCNN

model = SmallCNN()
dummy = torch.randn(1, 3, 32, 32)

# ---- fvcore ----
from fvcore.nn import FlopCountAnalysis
flops_fvcore = FlopCountAnalysis(model, dummy).total()

# ---- thop ----
from thop import profile
flops_thop, params_thop = profile(model, inputs=(dummy,))

# ---- torchprofile ----
from torchprofile import profile_macs
flops_tp = profile_macs(model, dummy)

# ---- ptflops ----
from ptflops import get_model_complexity_info
macs_ptflops, params_ptflops = get_model_complexity_info(
    model, (3, 32, 32), as_strings=False, print_per_layer_stat=False
)

# ---- calflops ----
from calflops import calculate_flops
flops_cal, params_cal, _ = calculate_flops(
    model, input_shape=(1, 3, 32, 32), output_as_string=False
)

# ---- torchinfo (仅参数量/内存, 非FLOPs) ----
from torchinfo import summary
summary(model, input_size=(1, 3, 32, 32))
```

> **注意**：不同库对 BatchNorm、ReLU、MaxPool、bias 等算子的计数规则不同，计算结果可能偏差 5-15%。学术论文应注明使用的库和版本。

### 10.9 压缩效果检测清单 (What to Measure Before & After)

每次执行模型压缩后，应至少检测以下维度：

| 维度 | 检测方法 | 推荐工具 |
|------|---------|---------|
| 参数量变化 | 统计非零参数 / 总参数 | `sum(p.numel())`, torchinfo |
| 模型文件大小 | 保存后量磁盘大小 | `os.path.getsize()`, ls -lh |
| FLOPs 变化 | 前向推理运算量 | fvcore / thop / torchprofile |
| 推理延迟 | batch=1 P50/P95/P99 | `torch.utils.benchmark`, 自写 timer |
| 推理吞吐 | 多 batch 吞吐 (samples/sec) | `torch.utils.benchmark` |
| 显存占用 | GPU peak memory | `torch.cuda.max_memory_allocated()`, pynvml |
| CPU 内存 | RSS / VMS | psutil |
| 精度损失 | Accuracy / MSE / Perplexity | task-specific eval |
| 输出一致性 | output MSE / KL divergence / cosine similarity | 自写 evaluator |
| GPU 利用率 | SM utilization / Tensor Core usage | nvidia-smi / torch.profiler |
| 功耗 | GPU/CPU power draw | pynvml / nvidia-smi |
| 能耗比 | samples / Joule | 综合计算 |

---

*本笔记为 MIT 6.5940 高效深度学习课程配套工业实战指南*
