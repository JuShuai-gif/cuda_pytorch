# MCUNet 学习计划：每天 45 分钟，从入门到理解核心

**一句话定位**：MCUNet 是 MIT HAN Lab 提出的系统-算法协同设计框架，通过 TinyNAS（两阶段神经架构搜索）+ TinyEngine（内存高效推理引擎）将深度学习部署到内存极度受限的 MCU 上。V1(推理)、V2(patch 推理)、V3(端侧训练)。NeurIPS 2020 Spotlight。

**总时长**：约 4 周（20 个学习日），每天 45 分钟。

**重要性说明**：
- ⭐⭐⭐⭐⭐ = 必须掌握（不然后续无法理解）
- ⭐⭐⭐⭐   = 核心理解（面试/开发中常见）
- ⭐⭐⭐     = 重要但可先走读（用到再细看）
- ⭐⭐       = 了解即可（高级/特定场景）

---

## 第 1 周：项目全貌 + TinyNAS 搜索空间（5 天）

### Day 1：项目骨架 + 模型总览 ⭐⭐⭐⭐⭐

**目标**：知道项目做什么，有哪些模型，怎么用

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 通读 `README.md` | ⭐⭐⭐⭐⭐ | 项目定位（MCU 端深度学习）、V1/V2/V3 三代演进；性能对比表（准确率 vs SRAM/Flash） | 所有后续阅读的上下文 |
| 20 min | 读 `mcunet/model_zoo.py` | ⭐⭐⭐⭐⭐ | 7 个预训练模型定义（4 个 ImageNet + 3 个 VWW + 1 个行人检测）；`build_mcunet_model()`` 统一加载接口 | Day 16 跑 demo 时直接复用 |
| 10 min | 对比模型规模 | ⭐⭐⭐⭐ | 从 0.06M 参数/150KB SRAM 到 0.74M 参数/500KB SRAM；理解 MCU 内存预算 | Day 6 理解搜索空间范围 |

**产出**：能说出 7 个预训练模型的名字和各自的内存/精度

### Day 2：TinyNAS 架构概览 ⭐⭐⭐⭐⭐

**目标**：理解两阶段 NAS 的整体设计

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 浏览 `mcunet/tinynas/` 目录结构 | ⭐⭐⭐⭐⭐ | elastic_nn（搜索空间）+ nn/networks（网络定义）+ tf_codebase（TFLite 转换） | Day 3-5 逐层深入 |
| 15 min | 读 `mcunet/tinynas/nn/networks/proxyless_nets.py` | ⭐⭐⭐⭐⭐ | MCUNet 主干网络类 `ProxylessNASNets`；弹性深度/宽度/核大小的配置字典 | Day 4 理解动态层如何实现 |
| 15 min | 看 `mcunet/tinynas/nn/networks/mobilenet_v2.py` | ⭐⭐⭐ | MobileNetV2 作为搜索空间基线：inverted residual block 结构 | Day 3 理解搜索空间单元 |

**产出**：能画出 MCUNet 网络结构的宏观层次（stem → blocks → head）

### Day 3：弹性神经网络模块 ⭐⭐⭐⭐

**目标**：理解搜索空间的核心抽象——弹性模块

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `mcunet/tinynas/elastic_nn/modules/dynamic_layers.py` | ⭐⭐⭐⭐ | `DynamicMBConvLayer`：弹性逐点卷积 + 弹性深度卷积；`DynamicLinearLayer`：弹性全连接层 | Day 5 理解 NAS 搜索过程 |
| 15 min | 读 `mcunet/tinynas/elastic_nn/modules/dynamic_op.py` | ⭐⭐⭐⭐ | `DynamicConv2d` / `DynamicBatchNorm2d`：弹性通道数（最大通道的子集）；`DynamicSE`：弹性 SE 模块 | 搜索时如何"切片"通道 |
| 10 min | 理解"弹性"的含义 | ⭐⭐⭐ | 一个模块可以配置成多种不同参数（不同通道数、核大小），从最大配置中 "切" 出子配置 | Once-for-All 的前身 |

**产出**：能解释弹性通道数是怎么实现的（取权重的 [0:out_ch] 子集）

### Day 4：弹性网络训练 ✨ ⭐⭐⭐⭐

**目标**：理解弹性网络如何在多种配置下训练

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 看弹性网络训练逻辑 | ⭐⭐⭐⭐ | 每个训练步骤随机采样一个子配置；前向只用采样的通道/层；渐进式收缩（大→小） | Day 8 OFA 的渐进收缩思路来源 |
| 15 min | 读 `mcunet/tinynas/elastic_nn/networks/` | ⭐⭐⭐ | 弹性网络封装：`set_active_subnet()` 切换到不同配置 | Day 10 理解如何评估子网络 |
| 10 min | 理解"渐进式收缩" | ⭐⭐⭐ | 先训练最大网络→逐渐缩小→最终支持所有子配置 | 弹性训练的核心技巧 |

**产出**：能解释"先大后小"的渐进式收缩训练策略

### Day 5：两阶段 NAS 搜索 ⭐⭐⭐⭐

**目标**：理解从搜索到最终模型的全流程

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 理解两阶段搜索 | ⭐⭐⭐⭐ | 阶段1：在弹性网络上进化搜索→得到最优配置；阶段2：从头训练最优配置→最终模型 | Day 6 理解"为什么不在弹性网络上直接用" |
| 15 min | 搜索目标函数 | ⭐⭐⭐ | 最大化准确率，满足 SRAM/Flash 约束；内存模型通过编译器的内存调度分析得到 | Day 6 理解 SRAM 估算 |
| 10 min | 进化算法 | ⭐⭐⭐ | 突变（改通道数/深度/核大小）+ 交叉 + 选择；用准确率预测器加速 | Once-for-All 的准确率预测器（Day 8） |

**产出**：能说出两阶段 NAS 为什么需要"先搜索再重训"

---

## 第 2 周：协同设计 + TinyEngine 整合（5 天）

### Day 6：系统-算法协同设计 ⭐⭐⭐⭐⭐

**目标**：理解 MCUNet 的核心思想——架构搜索和推理引擎的协同

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 理解协同设计概念 | ⭐⭐⭐⭐⭐ | 不是先设计架构再适配硬件，而是搜索时就用真实的内存模型约束；TinyNAS 搜索时调用 TinyEngine 的内存分析 | 面试必问的核心思想 |
| 15 min | 看 `mcunet/tinyengine/README.md` | ⭐⭐⭐⭐ | 指向独立 TinyEngine 仓库；理解推理引擎的接口 | tinyengine 项目的入门 |
| 15 min | SRAM/Flash 约束建模 | ⭐⭐⭐⭐ | Flash 存权重（只读），SRAM 存激活+临时 buffer；TinyEngine 的 first-fit 调度决定 SRAM 峰值 | Day 7 理解调度如何影响 NAS |

**产出**：能解释 NAS 搜索时如何知道一个架构的内存用量

### Day 7：TinyEngine 集成 ⭐⭐⭐⭐

**目标**：理解 TinyEngine 在 MCUNet 中的角色

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 看 TinyEngine 调用接口 | ⭐⭐⭐⭐ | TFLite 模型 → TinyEngine Python 编译前端 → C 代码生成 → 编译部署 | 和 tinyengine 项目对照理解 |
| 15 min | 看 `mcunet/tinynas/tf_codebase/generate_tflite.py` | ⭐⭐⭐ | PyTorch 模型 → INT8 量化 → TFLite 导出；训练后量化（Post-Training Quantization） | Day 16 理解部署流水线 |
| 15 min | 理解 Patch 推理（MCUNetV2） | ⭐⭐⭐ | 将前几层在空间上切成 patch 依次推理 → 降低中间特征图的 SRAM 峰值 | V2 的核心贡献 |

**产出**：能画出从 PyTorch 训练到 MCU 运行的完整流水线

### Day 8：MCUNet V2 — Patch 推理 ⭐⭐⭐

**目标**：理解 patch-based 推理降低内存峰值的原理

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 理解 patch 推理原理 | ⭐⭐⭐ | 特征图 H×W → (H/n)×(W/n) 个 patch；每个 patch 独立推理前几层 → 在中间层合并 → 继续 | 直观解释：把大图切成小块分批处理 |
| 15 min | 理解 patch 边界处理 | ⭐⭐⭐ | patch 重叠（overlap）避免边缘 artifact；patch 间信息不交叉 | 空间局部性假设 |
| 10 min | patch 数量 vs 精度 vs 内存 | ⭐⭐ | n_patches 越多 → 内存越低 → 但边界效应导致精度轻微下降 | V2 论文的核心 trade-off |

**产出**：能解释为什么 patch 推理能在几乎不损失精度的情况下减少 40%+ 的内存

### Day 9：模型评估流程 ⭐⭐⭐

**目标**：理解如何评估模型的准确率和效率

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `eval_torch.py` | ⭐⭐⭐ | PyTorch FP32 模型评估：ImageNet top-1/top-5；`build_mcunet_model()` 加载 + 推理 | Day 16 评估自己训练的模型 |
| 15 min | 读 `eval_tflite.py` | ⭐⭐⭐ | TFLite INT8 模型评估：量化后精度验证；对比 FP32 vs INT8 精度损失 | Day 17 精度对比 |
| 10 min | 看 `eval_det.py` | ⭐⭐ | 行人检测评估 + 可视化 | 了解即可 |
| 5 min | 看 `mcunet/utils/pytorch_utils.py` | ⭐⭐ | 辅助：`replace_bn_with_conv()` 等 BN 融合 | 导出 TFLite 前的必要步骤 |

**产出**：能跑出 ImageNet 模型在 PC 端的 FP32 和 INT8 准确率

### Day 10：自定义模块 + BN 融合 ⭐⭐⭐

**目标**：理解 PyTorch 侧的特殊模块

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `mcunet/utils/pytorch_modules.py` | ⭐⭐⭐ | `MyConv` / `MyLinear` 等自定义模块：在 forward 中处理 INT8 量化模拟 | Day 9 评估时的模块 |
| 15 min | 读 `mcunet/utils/bn_utils.py` | ⭐⭐⭐ | BN 融合：`Conv+BN → 融合后的 Conv`；训练后融合到权重里 | TFLite 导出前必做 |
| 15 min | 理解 INT8 推理模拟 | ⭐⭐⭐ | 训练后量化(PTQ)：用少量校准数据统计 scale/zero_point，然后量化 | 和 AWQ 的 activation-aware 对比 |

**产出**：能解释为什么 TFLite 导出前要做 BN 融合

---

## 第 3 周：MCUNetV3 + 持续学习 + 训练（5 天）

### Day 11：MCUNet V3 — 端侧训练 ⭐⭐⭐

**目标**：理解如何在 MCU 上做训练（不只是推理）

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 V3 相关说明 | ⭐⭐⭐ | MCUNetV3 在 <256KB SRAM 的 MCU 上做训练；QAS（量化感知缩放）+ 稀疏更新 + 编译时自动微分 | tiny-training 项目的前置 |
| 15 min | 理解训练 vs 推理的内存挑战 | ⭐⭐⭐ | 训练需要存：前向激活 + 梯度 + 优化器状态，内存是推理的 4-8 倍 | 为什么 MCU 训练如此难 |
| 10 min | 理解"只有 256KB"的含义 | ⭐⭐⭐ | 256KB = PyTorch 训练 1 张 ImageNet 图片所需内存的 1/1000 | 直观感受内存差距 |

**产出**：能说出 MCU 训练的三个核心技术（QAS、稀疏更新、编译时 AD）

### Day 12：模型架构深入 — Inverted Residual Block ⭐⭐⭐

**目标**：理解 MCUNet 使用的基础计算单元

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 理解 Inverted Residual | ⭐⭐⭐ | 1×1 expand → 3×3 depthwise → 1×1 project；瓶颈结构（中间宽两头窄） | 面试常问的 MobileNet 基础 |
| 15 min | 看 `mcunet/tinynas/nn/networks/mobilenet_v2.py` | ⭐⭐⭐ | `InvertedResidual` 类的 `forward()`；expansion ratio 控制中间通道数倍数 | Day 3 弹性模块的实现 |
| 15 min | 对比标准卷积 vs 深度可分离卷积 | ⭐⭐⭐ | 标准卷积：参数量 C_in×K×K×C_out；Depthwise+Pointwise：C_in×K×K + C_in×C_out | 为什么深度可分离适合 MCU |

**产出**：能计算一个 Inverted Residual Block 的参数量和 MAC

### Day 13：NAS 搜索空间设计原则 ⭐⭐⭐

**目标**：理解搜索空间设计的工程考量

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 分析搜索空间的自由度 | ⭐⭐⭐ | 弹性深度（每层可选保留/跳过）、弹性宽度（通道数）、弹性核大小（3/5/7） | 搜索空间爆炸问题的处理 |
| 10 min | 理解分辨率自适应 | ⭐⭐ | 不同分辨率 × 不同宽度 = 同一框架适配不同 MCU 内存 | 硬件感知搜索 |
| 15 min | 搜索效率 | ⭐⭐ | 弹性网络共享权重 → 评估子网络不需重新训练 → 一次训练多次搜索 | Day 8 OFA 的共享权重概念 |

**产出**：能说出 MCUNet 搜索空间包含哪几个维度

### Day 14：与 MobileNet/ShuffleNet 的对比 ⭐⭐⭐

**目标**：理解 MCUNet 和其他高效网络的异同

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 对比 MobileNetV2 | ⭐⭐⭐ | MobileNet 是手动设计的通用高效网络；MCUNet 是硬件感知的自动搜索网络；MCUNet 更小但更适配特定 MCU | Day 15 设计题 |
| 10 min | 对比 ShuffleNet | ⭐⭐ | ShuffleNet 用 channel shuffle 替代 1×1 卷积；在 MCU 上 shuffle 操作不友好 | 理解 MCU 的硬件约束 |
| 15 min | 理解"硬件感知" | ⭐⭐⭐ | 同一个准确率目标，不同 MCU（64KB vs 512KB SRAM）搜索出不同架构 | 协同设计的核心体现 |

**产出**：能说出 MCUNet 为什么比手工设计的网络更适合 MCU

### Day 15：完整追踪：从 NAS 搜索到 MCU 部署 ⭐⭐⭐⭐⭐

**目标**：把前三周知识串联起来

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | 阶段 0：定义搜索空间 | ⭐⭐⭐⭐⭐ | 弹性网络定义（Day 3-4） | 起点 |
| 15 min | 阶段 1：NAS 搜索 | ⭐⭐⭐⭐⭐ | Two-stage：弹性网络进化搜索 → 找到最优配置（Day 5） | 核心 |
| 10 min | 阶段 2：重训最优模型 | ⭐⭐⭐⭐ | FP32 PyTorch 训练 → INT8 PTQ → TFLite 导出（Day 13） | 模型准备 |
| 10 min | 阶段 3：部署 | ⭐⭐⭐⭐ | TFLite → TinyEngine 编译 → C 代码 → MCU 烧录（Day 7） | 终点 |

**产出**：能在白板上完整画出 MCUNet 从搜索到部署的全流程

---

## 第 4 周：实践 + 深入（5 天）

### Day 16：动手加载 + 推理预训练模型 ⭐⭐⭐⭐

**目标**：实际加载 MCUNet 模型并推理

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 5 min | 安装依赖 | ⭐⭐⭐⭐ | `pip install -e .` | 必须前置 |
| 15 min | 加载 mcunet-320kb-imagenet | ⭐⭐⭐⭐ | `build_mcunet_model()` 加载预训练权重 | Day 17 分析模型结构 |
| 15 min | 看模型结构 | ⭐⭐⭐ | 用 `print(model)` 看各层参数 | 验证 Day 1-5 的理解 |
| 10 min | 跑一次前向推理 | ⭐⭐⭐ | `model(torch.randn(1,3,160,160))` 验证输出 shape | 理解输入输出 |

**产出**：成功加载并运行 MCUNet 预训练模型

### Day 17：INT8 量化 + TFLite 导出 ⭐⭐⭐⭐

**目标**：实际完成 PTQ 并导出 TFLite

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 执行 INT8 训练后量化 | ⭐⭐⭐⭐ | 校准数据 → 计算 scale/zero_point → 量化权重和激活 | Day 13 的实践 |
| 15 min | TFLite 导出 | ⭐⭐⭐⭐ | `generate_tflite.py` 导出 INT8 .tflite 文件 | Day 18 部署前置 |
| 15 min | 对比 FP32 vs INT8 精度 | ⭐⭐⭐ | 量化后 top-1 精度下降通常 <1% | 验证量化质量 |

**产出**：成功导出一个 INT8 MCUNet TFLite 模型

### Day 18：MCUNetV2 Patch 推理实验 ⭐⭐⭐

**目标**：理解 patch 推理的实际效果

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 理解 patch 分割代码 | ⭐⭐⭐ | 如果有 patch 推理的代码，实际看 n_patches 参数如何影响内存分配 | Day 8 的实践 |
| 15 min | 观察内存峰值变化 | ⭐⭐⭐ | n_patches=2 vs 4 的 SRAM 峰值对比 | 验证 patch 推理效果 |
| 10 min | 边界影响观察 | ⭐⭐ | 不同 n_patches 下的精度差异 | 理解 trade-off |

**产出**：亲手验证 patch 推理的内存节省效果

### Day 19：与 tinyml 项目对照学习 ⭐⭐

**目标**：理解 tinyml monorepo 中各项目和 MCUNet 的关系

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 理解 TinyTL（迁移学习） | ⭐⭐ | 如何在 MCU 上只更新少量参数做迁移学习；与 MCUNetV3 的区别 | 扩展视野 |
| 15 min | 理解 NetAug（网络增强） | ⭐⭐ | 训练时用大网络辅助小网络，推理时丢弃大网络 | 知道"逆向知识蒸馏" |
| 15 min | 理解 OFA（Once-for-All） | ⭐⭐ | 一次训练弹性网络，进化搜索出多种硬件的最优子网络 | MCUNet 搜索空间的前身 |

**产出**：能说出 tinyml 中 4 个子项目的核心思想和区别

### Day 20：复习 + 自测 ⭐⭐⭐⭐⭐

**目标**：检验理解程度

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 45 min | 自测面试题 | ⭐⭐⭐⭐⭐ | 完整回答 13 道面试题 | 所有知识的大验收 |

**产出**：确认自己掌握了核心内容

---

## 附录：关键文件速查表

| 文件 | 核心函数 | 重要性 | 你的进度 |
|------|---------|--------|---------|
| `mcunet/model_zoo.py` | `build_mcunet_model()` | ⭐⭐⭐⭐⭐ | |
| `mcunet/tinynas/nn/networks/proxyless_nets.py` | `ProxylessNASNets` 类 | ⭐⭐⭐⭐⭐ | |
| `mcunet/tinynas/elastic_nn/modules/dynamic_layers.py` | `DynamicMBConvLayer` | ⭐⭐⭐⭐ | |
| `mcunet/tinynas/elastic_nn/modules/dynamic_op.py` | `DynamicConv2d` 等 | ⭐⭐⭐⭐ | |
| `mcunet/tinynas/nn/networks/mobilenet_v2.py` | `InvertedResidual` | ⭐⭐⭐ | |
| `eval_torch.py` | FP32 模型评估 | ⭐⭐⭐ | |
| `eval_tflite.py` | INT8 模型评估 | ⭐⭐⭐ | |
| `mcunet/utils/pytorch_modules.py` | `MyConv` / `MyLinear` | ⭐⭐⭐ | |
| `mcunet/utils/bn_utils.py` | BN 融合 | ⭐⭐⭐ | |
| `mcunet/tinyengine/README.md` | TinyEngine 接口 | ⭐⭐⭐ | |

## 附录：常见误区

| 误区 | 正解 |
|------|------|
| MCUNet 只是一个模型 | 它是 TinyNAS（搜索）+ TinyEngine（引擎）的协同设计框架，不是单一模型 |
| TinyNAS 搜索出来的架构直接在 MCU 上跑 | 需要 TinyEngine 编译成 C 代码才能部署 |
| 弹性网络训练完就能用 | 弹性网络是搜索工具，最终需要重训练选出的子配置 |
| MCUNetV2 和 V1 只是模型的区别 | V2 的核心是 patch-based 推理，降低了推理的内存峰值 |
| NAS 搜索慢 | 使用弹性共享权重 + 准确率预测器，评估一个子网络只需一次前向 |

## 附录 A：面试常问题目

### 基础题（必须答对）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 1 | MCUNet 是什么？核心贡献？ | 系统-算法协同设计框架。TinyNAS 自动搜索适配 MCU 内存的架构；TinyEngine 代码生成消除运行时开销。比 TFLite Micro 快 1.5-3x，内存省 2.7-4.8x。 | Day 1 |
| 2 | 什么是系统-算法协同设计？ | 架构搜索时就用真实的内存模型约束（调用 TinyEngine 的内存分析），而不是搜索完后才发现放不下。搜索方向和硬件约束同时优化。 | Day 6 |
| 3 | MCUNet 如何适配不同 MCU？ | TinyNAS 的搜索空间支持弹性深度/宽度/核大小；针对不同 SRAM/Flash 预算，进化搜索自动找到最优配置。 | Day 2, 5 |
| 4 | 为什么 MCU 上要用深度可分离卷积？ | 标准卷积参数和计算量大；深度可分离（depthwise+pointwise）将计算量降低约 8-9 倍。MCU 的 MAC 和内存都极为有限。 | Day 12 |
| 5 | V1/V2/V3 各自的贡献？ | V1：TinyNAS+TinyEngine 协同设计实现推理。V2：Patch 推理降低内存峰值。V3：在 MCU 上做训练（QAS+稀疏更新+编译时 AD）。 | Day 1, 8, 11 |

### 进阶题（区分水平）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 6 | 弹性通道数如何实现？ | `DynamicConv2d` 存储最大通道的完整权重。切换到子配置时，只用 `weight[:out_ch, :in_ch]` 子矩阵。前向后向都只作用于切片。 | Day 3 |
| 7 | 为什么 MCU 不适合用 ShuffleNet 的 channel shuffle？ | Shuffle 在 SIMD 上没有高效的向量化实现，且在 TinyEngine 的内存调度中会产生碎片化的内存访问模式。 | Day 14 |
| 8 | Patch 推理的边界效应如何缓解？ | Patch 之间设置 overlap 区域（通常 1-4 像素），相邻 patch 的计算结果在重叠部分取平均或直接丢弃。 | Day 8 |
| 9 | 训练后量化（PTQ）和量化感知训练（QAT）的区别？ | PTQ 不需要重新训练，用校准数据统计 scale。QAT 在训练时模拟量化误差，精度更高但成本更大。MCUNet 用 PTQ（INT8）。 | Day 9, 17 |
| 10 | 弹性网络的"渐进式收缩"是什么意思？ | 训练超参数从大到小逐渐收缩：先训练完整网络 → 逐渐减少宽度/深度/核大小 → 最终训练所有配置。避免大跨度迁移导致训练不稳定。 | Day 4 |

### 设计题（展示架构能力）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 11 | 如果要部署一个新任务（如语音识别）到 MCU，流程是什么？ | (1) 定义任务特定的搜索空间（输入/输出维度）；(2) MCUNet 弹性网络适配新任务；(3) TinyNAS 搜索最优架构；(4) TinyEngine 编译部署。 | Day 1, 15 |
| 12 | 256KB SRAM 放不下模型，有哪些优化手段？ | (1) Patch 推理(V2)降低激活内存；(2) 减少分辨率和通道数；(3) 权重压缩（INT4）；(4) 算子融合减少中间 buffer。 | Day 8, 6 |
| 13 | TinyNAS 和 Once-for-All 有什么异同？ | 相同：都用弹性网络 + 进化搜索。不同：TinyNAS 是两阶段（先搜索后重训），集成内存模型约束；OFA 是一次训练后直接提取，用准确率预测器加速。 | Day 5, 19 |

## 附录 B：学习达标标准

### 第 1 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能说出 7 个预训练模型；能解释弹性网络的多维度；能画出 MCUNet 搜索空间结构 |
| **优秀** | 能解释弹性通道数的权重切片实现；能说明两阶段 NAS 各阶段的作用 |

### 第 2 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能解释系统-算法协同设计；能说明 TinyEngine 在 MCUNet 中的角色；理解 patch 推理原理 |
| **优秀** | 能推导给定模型的内存估算（激活+权重）；能评估不同 n_patches 的精度影响 |

### 第 3 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能说出 MCU 训练的三个核心技术；理解 inverted residual block；能对比 MCUNet 和 MobileNet |
| **优秀** | 能画出完整部署流水线；能解释不同类型 NAS 的对比 |

### 第 4 周结束标准（最终验收）

| 级别 | 标准 |
|------|------|
| **达标** | 能加载预训练模型并推理；能导出 INT8 TFLite；能回答 5 道基础题 |
| **优秀** | 能回答所有 13 道面试题；能说出 MCUNet 和 tinyml 各子项目的关联 |
