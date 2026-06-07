# 附录 B：岗位导向学习路线 — 嵌入式高性能计算工程师（深圳）

> 本附录基于真实 JD（2026-05-31 深圳）定制。目标岗位：**中/高级嵌入式工程师（高性能计算）**。
> 
> 每个课程知识点都标注了 **JD 关联度**（★★★★★ = 面试必问，★★★ = 加分项）。
> 
> 学习策略：**不是把所有 note 都读一遍，而是按岗位需求精准学习**。

---

## 1. JD 拆解与课程映射

### JD 工作职责 1：NN 算法在移动端/桌面端的部署和优化

| JD 关键词               | 对应课程模块                                                                                             | 优先级 |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ------ |
| **移动端部署**              | note/lecture-10 (MCUNet), note/lecture-11 (TinyEngine), 附录A (硬件选型)                                 | ★★★★★   |
| **桌面端部署**              | note/lecture-13 (LLM Deployment), project/edge_ai_compression_deployment                                 | ★★★★    |
| **模型优化**                | note/lecture-03/04 (剪枝), note/lecture-05/06 (量化), note/lecture-09 (蒸馏)                             | ★★★★★   |
| **ONNX/TensorRT 导出**      | project/edge_ai_compression_deployment/export, src/lecture-22 (端到端压缩管线)                           | ★★★★★   |
| **延迟/内存/功耗基准测试**   | note/lecture-02 (效率指标), src/lecture-02 (benchmark), 附录A Section 6 (三步检查法)                     | ★★★★★   |
| **batch=1 推理优化**        | 附录A Section 2.2 (batch=1 利用率暴跌真相), note/lecture-11 (TinyEngine 优化)                           | ★★★★    |

### JD 工作职责 2：与开发对接联调，保证算法在工程端落地

| JD 关键词           | 对应课程模块                                     | 优先级 |
| ------------------- | ------------------------------------------------ | ------ |
| **工程落地**            | project/edge_ai_compression_deployment (完整流水线) | ★★★★★   |
| **精度回归测试**        | note/lecture-06 Section 11 (落地checklist)       | ★★★★    |
| **ONNX 模型验证**       | 附录A Section 6 Step 3 (算子兼容性检查)          | ★★★★    |
| **CI/CD 集成**          | 附录A Section 8 (理想部署流程)                   | ★★★     |
| **多平台兼容性**        | 附录A Section 1 (22 种板卡对比)                  | ★★★★    |

### JD 要求 1：计算机体系架构 + 并行计算 + ARM/X86 指令集

| JD 关键词               | 对应课程模块                                                                   | 优先级 |
| ----------------------- | ------------------------------------------------------------------------------ | ------ |
| **计算机体系架构**          | note/lecture-11 (TinyEngine: 内存层次/缓存/寄存器), 附录A Section 3 (Memory Wall) | ★★★★★   |
| **并行计算 (SIMD/NEON)**   | note/lecture-11 (TinyEngine: SIMD 编程), src/lecture-11 (Im2Col/Winograd)       | ★★★★★   |
| **ARM 指令集**             | note/lecture-11 (Cortex-M SIMD/CMSIS-NN), src/lecture-11                        | ★★★★★   |
| **X86 指令集**             | src/lecture-11 (CPU 基准测试可扩展到 x86 AVX)                                   | ★★★     |
| **内存墙/带宽瓶颈**        | 附录A Section 3 (彻底讲透), 附录A Section 6 Step 2                              | ★★★★★   |
| **Roofline Model**         | 附录A Section 6 (隐含)                                                         | ★★★★    |

### JD 要求 2：异构计算框架 (OpenCL / Metal / NEON / CUDA)

| JD 关键词               | 对应课程模块                                                                                       | 优先级 |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ------ |
| **NEON (ARM SIMD)**        | note/lecture-11, src/lecture-11 (im2col + Windows 卷积的 SIMD 优化)                                 | ★★★★★   |
| **CUDA**                   | note/lecture-11 (CUDA 编程), note/lecture-19/20 (分布式训练涉及 GPU 并行)                           | ★★★★    |
| **OpenCL**                 | 课程较少直接涉及，但 note/lecture-11 的并行编程思想可迁移。**建议自学：** Mali GPU OpenCL kernel      | ★★★     |
| **Metal (Apple GPU)**      | 课程较少直接涉及，但 note/lecture-16 (ViT) 涉及 Apple ANE。**建议看：** CoreML + MPS backend 源码    | ★★★     |
| **算子编写与优化**         | src/lecture-11 (自定义 Im2Col, Winograd, Conv+BN+ReLU 融合)                                          | ★★★★★   |

### JD 要求 3：移动端推理框架 (TFLite / CoreML / PyTorch Mobile / MNN)

| JD 关键词               | 对应课程模块                                                                                                  | 优先级 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- | ------ |
| **TFLite**                 | note/lecture-10 (MCUNet → TFLite Micro), labs/lab-02 (量化实验 → TFLite INT8), 附录A Section 4.1 (模型门槛)      | ★★★★★   |
| **CoreML**                 | note/lecture-16 (ViT on Apple ANE), 附录A Section 9 (工具链速查: coremltools)                                  | ★★★★    |
| **PyTorch Mobile**         | 课程全部 PyTorch 代码可导出 torchscript / ONNX → 与 PyTorch Mobile 等价                                        | ★★★★    |
| **MNN (阿里开源)**          | 课程未直接涉及，**但需要重点补充**（见下文 Section 5）                                                            | ★★★★★   |
| **ncnn (腾讯开源)**         | 课程未直接涉及，**但需要重点补充**（见下文 Section 5）                                                            | ★★★★    |
| **ONNX Runtime (跨框架)**   | project/edge_ai_compression_deployment/export, 附录A Section 9                                                 | ★★★★★   |

---

## 2. 精准学习路线（30 天）

### 第 1 周：基础夯实 + 面试高频

```
Day 1-2: 效率指标 + Memory Wall
  ├── note/lecture-01 (引言) — 30 min
  ├── note/lecture-02 (效率指标) — 1h
  ├── 附录A Section 2 (TOPS 骗局) — 1h
  └── 附录A Section 3 (Memory Wall) — 1.5h ★ 核心

Day 3-4: 量化（面试必问第一名）
  ├── note/lecture-05 (量化基础) — 1.5h
  ├── note/lecture-06 (PTQ/QAT/混合精度) — 1.5h
  ├── src/lecture-05 (手写线性量化) — 跑通代码
  └── src/lecture-06 (PTQ + QAT 完整管线) — 跑通代码

Day 5-6: 剪枝
  ├── note/lecture-03 (剪枝基础 + 结构化 vs 非结构化) — 1.5h
  ├── note/lecture-04 (自动剪枝率 + 系统支持) — 1h
  ├── src/lecture-03 (敏感性扫描) — 跑通代码
  └── src/lecture-04 (通道剪枝 + 微调) — 跑通代码

Day 7: 硬件基础
  ├── 附录A Section 1 (22 种硬件全景) — 1h ★★★
  ├── 附录A Section 4 (模型-硬件门槛) — 1h
  └── 附录A Section 6 (三步检查法) — 1h ★★
```

### 第 2 周：推理引擎 + 异构计算

```
Day 8-9: TinyEngine + SIMD (面试高频第二名)
  ├── note/lecture-11 (TinyEngine) — 2h ★★★
  ├── src/lecture-11 (im2col/Winograd/算子融合) — 跑通代码
  └── 重点理解：
      - Im2Col 为什么内存膨胀 10-20x
      - Winograd F(2,3) 为什么理论快但 INT8 下精度差
      - Conv+BN+ReLU 融合前后的内存读写次数对比

Day 10-11: ARM NEON 深入
  ├── 自行学习 ARM CMSIS-NN 源码 (github.com/ARM-software/CMSIS-NN)
  │   重点看: arm_convolve_s8.c, arm_depthwise_conv_s8.c
  ├── 理解: SMLAD 指令（一次做 4 个 INT16 乘加）
  ├── 理解: 为什么 CMSIS-NN 要求 scale 必须是 2 的幂
  └── src/lecture-11 中的 SIMD 模拟代码对照理解

Day 12-13: CUDA 编程基础
  ├── note/lecture-11 (CUDA 部分) — 1h
  ├── 自行学习 CUDA C++ Programming Guide:
  │   - Thread/Block/Grid 层级
  │   - Shared memory vs Global memory
  │   - Bank conflict
  │   - Tensor Core 的 mma.sync 指令
  └── 理解: 为什么 2:4 sparsity 在 A100 上快 2x

Day 14: 移动端推理框架对比
  ├── 附录A Section 9 (工具链速查) — 1h
  └── 动手: 用 TFLite benchmark_model 测试 MobileNetV2 延迟
```

### 第 3 周：端到端部署 + 动手项目

```
Day 15-16: ONNX + TensorRT 全链路
  ├── project/edge_ai_compression_deployment/export/exporter.py — 读代码
  ├── project/edge_ai_compression_deployment/benchmark/benchmarker.py — 读代码
  ├── 动手: 导出一个 ONNX 模型 → onnx.checker 验证 → onnxruntime 推理
  └── 理解: dynamic batch / dynamic shape 在 ONNX 中的表示

Day 17-18: 完整压缩流水线
  ├── project/edge_ai_compression_deployment/main.py — 读代码
  ├── src/lecture-22 (端到端管线) — 跑通代码
  └── 重点理解:
      - 为什么先剪枝再量化（顺序不能反）
      - 剪枝+量化叠加后，精度损失是非线性的

Day 19-20: 手写一个部署 demo
  └── 挑战: 用 PyTorch 训练一个小 CNN → 通道剪枝 → INT8 量化
      → ONNX 导出 → onnxruntime C++ API 推理 → 对比延迟和精度

Day 21: LLM 部署基础
  ├── note/lecture-12 (Transformer 基础) — 1h
  ├── note/lecture-13 (LLM 部署: AWQ/vLLM/FlashAttention) — 2h
  └── 理解: LLM decode 为什么是纯 memory-bound（附录A Section 3.3）
```

### 第 4 周：面试冲刺 + MNN/ncnn 补充

```
Day 22-23: MNN 深度学习
  ├── MNN 源码阅读重点:
  │   - source/geometry/ (几何计算, 算子转换)
  │   - source/core/WrapExecution.cpp (后端选择逻辑)
  │   - express/ (表达式, 类似 PyTorch 的 eager mode)
  ├── 理解: MNN 的 "几何计算" → 把各种算子统一转换为 matmul + 内存重排
  └── 理解: MNN 的 Session 预推理 (Resize → 内存预分配 → 图优化)

Day 24: ncnn 对比学习
  ├── ncnn 特点: 纯 C++、无依赖、手工汇编优化
  ├── 理解: ncnn 和 MNN 的区别
  │   - ncnn: 轻量极致, 适合 ARM CPU, 算子手工 SIMD 汇编
  │   - MNN: 通用性更好, 支持 OpenCL/Metal/Vulkan, Session 预推理
  └── 面试回答模板: "我选 MNN 因为需要 OpenCL 后端加速；
      选 ncnn 因为 APK 体积敏感且只需要 CPU 推理"

Day 25-26: 面试 Mock
  ├── 重点刷 Section 9 (面试问题) 中的所有题
  ├── 特别是:
  │   - lecture-05/06: "INT8 量化为什么不掉精度？INT4 为什么开始掉？"
  │   - lecture-11: "Im2Col vs Winograd 分别在什么场景下更好？"
  │   - 附录A: "100 TOPS 的 NPU 和 100 TOPS 的 GPU，实际跑模型哪个快？"
  │   - lecture-13: "LLM decode 为什么是 memory-bound？"
  └── 每题写出 3 分钟能讲完的回答（口语化、有数字、有对比）

Day 27-28: 代码手写练习
  ├── 手写: 线性量化函数 (FP32→INT8)  ← 笔试高频
  ├── 手写: im2col + GEMM 卷积
  ├── 手写: 通道剪枝（按 F 范数排序）
  ├── 手写: BN 融合到 Conv 的权重变换
  └── 手写: 一个简单的 NEON SIMD 向量加法 kernel

Day 29-30: 项目复盘 + 简历优化
  ├── 把 project/edge_ai_compression_deployment 写成简历项目
  └── 项目描述模板见 Section 7
```

---

## 3. 核心知识点速查表（按面试频率排序）

| 排名 | 知识点                       | 来源                           | 面试出现概率 | 掌握程度要求                       |
| ---- | ---------------------------- | ------------------------------ | ------------ | ---------------------------------- |
| 1    | **INT8 量化原理 (scale/zp/校准)** | note/lecture-05/06             | 95%          | 能手写, 能解释 per-channel vs per-tensor |
| 2    | **通道剪枝 vs 非结构化剪枝**     | note/lecture-03/04             | 90%          | 能说清硬件加速差异, 2:4 sparsity     |
| 3    | **Memory Wall / Roofline**       | 附录A Section 3               | 85%          | 能画图说明 compute-bound vs memory-bound |
| 4    | **Im2Col + GEMM / Winograd**     | note/lecture-11               | 80%          | 能写 Im2Col, 能分析内存膨胀          |
| 5    | **ONNX 导出 + 常见坑**           | project/export, 附录A Section 6.3 | 75%      | 能说出 5 个常见不兼容算子            |
| 6    | **ARM NEON SIMD**                | note/lecture-11, CMSIS-NN 源码 | 75%          | 能解释 SMLAD, 能写简单向量化         |
| 7    | **TFLite / MNN / ncnn 对比**     | 附录A + 自学                  | 70%          | 能说出各自适用场景和架构差异          |
| 8    | **BN 融合**                      | note/lecture-06/11            | 70%          | 能手写融合公式                      |
| 9    | **TensorRT 优化流程**            | 附录A Section 9               | 65%          | 能说清 builder → network → engine   |
| 10   | **batch=1 推理优化**            | 附录A Section 2.2             | 60%          | 能解释 GPU batch=1 利用率为什么低     |

---

## 4. 必须补充的课程外知识（JD 明确要求但课程未覆盖）

### 4.1 MNN（阿里开源）- ★★★★★

```
关键概念：
  - Session: 预推理 = Resize(分配内存) → 图优化 → 内存复用
  - Geometry: 把任意算子转换为 "几何计算"（本质是 matmul + layout transform）
  - Backend: CPU/OpenCL/Metal/Vulkan 统一抽象层
  - 内存优化: 引用计数 + 内存池 (类似 vLLM 的 PagedAttention 思想)

面试常问：
  Q: "MNN 为什么比 TFLite 快？"
  A: ① Session 预推理避免了每次推理的内存分配
     ② 几何计算把不同算子的计算模式统一化，减少了 dispatch 开销
     ③ ARM 汇编级别的优化（类似 CMSIS-NN），但支持多后端

学习路径：
  1. 读 MNN 官方文档: https://www.yuque.com/mnn/cn
  2. 跑 MNN 的 benchmark: ./benchmark.out models/ 4 10 0
  3. 重点源码: Express.cpp, Session.cpp, GeometryConv2D.cpp
```

### 4.2 ncnn（腾讯开源）- ★★★★

```
关键概念：
  - 纯 C++，零依赖 → APK 体积增加 <500KB
  - 手工 SIMD 汇编 (ARM NEON / x86 AVX)
  - Vulkan GPU 后端 (移动端 GPU 推理)
  - 8-bit 量化：无 calibration 的 naive INT8 (symmetric, per-tensor)

面试常问：
  Q: "ncnn 在什么场景下比 MNN 好？"
  A: APK 体积极度敏感 (如微信小程序、轻量 SDK)
     只需要 CPU 推理，不需要 OpenCL/Metal 后端
     模型结构简单 (CNN 为主，Transformer 支持较弱)

学习路径：
  1. 读 ncnn wiki: https://github.com/Tencent/ncnn/wiki
  2. 转一个模型: onnx2ncnn model.onnx model.param model.bin
  3. 重点理解: ncnnoptimize 做了什么优化
```

### 4.3 OpenCL / Metal 编程 - ★★★

```
OpenCL（跨平台 GPU 编程，主要是手机 GPU）:
  - 掌握: kernel 编写 → command queue → buffer 管理
  - 对应场景: 骁龙 Adreno GPU / 麒麟 Mali GPU 上的定制 kernel

Metal (Apple GPU):
  - 掌握: MPS (Metal Performance Shaders) 的基本用法
  - 对应场景: iPhone/iPad 上的自定义算子

学习路径：
  1. OpenCL: 从矩阵乘法 kernel 开始（类比 CUDA 写法的迁移）
  2. Metal: 用 CoreML + MPS 组合，大多数情况不需要手写 Metal Shading Language
```

---

## 5. 常见面试问题（针对此岗位的补充）

除了 23 篇 note 中各有的 4-6 道面试题外，以下是针对此岗位的特化问题：

**Q1**：你在手机上部署了一个 YOLOv8-nano，FP32 延迟 45ms，目标 <15ms。请列出你的优化路径和每步的预期收益。

> **参考答案**：
> 1. INT8 量化 → 延迟降到 20-25ms（带宽减半 + INT8 指令加速）
> 2. 通道剪枝 30% → 延迟降到 12-15ms（通道减少 → 计算量 ~减半）
> 3. BN 融合 + Conv-ReLU 融合 → 延迟降到 10-13ms（减少 2 次内存往返）
> 4. 如果还不够 → 换 EfficientViT 或 YOLOv8-pico（天生小的架构）
> 5. 如果目标硬件是骁龙 → 用 SNPE/QNN 替代 TFLite（NPU 比 CPU 快 3-5x）

**Q2**：为什么同样的 INT8 模型，在骁龙 865 上比在骁龙 8 Gen 3 上慢 3 倍？

> **参考答案**：
> 1. 骁龙 8 Gen 3 的 Hexagon NPU 有专门的 INT8 Tensor Accelerator，865 没有
> 2. LPDDR5 (77 GB/s) vs LPDDR4x (34 GB/s) → 内存带宽差 2.3x
> 3. 8 Gen 3 的 Hexagon 支持 depthwise conv 硬件加速，865 的 depthwise conv 掉回 CPU
> 4. 所以不是 "算力差 3 倍"，而是 "架构代差 + 内存带宽 + 算子支持" 的复合效应

**Q3**：你如何向算法同事解释——为什么他们设计的 "轻量" 模型（depthwise conv 多）在手机上反而慢？

> **参考答案**：
> 用 Roofline Model 解释：
> - 普通 3x3 Conv: 计算密度高（每个数据做 9 次乘加）→ compute-bound → GPU 利用率高
> - Depthwise Conv: 计算密度低（每个数据只做 1 次乘加）→ memory-bound → 带宽是瓶颈
> - 所以尽管 Depthwise Conv 的 FLOPs 是普通 Conv 的 1/C_out，但延迟可能只降了 20%
> - 建议：用 group=2 或 group=4 的 group conv 替代 depthwise，平衡计算密度和参数量

---

## 6. 简历项目描述模板

把 course project 写成简历条目：

```
项目: 端侧 AI 模型压缩与部署全链路 (基于 MIT 6.5940 课程框架)
时间: 2026.03-2026.06

技术栈: PyTorch, ONNX, ONNX Runtime, TensorRT(模拟), TFLite, CMSIS-NN

工作内容:
  - 设计了完整的模型压缩流水线: 结构化通道剪枝(30-50% sparsity) →
    INT8 PTQ/QAT 量化 → 知识蒸馏 → ONNX 导出 → 多平台基准测试
  - 实现了针对 ARM NEON 的 im2col+GEMM 卷积优化和 Conv+BN+ReLU 算子融合,
    在模拟 Cortex-M7 环境下获得 2-5x 延迟优化
  - 建立了准确率/延迟/内存/功耗四维评估体系, 自动生成对比报告,
    覆盖 ResNet/MobileNet/ViT/EfficientViT 等多架构
  - 模型量化后延迟降低 50-70%, 内存占用降低 75%, 精度损失 <0.5%

量化成果:
  - MobileNetV2 INT8: 延迟从 12ms → 4ms（-67%）, 模型大小 13.4MB → 3.6MB
  - 在模拟 256KB SRAM MCU 上成功部署 MCUNet 架构, 关键调检测延迟 <10ms
```

---

## 7. 快速诊断：你离这个岗位还有多远

```
□ 能手写线性量化的 scale + zero_point 计算公式？
□ 能解释 per-channel 量化为什么比 per-tensor 精度高？
□ 能说出 3 个 ONNX 导出时的常见不兼容算子？
□ 能画 Memory Wall 图并标注 compute-bound vs memory-bound 区域？
□ 能写出 Im2Col 的伪代码并分析内存膨胀？
□ 能说出 MNN 的 Session 预推理机制？
□ 能说出 ncnn 和 MNN 的两个核心差异？
□ 能解释为什么 GPU batch=1 时利用率只有 5-15%？
□ 能用 TFLite benchmark_model 在自己的手机上跑过模型？
□ 曾把 PyTorch 模型成功部署到 Android/iOS 或 MCU 上？

如果 <5 个 ✓: 先按 Section 2 的 30 天学习路线执行
如果 5-7 个 ✓: 重点攻克 Section 4 中的 MNN/ncnn/OpenCL 部分
如果 8-10 个 ✓: 你已经 meet the bar, 可以直接投, 重点准备行为面试
```

---

> **核心信息**: 这个岗位的本质是 **"把算法论文变成手机/嵌入式设备上能跑且不卡的东西"**。你需要的不是读更多论文，而是：
> 1. 理解硬件（ARM/Mali/Adreno/ANE/NPU — 各自的脾气）
> 2. 掌握工具（TFLite/MNN/ncnn/CoreML — 知道什么时候用哪个）
> 3. 建立直觉（看一眼模型结构, 就知道它的瓶颈是带宽还是计算, 能不能量化, 在什么硬件上会崩）
>
> MIT 6.5940 的课程内容覆盖了以上 80% 的理论和 50% 的工程实践。剩下的 50% 工程实践（MNN/ncnn/OpenCL）需要自己补。
