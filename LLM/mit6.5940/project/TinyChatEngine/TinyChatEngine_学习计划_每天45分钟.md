# TinyChatEngine 学习计划：每天 45 分钟，从入门到理解核心

**一句话定位**：TinyChatEngine 是 MIT HAN Lab 的纯 C/C++ 端侧 LLM/VLM 推理引擎。无任何 Python/库依赖，手写平台专属 SIMD 内核（NEON/AVX/CUDA/Metal），将 AWQ/SmoothQuant 量化后的模型高效部署到 x86/ARM/CUDA/Apple 平台上。MLSys 2024。

**总时长**：约 4 周（20 个学习日），每天 45 分钟。

**重要性说明**：
- ⭐⭐⭐⭐⭐ = 必须掌握（不然后续无法理解）
- ⭐⭐⭐⭐   = 核心理解（面试/开发中常见）
- ⭐⭐⭐     = 重要但可先走读（用到再细看）
- ⭐⭐       = 了解即可（高级/特定场景）

---

## 第 1 周：架构全貌 + 模型加载（5 天）

### Day 1：项目骨架 + 构建系统 ⭐⭐⭐⭐⭐

**目标**：知道项目做什么，怎么编译，怎么运行

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 通读 `README.md` | ⭐⭐⭐⭐⭐ | 项目定位（纯 C++ 零依赖 LLM 推理）、支持平台（x86/ARM/CUDA/Metal）、支持模型列表 | 所有后续阅读的上下文 |
| 15 min | 看 `llm/Makefile` | ⭐⭐⭐⭐⭐ | 构建系统：平台宏 `QM_ARM`/`QM_x86`/`QM_CUDA`/`QM_METAL`；编译选项和链接 | Day 18 编译时直接看 |
| 10 min | 看 `llm/scripts/chat.sh` | ⭐⭐⭐⭐ | 聊天入口脚本：下载模型 → 运行聊天；参数传递 | Day 18 跑聊天 |
| 5 min | 浏览目录结构 | ⭐⭐⭐ | `kernels/`（平台 SIMD 内核） + `llm/src/`（推理逻辑） + `llm/tools/`（Python 工具） | Day 2-5 定向阅读 |

**产出**：能说出 4 个平台后端和编译方式

### Day 2：推理入口 + 平台调度 ⭐⭐⭐⭐⭐

**目标**：理解主函数和平台分发的顶层逻辑

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `llm/src/interface.cc` | ⭐⭐⭐⭐⭐ | 主入口 `main()`：解析参数 → 加载模型 → 初始化 → 聊天循环（输入→生成→输出） | Day 14 追踪完整流程 |
| 15 min | 读 `kernels/matmul.h` + `matmul_int4.cc` | ⭐⭐⭐⭐⭐ | 矩阵乘法调度器：根据平台宏 `#ifdef QM_ARM` 等分发到不同内核；`int4MatMul()` 统一接口 | Day 6-9 各平台内核入口 |
| 10 min | 读 `kernels/matmul_int8.cc` | ⭐⭐⭐ | INT8 版矩阵乘法调度器，架构和 INT4 相同 | 对比理解 |
| 5 min | 看 `kernels/pthread_pool.cc` | ⭐⭐⭐ | 多线程池：CPU 并行计算的基础设施 | Day 7-8 CPU 内核 |

**产出**：能画出从 `main()` 到平台 SIMD 内核的调用链

### Day 3：神经网络模块 — 解码器层 ⭐⭐⭐⭐⭐

**目标**：理解一个 Transformer 解码器层的完整 C++ 实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `llm/src/nn_modules/non_cuda/Int4llamaDecoderLayer.cc` | ⭐⭐⭐⭐⭐ | Llama 解码器层的 CPU 实现：QKV 投影（INT4 MatMul）→ RoPE → 注意力 → output 投影（INT4 MatMul）→ FFN（gate+up INT4 MatMul → SiLU → down INT4 MatMul） | Day 15 端到端追踪 |
| 15 min | 理解各算子的调用顺序 | ⭐⭐⭐⭐ | `int4MatMul()` → `RoPE()` → `attention()` → `int4MatMul()` → `SiLU()` → `int4MatMul()` | Day 14 画计算图 |
| 5 min | 看 `llm/include/` 头文件 | ⭐⭐⭐ | 理解各模块的公开接口 | Day 5 算子头文件 |

**产出**：能画出 Llama 解码器层的完整计算图

### Day 4：模型生成 + Token 采样 ⭐⭐⭐⭐

**目标**：理解 token 生成和采样策略

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `llm/src/Generate.cc` | ⭐⭐⭐⭐ | Token 采样逻辑：贪心 `argmax`、`temperature` 缩放、`top_p` 核采样、重复惩罚 `repetition_penalty` | Day 18 改参数体验 |
| 10 min | 理解采样流程 | ⭐⭐⭐ | logits → temperature 缩放 → softmax → top-p 过滤 → 从概率分布采样 | Day 14 追踪生成 |
| 15 min | 读对应模型的 Generate（如 `LLaMAGenerate.cc`） | ⭐⭐⭐ | 模型特定的生成流程：tokenizer 加载、KV cache 管理、decoder 层循环 | Day 5 模型定义 |

**产出**：能解释 temperature 和 top-p 对生成质量的影响

### Day 5：Tokenizer + 算子库 ⭐⭐⭐

**目标**：理解文本到 token 的转换和基础算子

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `llm/src/LLaMATokenizer.cc` | ⭐⭐⭐ | Llama SentencePiece Tokenizer：合并规则、vocab 查找、编码/解码 | Day 14 tokenize 步骤 |
| 10 min | 对比 OPT/GPTBigCode Tokenizer | ⭐⭐ | 不同模型的 tokenizer 差异（BPE vs SentencePiece） | 了解即可 |
| 10 min | 看 `llm/src/ops/` 目录 | ⭐⭐⭐ | 逐元素操作：SiLU/GeLU 激活、LayerNorm/RMSNorm、RoPE 位置编码 | Day 3 解码器层的依赖 |
| 10 min | 理解 RoPE 实现 | ⭐⭐⭐ | 旋转位置编码：`cos`/`sin` 预计算 → 向量旋转 | 面试常问 |

**产出**：能说出 Llama 的 tokenizer 类型和位置编码方式

---

## 第 2 周：CPU 平台内核深层（5 天）

### Day 6：ARM NEON INT4 矩阵乘法 ⭐⭐⭐⭐

**目标**：理解 ARM 平台 INT4 内核的 SIMD 实现细节

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `kernels/neon/matmul_neon_int4.cc` | ⭐⭐⭐⭐ | INT4 权重内存布局（QM_ARM）；`vld4q` 一次加载 64 个 INT4 权重；`vsubl` + `vmlal` 解包并乘累加 | Day 12 对比 AVX |
| 10 min | 理解权重重排列 | ⭐⭐⭐⭐ | 离线重排权重匹配 128-bit NEON SIMD；每 64 个 INT4 为一块，按 NEON 寄存器布局交错排列 | Day 8 对比 x86 256-bit |
| 10 min | 看 `kernels/neon/matmul_neon_int8.cc` | ⭐⭐⭐ | INT8 内核（作为对比基准）；更简单的布局但 2x 内存带宽 | 理解 INT4 vs INT8 |

**产出**：能手绘 NEON INT4 权重布局和加载模式

### Day 7：ARM NEON 优化技巧 ⭐⭐⭐⭐

**目标**：理解 NEON 内核的深层优化

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 理解分块策略 | ⭐⭐⭐⭐ | tile 大小选择（M×N×K 分块）；L1/L2 cache 友好；寄存器重用 | Day 9 跨平台对比 |
| 15 min | 理解循环展开 | ⭐⭐⭐ | 手工展开内层循环（减少循环开销）；指令级并行 | 通用 SIMD 优化 |
| 10 min | 理解预取 | ⭐⭐ | `__builtin_prefetch()` 预加载下一块数据到 cache | 深入优化 |
| 5 min | 多线程并行 | ⭐⭐⭐ | `pthread_pool` 将 M 维度切分到多核 | Day 8 同样适用 |

**产出**：能说出 NEON INT4 内核的三个关键优化点

### Day 8：x86 AVX INT4 矩阵乘法 ⭐⭐⭐

**目标**：理解 x86 平台 INT4 内核与 ARM 的异同

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `kernels/avx/matmul_avx_int4.cc` | ⭐⭐⭐ | AVX 256-bit SIMD（NEON 的 2x 吞吐）；权重布局（QM_x86）与 NEON 不同（256-bit 对齐）；`_mm256` 内联函数 | Day 12 跨平台对比总结 |
| 10 min | 对比 NEON 和 AVX 的权重布局 | ⭐⭐⭐ | NEON(QM_ARM)：128-bit 布局；AVX(QM_x86)：256-bit 布局。离线重排时的填充方式不同 | 理解平台差异化 |
| 10 min | 性能对比 | ⭐⭐ | AVX 理论上 2x NEON 吞吐量；实际增益受内存带宽限制 | decode 阶段是 memory bound |

**产出**：能画图对比 NEON 和 AVX 的 INT4 解包流程

### Day 9：跨平台内核设计总结 ⭐⭐⭐⭐

**目标**：理解四个平台的共性抽象和各自优化

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 总结四个平台的共性 | ⭐⭐⭐⭐ | (1) 离线权重重排列；(2) 分块矩阵乘法；(3) 反量化与乘累加融合；(4) 循环展开 | 面试"跨平台 LLM 推理" |
| 10 min | 总结各平台差异 | ⭐⭐⭐⭐ | NEON:128-bit 2D寄存器；AVX:256-bit；CUDA:warp级并行；Metal:threadgroup | 面试"平台适配" |
| 10 min | 理解 WHY 离线重排 | ⭐⭐⭐⭐ | 运行时不做解包：一次加载 64/128 个 INT4 + 移位/掩码提取 → 直接进入乘累加 | 核心性能来源 |
| 10 min | 理解量化方法枚举 | ⭐⭐⭐ | `QM_ARM`/`QM_x86`/`QM_CUDA` 各对应不同的权重布局和内核路径 | Day 2 的分发基础 |

**产出**：能总结"为什么 TinyChatEngine 的 INT4 推理这么快"

### Day 10：模型定义 + 多模型支持 ⭐⭐⭐

**目标**：理解不同模型架构的差异如何被统一

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 对比不同模型 decoder 层 | ⭐⭐⭐ | Llama：SwiGLU FFN + RoPE；OPT：ReLU FFN + learned PE；Falcon：parallel attention+FFN | Day 15 模型差异 |
| 15 min | 看 `non_cuda/` 下的模型文件 | ⭐⭐⭐ | 每个模型独立的 `.cc` 文件；共享 `ops/` 中的基础算子；差异在 attention/FFN 组合 | 走读即可 |
| 10 min | 理解 "LLaMA-like" 抽象 | ⭐⭐ | Mistral/Llama3/CodeLlama 都继承 LLaMA 的架构 | 代码复用 |
| 5 min | 看 `tools/` 下的 Python 工具 | ⭐⭐ | `model_quantizer.py` 模型量化转换；`download_model.py` 下载 | 了解即可 |

**产出**：能说出至少 3 种模型架构的 key 差异（FFN 类型、PE 类型、归一化位置）

---

## 第 3 周：GPU 内核 + 多模态 + Pipeline（5 天）

### Day 11：CUDA 内核 — Attention + Decoder ⭐⭐⭐

**目标**：理解 GPU 上的推理实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `kernels/cuda/gemv_cuda.cu` | ⭐⭐⭐ | GEMV CUDA 内核：warp 内并行 reduction；每个 warp 处理一个输出元素 | Day 12 对比 CPU GEMV |
| 15 min | 读 `kernels/cuda/matmul_int4.cu` | ⭐⭐⭐ | CUDA INT4 矩阵乘法：thread block × warp 两级并行；shared memory 缓存权重 tile | 理解 GPU 并行模式 |
| 15 min | 看 `llm/src/nn_modules/cuda/Int4llamaDecoderLayer.cu` | ⭐⭐⭐ | CUDA 版的 Llama 解码器层：与 CPU 版的算子一一对应，但用 CUDA 内核 | Day 13 多模态 CUDA |

**产出**：能对比 CPU warp-level 并行和 GPU warp-level 并行的相似性

### Day 12：Apple Metal 内核 ⭐⭐

**目标**：了解 Apple GPU 平台的实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `kernels/metal/op.metal` | ⭐⭐ | Metal Shading Language 矩阵乘法；threadgroup 内存共享 | 了解即可 |
| 10 min | 读 `kernels/metal/MetalMatmulInt4.cpp` | ⭐⭐ | Metal 调度器：Objective-C++ 桥接 Metal Shader | 知道调用方式 |
| 10 min | 对比 4 个后端性能 | ⭐⭐ | x86 AVX > ARM NEON > CUDA(移动) > Metal | 知道排名 |
| 10 min | 理解 "zero-dependency" 的含义 | ⭐⭐ | 不依赖 BLAS/PyTorch/CUDA SDK（除 CUDA 后端外）；全部手写实现 | 面试亮点 |

**产出**：能说出 4 个平台的性能排序和原因

### Day 13：多模态推理 — LLaVA/VILA ⭐⭐⭐

**目标**：理解 VLM 如何在纯 C++ 中实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `llm/src/nn_modules/non_cuda/LLaVAGenerate.cc` | ⭐⭐⭐ | LLaVA 的生成流程：CLIP 编码图像 → visual tokens → 拼接到 text tokens → LLM 生成 | Day 13 完整 VLM 流程 |
| 15 min | 读 `llm/src/nn_modules/non_cuda/Fp32CLIPVisionTransformer.cc` | ⭐⭐⭐ | CLIP ViT 的 C++ 实现：Patch Embedding → Transformer Encoder × N → image features | 理解视觉编码器 |
| 10 min | 对比纯文本 vs 多模态 | ⭐⭐⭐ | 多模态多了 CLIP 前向 + visual token 拼接；LLM 部分完全相同 | 理解 VLM 架构 |

**产出**：能画出 LLaVA 从图片到文本的完整数据流

### Day 14：完整追踪：Prompt → Token 生成 ⭐⭐⭐⭐

**目标**：把前三周知识串联起来

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 5 min | Tokenize | ⭐⭐⭐⭐ | 输入文本 → SentencePiece → token IDs（Day 5） | 起点 |
| 10 min | Prefill | ⭐⭐⭐⭐ | 全部 prompt tokens 并行前向；compute bound；INT4 GEMM | 计算峰值 |
| 15 min | Decode × N | ⭐⭐⭐⭐ | 逐 token 自回归；每步：embed → decoder 层 × L（QKV→attention→FFN）→ LM head → sample | 主要耗时 |
| 10 min | Detokenize | ⭐⭐⭐ | token ID → 文本；stream 输出 | 终点 |
| 5 min | 分析瓶颈 | ⭐⭐⭐ | decode 的 GEMV 是 memory bound；prefill 的 GEMM 是 compute bound | 性能优化方向 |

**产出**：能在白板上完整画出一个 prompt 从输入到生成的 C++ 执行路径

### Day 15：Advanced — 语音 + 系统调用 ⭐⭐

**目标**：了解高级功能和应用层

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 看 `llm/application/` | ⭐⭐ | 语音到语音聊天应用的桥接代码 | 了解即可 |
| 10 min | 理解 API 设计 | ⭐⭐ | 作为库嵌入到其他应用 | 部署场景 |
| 10 min | 看性能基准脚本 | ⭐⭐ | token/s 测量；内存占用统计 | 性能评估 |
| 10 min | 看多模态脚本 | ⭐⭐ | `llm/scripts/vila.sh`、`llava.sh`、`code.sh` | 知道用法 |

**产出**：知道 TinyChatEngine 的应用场景和扩展方式

---

## 第 4 周：实践 + 深入（5 天）

### Day 16：环境配置 + 编译 ⭐⭐⭐⭐

**目标**：成功编译 TinyChatEngine

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 5 min | 检查编译依赖 | ⭐⭐⭐ | `g++`/`clang++`（CPU）+ 可选 CUDA/Metal 工具链 | Day 1 Makefile 的对照 |
| 20 min | 编译 CPU 版本 | ⭐⭐⭐⭐ | `cd llm && make`；观察编译平台宏和输出 | Day 17 运行前置 |
| 10 min | 检查产物 | ⭐⭐⭐ | 生成的二进制文件；确认平台宏生效 | 验证编译 |
| 10 min | 编译失败排查 | ⭐⭐⭐ | 常见问题：AVX/NEON 指令不支持（旧 CPU） | 调试能力 |

**产出**：成功编译出 CPU 版本的 TinyChatEngine 二进制

### Day 17：运行聊天 + 性能分析 ⭐⭐⭐⭐

**目标**：实际运行量化模型并分析性能

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | 下载模型 | ⭐⭐⭐ | `llm/scripts/chat.sh` 或 `download_model.py` | Day 18 不同模型 |
| 15 min | 运行聊天 | ⭐⭐⭐⭐ | 输入 prompt → 观察 token/s 和生成质量 | Day 18 参数对比 |
| 10 min | 分析 token/s | ⭐⭐⭐ | 预填充阶段（第一个 token 慢）vs decode 阶段（后续稳定） | Day 14 性能分析 |
| 10 min | 观察内存占用 | ⭐⭐⭐ | `htop` 或 `top` 看内存；估算模型大小 + KV cache | 理解资源消耗 |

**产出**：成功运行量化 LLM 聊天并获得 token/s 指标

### Day 18：对比不同采样参数 ⭐⭐⭐

**目标**：理解采样参数对生成的影响

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | 对比 temperature（0.1/0.7/1.5） | ⭐⭐⭐ | 温度越低越确定性、越高越随机；0=贪心 | Day 4 理论验证 |
| 10 min | 对比 top-p（0.5/0.9/1.0） | ⭐⭐⭐ | top-p 控制累积概率阈值；限制低概率 token | 平衡质量和多样性 |
| 10 min | 对比重复惩罚（1.0/1.1/1.2） | ⭐⭐⭐ | 惩罚已生成的 token 降低重复 | 理解去重策略 |
| 15 min | 组合测试 + 记录 | ⭐⭐⭐ | 找最佳参数组合 | 调参经验 |

**产出**：一份采样参数 vs 生成质量的对比表

### Day 19：对比不同模型的推理 ⭐⭐

**目标**：体验不同模型的差异

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 试 LLM（Llama/OPT/StarCoder） | ⭐⭐ | 不同模型架构的推理速度和内存差异 | 理解模型差异 |
| 15 min | 试 VLM（LLaVA/VILA） | ⭐⭐ | 多模态推理的额外交互 | Day 13 的实践 |
| 15 min | 记录各模型的 token/s | ⭐⭐ | 对比不同模型在相同硬件上的性能 | 性能基准 |

**产出**：一份不同模型在本地硬件的性能对比表

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
| `llm/Makefile` | 构建系统 | ⭐⭐⭐⭐⭐ | |
| `llm/src/interface.cc` | `main()` 聊天入口 | ⭐⭐⭐⭐⭐ | |
| `kernels/matmul.h` + `matmul_int4.cc` | `int4MatMul()` 调度器 | ⭐⭐⭐⭐⭐ | |
| `llm/src/nn_modules/non_cuda/Int4llamaDecoderLayer.cc` | Llama 解码器层 | ⭐⭐⭐⭐⭐ | |
| `kernels/neon/matmul_neon_int4.cc` | ARM NEON INT4 内核 | ⭐⭐⭐⭐ | |
| `kernels/avx/matmul_avx_int4.cc` | x86 AVX INT4 内核 | ⭐⭐⭐ | |
| `kernels/cuda/gemv_cuda.cu` | CUDA GEMV 内核 | ⭐⭐⭐ | |
| `llm/src/Generate.cc` | Token 采样 | ⭐⭐⭐⭐ | |
| `llm/src/LLaMATokenizer.cc` | Tokenizer | ⭐⭐⭐ | |
| `llm/src/ops/` | 基础算子 | ⭐⭐⭐ | |

## 附录：常见误区

| 误区 | 正解 |
|------|------|
| TinyChatEngine 做量化 | 不做量化，只部署 AWQ/SmoothQuant 已量化好的模型 |
| C++ 代码是自动生成的 | 手写的 C++ 和 CUDA/Metal 代码，不是代码生成 |
| 依赖 BLAS 或 PyTorch | 零依赖（除平台 SDK），所有矩阵乘法手写 SIMD |
| 各平台内核算法相同 | 权重布局、指令、分块策略因平台而异（QM_ARM/QM_x86/QM_CUDA） |
| decode 阶段用 GEMM | 每次只处理 1 个新 token，是向量×矩阵（GEMV），不是矩阵×矩阵（GEMM） |

## 附录 A：面试常问题目

### 基础题（必须答对）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 1 | TinyChatEngine 是什么？和其他推理引擎有什么区别？ | 纯 C++ 零依赖推理引擎，部署 AWQ/SmoothQuant 量化模型。vs llama.cpp：更聚焦 INT4 硬件优化。vs vLLM：不依赖 Python/PyTorch。 | Day 1 |
| 2 | 4 个平台后端怎么选择？ | 编译时通过宏 `QM_ARM`/`QM_x86`/`QM_CUDA`/`QM_METAL` 选择。`int4MatMul()` 内部 `#ifdef` 分发到对应内核。 | Day 2 |
| 3 | 离线权重重排列为什么重要？ | 运行时无需解包：权重已在内存中按 SIMD 位宽对齐，一次 load+shift+accumulate，避免逐元素解包开销。 | Day 6, 9 |
| 4 | Prefill 和 Decode 各有何特点？ | Prefill：并行处理全部 prompt，compute bound，用 GEMM。Decode：逐 token 生成，memory bound，用 GEMV。 | Day 14 |
| 5 | 支持哪些采样策略？ | 贪心(argmax)、temperature 缩放、top-p 核采样、top-k 过滤、重复惩罚。全部在 `Generate.cc` 中实现。 | Day 4 |

### 进阶题（区分水平）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 6 | NEON INT4 权重 128-bit 布局具体怎么排列？ | 每 64 个 INT4 权重为一组（=32 字节=2×128-bit）。`vld4q` 一次加载 4 个 128-bit 寄存器（=64 个 int4）。4 个寄存器的低位字节组成 INT8 值→vmlal 乘累加。 | Day 6 |
| 7 | AVX 相比 NEON 快多少？为什么？ | 理论 2x（256 vs 128 bit）。实际 1.3-1.6x，decode 是 memory bound，带宽是共同瓶颈。prefill 接近 2x。 | Day 8 |
| 8 | KV cache 在 C++ 中怎么管理？ | 预分配连续内存，按层和 head 索引。每步 decode 追加新 token 的 K/V，不重新计算之前的所有 token。 | Day 4, 14 |
| 9 | 如何支持新模型架构？ | (1) `nn_modules/` 增加新模型文件；(2) 实现 decoder 层的 forward；(3) 添加 tokenizer；(4) 确保矩阵乘法复用已有内核。 | Day 10 |
| 10 | 为什么 decode 是 memory bound？ | 每次只计算一个 token 的 GEMV，计算量 O(d_model²) 但权重读取 O(d_model²)。算术强度极低，瓶颈在内存带宽。 | Day 14 |

### 设计题（展示架构能力）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 11 | 如果在 ARM SVE（可变长度 SIMD）上优化，怎么利用？ | SVE 向量长度在 128-2048 bit 可伸缩。(1) 权重布局改为按 Z 寄存器长度分组；(2) 用 `svld1` 替代 `vld4q`；(3) 用 predicate 处理边界，写一份代码适配所有宽度。 | Day 6, 9 |
| 12 | 多用户并发推理怎么设计？ | (1) 共享只读的模型权重（放在 shared memory）；(2) 每个用户独立 KV cache；(3) CPU 多线程池分配线程给不同用户；(4) 每个用户的 decode 是独立的 GEMV 调用。 | Day 2, 7 |
| 13 | 如果只有 4GB 内存，如何部署 7B 模型？ | 7B INT4 ≈ 3.5GB，剩下 0.5GB 给 KV cache（够 ~2000 tokens 上下文）。如果需要更长上下文：(1) INT3 量化（~2.6GB）；(2) GQA 减少 KV cache；(3) CPU offload。 | Day 4, 17 |

## 附录 B：学习达标标准

### 第 1 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能画出 Llama 解码器层的计算图；能说出 4 个平台后端；能编译项目 |
| **优秀** | 能理解 Makefile 的编译宏和平台选择机制；能对比至少 2 种 tokenizer |

### 第 2 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能解释 NEON INT4 的加载和解包过程；能说出离线重排的原因 |
| **优秀** | 能对比 NEON 和 AVX 的权重布局差异；能解释分块策略的 cache 考虑 |

### 第 3 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能追踪 prompt 到生成的完整流程；能画出 VLM 的数据流 |
| **优秀** | 能分析 prefill/decode 的计算-内存边界；能对比 CPU vs GPU GEMV 实现 |

### 第 4 周结束标准（最终验收）

| 级别 | 标准 |
|------|------|
| **达标** | 能编译运行聊天；能对比不同采样参数的效果；能回答 5 道基础题 |
| **优秀** | 能回答全部 13 道面试题；能解释如果要支持新模型需要改什么 |
