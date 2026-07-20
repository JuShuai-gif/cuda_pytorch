# LLM-AWQ 学习计划：每天 45 分钟，从入门到理解核心

**一句话定位**：LLM-AWQ（Activation-aware Weight Quantization）是 MIT HAN Lab 提出的激活感知权重量化方法，通过分析激活分布自动保护重要权重通道，将 LLM 压缩到 INT3/INT4 精度且几乎不损失性能。MLSys 2024 最佳论文。

**总时长**：约 4 周（20 个学习日），每天 45 分钟。

**重要性说明**：
- ⭐⭐⭐⭐⭐ = 必须掌握（不然后续无法理解）
- ⭐⭐⭐⭐   = 核心理解（面试/开发中常见）
- ⭐⭐⭐     = 重要但可先走读（用到再细看）
- ⭐⭐       = 了解即可（高级/特定场景）

---

## 第 1 周：核心算法 — AWQ 是怎么工作的（5 天）

### Day 1：项目骨架 + 快速上手 ⭐⭐⭐⭐⭐

**目标**：知道项目做什么，文件怎么组织，关键入口在哪里

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 通读 `README.md` | ⭐⭐⭐⭐⭐ | 项目定位（INT3/4 LLM 量化）、支持模型列表、性能对比（速度 3x+、内存 3x+） | 所有后续阅读的上下文 |
| 15 min | 看 `awq/entry.py` | ⭐⭐⭐⭐⭐ | 主入口：`run_awq()` 搜索 → 伪量化评估 → 真实量化导出；命令行参数结构 | Day 16 跑 demo 时直接复用 |
| 15 min | 浏览 `scripts/` 目录 | ⭐⭐⭐⭐ | 各模型示例脚本（Llama/DeepSeek/VILA 等），理解"不同模型用同一流程" | Day 5 看 VLM 时找对应脚本 |

**产出**：能说出 AWQ 三步流程（搜索 → 评估 → 导出）和项目目录结构

### Day 2：核心量化流程 ⭐⭐⭐⭐⭐

**目标**：理解 `run_awq()` 从输入到输出的完整流程

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `awq/quantize/pre_quant.py` | ⭐⭐⭐⭐⭐ | `run_awq()`：加载模型 → 收集激活统计 → `auto_scale` 计算缩放 → `auto_clip` 计算裁剪 → 应用量化 | 整个 AWQ 的核心路径 |
| 15 min | 读 `awq/quantize/quantizer.py` | ⭐⭐⭐⭐ | 伪量化 `pseudo_quantize_tensor()` vs 真实量化 `real_quantize_model_weight()`；scale/zero 计算 | Day 8 真实量化部署 |
| 10 min | 读 `awq/utils/calib_data.py` | ⭐⭐⭐ | 校准数据集加载：从 Pile/WikiText 采样 ~128 条短句做校准 | Day 10 理解校准数据选择的影响 |

**产出**：能手绘 AWQ 核心流程：校准数据 → 激活统计 → auto_scale → auto_clip → 量化权重

### Day 3：Auto Scale — 核心创新 ⭐⭐⭐⭐⭐

**目标**：理解 auto_scale 的数学原理和代码实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `awq/quantize/auto_scale.py` | ⭐⭐⭐⭐⭐ | `apply_scale()` 核心逻辑：逐层分析激活幅度 → 识别显著通道 → 乘法缩放因子移动到权重 → 保护显著通道精度 | 面试必问的核心算法 |
| 15 min | 理解"显著通道"概念 | ⭐⭐⭐⭐⭐ | 不是所有权重同等重要；激活幅度大的通道 = 显著通道；量化前放大显著通道，量化后权重吸收缩放 | SmoothQuant 的对比（Day 15） |
| 5 min | 看 `smooth.py` | ⭐⭐⭐ | LM head + LayerNorm 的平滑处理 | 知道和 SmoothQuant 的区别 |

**产出**：能解释为什么"按激活幅度缩放"比"按权重幅度缩放"更好

### Day 4：Auto Clip + 分组量化 ⭐⭐⭐⭐

**目标**：理解裁剪和分组量化的实现细节

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `awq/quantize/auto_clip.py` | ⭐⭐⭐⭐ | 逐层搜索最佳裁剪范围 `[clip_range_min, clip_range_max]`；grid search 遍历候选裁剪值 → 最小化 MSE | 量化超参数搜索的通用技巧 |
| 15 min | 理解分组量化 | ⭐⭐⭐⭐ | 每 128 个权重共享一个 scale（group_size=128）；vs 逐 channel vs 逐 tensor 精度对比 | TinyChatEngine 中如何解包（Day 11） |
| 10 min | 读 `awq/quantize/qmodule.py` | ⭐⭐⭐ | `WQLinear`：INT4 线性层的 PyTorch 实现；前向时反量化权重 + FP16 计算（伪量化模式） | Day 12 自定义 PyTorch 模块 |

**产出**：能解释 auto_clip 如何通过 MSE 最小化找到最优裁剪范围

### Day 5：VLM 量化 + visual encoder ⭐⭐⭐

**目标**：理解视觉语言模型的量化与纯文本模型有哪些不同

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 看 `scripts/vila_example.sh` | ⭐⭐⭐ | VLM 量化入口：量化 LLM + vision encoder；注意 vision 部分有些层不量化 | Day 17 跑 VLM demo |
| 15 min | 看 `tinychat/models/vila_llama.py` / `llava_llama.py` | ⭐⭐⭐ | VLM 模型包装器：LLaVA/VILA 如何拼接 vision tokens + text tokens | 理解多模态推理 |
| 15 min | 对比 VLM vs LLM 量化脚本 | ⭐⭐⭐ | VLM 多了 vision encoder 处理；CLIP 的 LN 层不量化（保持精度） | 知道哪些组件可以量化 |

**产出**：能说出 VLM 量化相比纯文本多了哪些处理

---

## 第 2 周：TinyChat 推理 + CUDA 内核（5 天）

### Day 6：TinyChat 推理概览 ⭐⭐⭐⭐

**目标**：理解量化模型如何在 TinyChat 中跑起来

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 看 `tinychat/models/llama.py` | ⭐⭐⭐⭐ | TinyChat Llama 模型类：`decode()` 核心推理循环；KV cache 管理 | Day 9 推理性能分析 |
| 20 min | 看 `tinychat/modules/fused_attn.py` | ⭐⭐⭐⭐ | 融合注意力模块：把 QKV 投影 + RoPE + attention + output 投影打包在一起 | 理解"融合"对端侧推理的价值 |
| 10 min | 看 `tinychat/modules/fused_mlp.py` | ⭐⭐⭐ | MLP 融合：gate + up + down 三个 INT4 矩阵乘法的融合调用 | Day 11 对应 C++ 融合实现 |

**产出**：能画出 TinyChat 解码器层的计算图（QKV → Attention → MLP）

### Day 7：CUDA GEMM/GEMV 内核 ⭐⭐⭐⭐

**目标**：理解 GPU 上 INT4 矩阵乘法的核心实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 浏览 `awq/kernels/csrc/quantization_new/gemm/` | ⭐⭐⭐⭐ | INT4 GEMM内核目录结构；预填充阶段的矩阵乘法（batch 维度并行） | Day 9 理解 prefill vs decode 瓶颈 |
| 15 min | 浏览 `awq/kernels/csrc/quantization_new/gemv/` | ⭐⭐⭐⭐ | INT4 GEMV 内核；解码阶段（每次只算一个 token 的向量-矩阵乘法） | Day 9 理解"为什么 decode 是 memory bound" |
| 15 min | 理解反量化方式 | ⭐⭐⭐⭐ | 内核中 INT4→FP16 反量化与乘累加融合（避免额外内存读写） | 端侧推理优化的核心技巧 |

**产出**：能区分 GEMM（prefill）和 GEMV（decode）两种计算模式

### Day 8：真实量化模型导出 ⭐⭐⭐

**目标**：理解量化权重如何打包和导出给 TinyChatEngine

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | 看 AWQ 导出脚本 | ⭐⭐⭐ | 权重 int4 打包（2 个 int4 压成 1 个 int8）；scale/zero 按 group 排列 | Day 13 C++ 侧解包 |
| 15 min | 理解 `real_quantize_model_weight()` | ⭐⭐⭐ | 真正执行权重量化（非伪量化）；保存为 `.pt` 格式 | 和伪量化对比理解 |
| 20 min | 看 `scripts/` 中对应模型的导出 | ⭐⭐⭐ | 各模型导出流程：Llama/Mistral/Falcon/StarCoder 等 | 理解模型兼容性 |

**产出**：能说出量化模型文件包含什么（INT4 权重数组 + 逐组 scale/zero）

### Day 9：Prefill vs Decode 性能分析 ⭐⭐⭐⭐

**目标**：理解 LLM 推理的两个阶段及其计算特征

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 理解 Prefill 阶段 | ⭐⭐⭐⭐ | 并行处理全部 prompt tokens；compute bound（瓶颈在计算）；INT4 GEMM 内核 | 为什么 prefill 用 GEMM |
| 20 min | 理解 Decode 阶段 | ⭐⭐⭐⭐ | 自回归生成，每次只处理 1 个新 token；memory bound（瓶颈在内存带宽）；INT4 GEMV 内核 | 为什么 decode 是端侧推理瓶颈 |
| 5 min | KV cache 机制 | ⭐⭐⭐ | 缓存已计算的 K/V 避免重复计算；内存占用随序列长度线性增长 | 端侧推理的内存限制 |

**产出**：能解释为什么 prefill 是 compute bound 而 decode 是 memory bound

### Day 10：校准数据选择 + 量化技巧 ⭐⭐⭐

**目标**：理解校准数据对 AWQ 质量的影响

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 分析 `calib_data.py` 的采样策略 | ⭐⭐⭐ | 128 条 × 512 tokens 的短校准数据；从 Pile/WikiText 中均匀采样 | Day 17 做对比实验 |
| 15 min | 理解校准数据量和精度的关系 | ⭐⭐⭐ | 太少→激活统计不准确；太多→计算成本大；128 条是经验最优 | 量化项目的通用调参经验 |
| 15 min | 看 AWQ 论文关键图 | ⭐⭐⭐ | 通道级激活异常值分布；缩放前后的 MSE 对比 | 加深对"激活感知"的理解 |

**产出**：能解释校准数据大小如何影响量化精度

---

## 第 3 周：C++ 运行时 + 多平台部署（5 天）

### Day 11：TinyChatEngine 架构 ⭐⭐⭐

**目标**：理解纯 C++ 推理引擎的整体设计

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 通读 TinyChatEngine README | ⭐⭐⭐ | 纯 C++ 零依赖；平台抽象（NEON/AVX/CUDA/Metal）；INT4/INT8 支持 | Day 15 理解跨平台设计 |
| 15 min | 看 `kernels/matmul.h` + `matmul_int4.cc` | ⭐⭐⭐ | 矩阵乘法调度器：根据平台宏 `QM_ARM`/`QM_x86`/`QM_CUDA` 分发到不同内核 | Day 12-14 各平台内核入口 |
| 15 min | 看 `kernels/pthread_pool.cc` | ⭐⭐ | CPU 多线程并行；ThreadPool 实现 | 了解即可 |

**产出**：能画出 TinyChatEngine 从上层模型到平台 SIMD 的调用链

### Day 12：ARM NEON 内核 ⭐⭐⭐

**目标**：理解 ARM 平台 INT4 矩阵乘法的 SIMD 实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `kernels/neon/matmul_neon_int4.cc` | ⭐⭐⭐ | INT4 权重在内存中的解包布局（QM_ARM）；`vld4q` 一次加载 64 个 int4 权重；反量化与乘累加融合 | 理解 ARM 端侧推理性能的关键 |
| 10 min | 理解权重重排列 | ⭐⭐⭐ | 离线重排权重以匹配 128-bit NEON SIMD 布局；避免运行时重排开销 | 和 x86 AVX 256-bit 布局对比 |
| 10 min | 看 `kernels/neon/matmul_neon_int8.cc` | ⭐⭐ | INT8 内核作为对比基准；比 INT4 更简单但带宽需求更大 | 理解 INT4 vs INT8 权衡 |

**产出**：能解释为什么 ARM NEON INT4 需要离线权重重排列

### Day 13：x86 AVX + CUDA 内核 ⭐⭐⭐

**目标**：理解 x86 和 GPU 平台的矩阵乘法实现差异

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `kernels/avx/matmul_avx_int4.cc` | ⭐⭐⭐ | x86 AVX 256-bit SIMD 内核；与 NEON 的 128-bit 对比（吞吐量 2x） | Day 15 跨平台性能对比 |
| 15 min | 读 `kernels/cuda/gemv_cuda.cu` | ⭐⭐⭐ | CUDA GEMV 内核：warp 级并行；shared memory 缓存权重 | decode 阶段 GPU 加速 |
| 15 min | 对比 CPU vs GPU 内核设计 | ⭐⭐ | CPU 用 SIMD 向量化 + 多线程；GPU 用 warp + thread block | 不同硬件的并行策略 |

**产出**：能说出 NEON、AVX、CUDA 三种后端的核心区别

### Day 14：推理 Pipeline + Token 生成 ⭐⭐⭐⭐

**目标**：理解从输入文本到输出 token 的完整流程

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 读 `llm/src/interface.cc` | ⭐⭐⭐⭐ | 主聊天循环：输入 prompt → tokenize → 逐 token 生成 → detokenize → 输出 | 理解端到端交互 |
| 15 min | 读 `llm/src/Generate.cc` | ⭐⭐⭐⭐ | 采样策略：贪心解码、temperature、top-p、重复惩罚 | Day 18 改参数体验 |
| 15 min | 看 `llm/src/nn_modules/non_cuda/Int4llamaDecoderLayer.cc` | ⭐⭐⭐ | 解码器层的完整前向：QKV 投影 → RoPE → attention → MLP | Day 15 追踪完整前向 |

**产出**：能完整追踪一个 token 从 tokenizer 到生成的全过程

### Day 15：完整追踪一个 Token 的推理 ⭐⭐⭐⭐⭐

**目标**：把前三周的知识串联起来

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | Tokenize | ⭐⭐⭐⭐⭐ | LlamaTokenizer 将文本切成 token IDs | 入口 |
| 10 min | Embedding | ⭐⭐⭐⭐ | 查表获取 token embedding | 第一层 |
| 15 min | 解码器层 × N | ⭐⭐⭐⭐⭐ | 每层：QKV 投影(INT4 GEMM/GEMV)→Rotary→Attention→MLP(INT4 GEMM/GEMV) | 计算瓶颈所在 |
| 5 min | LM Head | ⭐⭐⭐ | 最后线性层 → logits | 输出 token 概率 |
| 5 min | 采样 | ⭐⭐⭐ | 从 logits 中选择下一个 token（贪心/温度采样/top-p） | 生成结果 |

**产出**：能在白板上完整画出一个 token 从输入到生成的全流程

---

## 第 4 周：实践 + 深入（5 天）

### Day 16：动手跑 AWQ 量化 + Python 推理 ⭐⭐⭐⭐

**目标**：实际跑通一次模型量化

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 5 min | 安装依赖 | ⭐⭐⭐⭐ | `pip install -r requirements.txt` | 必须前置 |
| 15 min | 跑一个小模型（如 TinyLlama）| ⭐⭐⭐⭐ | `python -m awq.entry --model path/to/model` | Day 17 修改实验的基础 |
| 15 min | 检查输出的量化模型 | ⭐⭐⭐ | `.pt` 文件包含 INT4 权重 + scale/zero | 理解产出物 |
| 10 min | 对比量化前后推理输出 | ⭐⭐⭐ | 观察困惑度/logit 变化 | 验证量化精度 |

**产出**：成功完成一个小模型的 INT4 量化

### Day 17：修改参数 + 观察影响 ⭐⭐⭐⭐

**目标**：通过小修改加深理解

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 修改 group_size（128→64→256） | ⭐⭐⭐⭐ | 观察精度 vs 存储的 trade-off；更小的组 = 更多 scale = 更大模型但更高精度 | 理解分组量化的权衡 |
| 15 min | 修改校准数据量（128→32→512） | ⭐⭐⭐⭐ | 观察激活统计的稳定性；太少的数据导致 auto_scale 不稳定 | 理解校准数据的重要性 |
| 15 min | 禁用 auto_clip 对比 | ⭐⭐⭐⭐ | 观察没有 auto_clip 时量化精度的下降 | 验证 auto_clip 的价值 |

**产出**：亲眼看到各超参数对量化精度的影响

### Day 18：TinyChatEngine 编译 + 聊天 ⭐⭐⭐

**目标**：编译并运行 C++ 推理引擎

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | 编译 TinyChatEngine | ⭐⭐⭐ | `cd llm && make`；检查 NEON/AVX 编译宏 | 理解构建流程 |
| 15 min | 下载模型 + 运行聊天 | ⭐⭐⭐ | `llm/scripts/chat.sh` 下载模型并启动聊天 | 端到端验证 |
| 10 min | 调整采样参数 | ⭐⭐⭐ | temperature=0.7 vs 1.0 vs 0.5 观察生成质量 | 理解采样策略 |
| 10 min | 观察内存和速度 | ⭐⭐ | CPU 利用率、内存占用、token/s | 对照理解 Day 9 的分析 |

**产出**：成功在本地设备上运行量化 LLM 聊天

### Day 19：多模态推理（可选）⭐⭐

**目标**：体验 VLM 量化推理

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 跑 VILA/LLaVA 量化 | ⭐⭐ | vision encoder 量化 + LLM 量化的组合 | 理解多模态部署 |
| 20 min | 观察 visual tokens 流 | ⭐⭐ | CLIP 编码图像 → visual tokens → 送入 LLM → 生成回复 | 理解 VLM 数据流 |

**产出**：能用图片和量化模型进行多模态对话

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
| `awq/entry.py` | `run_awq()` 主入口 | ⭐⭐⭐⭐⭐ | |
| `awq/quantize/pre_quant.py` | `run_awq()` / `apply_awq()` 核心流程 | ⭐⭐⭐⭐⭐ | |
| `awq/quantize/auto_scale.py` | `apply_scale()` 自动缩放 | ⭐⭐⭐⭐⭐ | |
| `awq/quantize/auto_clip.py` | 最佳裁剪范围搜索 | ⭐⭐⭐⭐ | |
| `awq/quantize/quantizer.py` | 伪量化 vs 真实量化 | ⭐⭐⭐⭐ | |
| `awq/quantize/qmodule.py` | WQLinear INT4 模块 | ⭐⭐⭐ | |
| `tinychat/models/llama.py` | Llama 推理模型类 | ⭐⭐⭐⭐ | |
| `kernels/neon/matmul_neon_int4.cc` | ARM INT4 矩阵乘法 | ⭐⭐⭐ | |
| `kernels/avx/matmul_avx_int4.cc` | x86 INT4 矩阵乘法 | ⭐⭐⭐ | |
| `kernels/cuda/gemv_cuda.cu` | CUDA GEMV 内核 | ⭐⭐⭐ | |

## 附录：常见误区

| 误区 | 正解 |
|------|------|
| AWQ 和 SmoothQuant 一样 | AWQ 只量化权重（W4A16），SmoothQuant 同时量化权重和激活（W8A8） |
| 所有层都需要 auto_scale | LayerNorm/LM Head 用专门的 smooth 处理，不参与 auto_scale |
| 量化后必须用 CUDA 推理 | TinyChatEngine 支持纯 CPU（NEON/AVX）推理，无需 GPU |
| group_size 越小精度越高 | 对的，但存储开销也更大。group_size=128 是精度/大小的最佳平衡点 |
| 校准数据越多越好 | 128 条短文本足够稳定；更多数据增加计算开销但精度提升很小 |

## 附录 A：面试常问题目

### 基础题（必须答对）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 1 | AWQ 是什么？核心创新点？ | 激活感知权重量化：分析激活分布→识别显著通道→缩放保护精度。W4A16（仅量化权重）。比 GPTQ/RTN 精度更高。 | Day 1 |
| 2 | 为什么按激活幅度而不是权重幅度判断重要性？ | 权重大≠对输出影响大。激活幅度大的通道在矩阵乘法中产生更大的内积贡献，量化这些通道的误差会被放大。 | Day 3 |
| 3 | auto_scale 的数学公式是什么？ | `s = act_scale^alpha`（alpha=1.0）。按激活幅度缩放权重通道，保护显著通道。最优缩放因子通过 grid search 最小化 MSE 得到。 | Day 3 |
| 4 | 分组量化和逐通道量化的区别？ | 逐通道：每个输出通道一个 scale（group_size 是整个输入通道）。分组量化：每组 K 个输入元素共享 scale。AWQ 默认 group_size=128。 | Day 4 |
| 5 | auto_clip 做什么？ | 逐层搜索最优裁剪范围，通过 grid search 遍历候选 clip 值，选择 MSE 最小的。解决 outlier 导致的量化范围浪费问题。 | Day 4 |

### 进阶题（区分水平）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 6 | AWQ vs SmoothQuant 的核心区别？ | AWQ 只量化权重(W4A16)，SmoothQuant 量化权重+激活(W8A8)。两者都用"平滑"技巧迁移量化难度。AWQ 精度损失更小但加速比不同。 | Day 3, 15 |
| 7 | 为什么 prefill 用 GEMM 而 decode 用 GEMV？ | Prefill 一次输入 N 个 tokens，是矩阵×矩阵（GEMM）。Decode 每次只输入 1 个 token，是向量×矩阵（GEMV）。前者 compute bound，后者 memory bound。 | Day 7, 9 |
| 8 | INT4 在 SIMD 上如何高效解包？ | 离线将 INT4 权重按 SIMD 位宽重排（ARM 128-bit, x86 256-bit）。运行时用 SIMD shuffle/zip 指令高效解包 2 个 INT4→1 个 INT8，然后 vmlal 乘累加。 | Day 12 |
| 9 | KV cache 在端侧推理中的内存压力？ | 每层每个 token 存 K+V（各 head_dim × n_kv_heads）。长序列下 KV cache 可能是模型权重的数倍。GQA/MQA 是缓解方案。 | Day 9 |
| 10 | 为什么 VLM 的 vision encoder 某些层不量化？ | CLIP 的 LN 层对精度极度敏感，量化后图像特征质量下降明显。视觉编码器的某些激活层同理。 | Day 5 |

### 设计题（展示架构能力）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 11 | 如果要支持一个新 LLM 架构（如 Mamba），需要改哪些文件？ | (1) `tinychat/models/` 新建模型包装器；(2) `awq/entry.py` 添加模型加载逻辑；(3) `scripts/` 添加示例脚本；(4) 如果涉及新算子，更新 `tinychat/modules/` | Day 1, 6 |
| 12 | 在 4GB 内存的树莓派上部署 7B 模型有什么优化手段？ | (1) INT4 量化（模型~4GB→~1GB）；(2) 用 NEON 内核；(3) 考虑 INT3 进一步压缩；(4) 4-bit KV cache；(5) CPU offload（部分层放 swap） | Day 9, 12 |
| 13 | AWQ 量化后模型精度下降怎么排查？ | (1) 检查校准数据集是否具代表性；(2) 调整 auto_scale 的 alpha 参数；(3) 增大 group_size；(4) 确认某些敏感层不被量化；(5) 检查 auto_clip 裁掉太多值 | Day 17 |

## 附录 B：学习达标标准

### 第 1 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能手绘 AWQ 量化流程（校准→auto_scale→auto_clip→量化）；能解释"激活感知"的动机 |
| **优秀** | 能推导 auto_scale 的 MSE 优化目标；能解释为什么 alpha=1.0 是最优选择 |

### 第 2 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能区分 prefill/decoder 阶段；能说出 GEMM vs GEMV 的应用场景；理解 KV cache 原理 |
| **优秀** | 能分析 prefill 的矩阵维度变化；估算给定模型的 KV cache 内存占用 |

### 第 3 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能说出 NEON/AVX/CUDA 三种后端的关键区别；能追踪一个 token 的完整推理路径 |
| **优秀** | 能解释 INT4 权重布局如何与 SIMD 指令配合；能分析 decode 阶段的性能瓶颈 |

### 第 4 周结束标准（最终验收）

| 级别 | 标准 |
|------|------|
| **达标** | 能跑通 AWQ 量化；能编译 TinyChatEngine 运行聊天；能回答 5 道基础题 |
| **优秀** | 能回答所有 13 道面试题；能解释如果要量化和部署一个新模型需要做什么 |
