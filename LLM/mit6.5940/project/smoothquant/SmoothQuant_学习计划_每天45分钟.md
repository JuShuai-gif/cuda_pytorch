# SmoothQuant 学习计划：每天 45 分钟，从入门到理解核心

**一句话定位**：SmoothQuant 是 MIT HAN Lab 提出的训练无关、精度保持的 W8A8 量化方法。通过数学等效变换将激活的量化难度"平滑"迁移到权重上，解决 LLM 激活异常值导致量化精度损失的问题。支持最高 530B 参数模型。ICML 2023。

**总时长**：约 3 周（15 个学习日），每天 45 分钟。

**重要性说明**：
- ⭐⭐⭐⭐⭐ = 必须掌握（不然后续无法理解）
- ⭐⭐⭐⭐   = 核心理解（面试/开发中常见）
- ⭐⭐⭐     = 重要但可先走读（用到再细看）
- ⭐⭐       = 了解即可（高级/特定场景）

---

## 第 1 周：核心算法 — SmoothQuant 怎么工作（5 天）

### Day 1：项目骨架 + 快速上手 ⭐⭐⭐⭐⭐

**目标**：知道项目做什么，文件怎么组织

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 通读 `README.md` | ⭐⭐⭐⭐⭐ | 项目定位（W8A8 LLM 量化）、论文核心结果（精度损失 <0.5 ppl）、支持模型列表 | 所有后续阅读的上下文 |
| 15 min | 浏览项目目录 | ⭐⭐⭐⭐ | `smoothquant/` 核心代码；`examples/` Jupyter notebooks + 脚本；`act_scales/` 预计算激活统计 | Day 2-5 定向阅读 |
| 15 min | 看 `smoothquant/ppl_eval.py` | ⭐⭐⭐⭐ | 困惑度评估脚本：支持 Llama/Mistral/Mixtral/Falcon/OPT/BLOOM | Day 14 跑评估时直接看 |

**产出**：能说出 SmoothQuant 支持哪些模型，能跑通困惑度评估命令

### Day 2：核心算法 — smooth 变换 ⭐⭐⭐⭐⭐

**目标**：理解 SmoothQuant 的数学本质

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `smoothquant/smooth.py` | ⭐⭐⭐⭐⭐ | `smooth_ln_fcs()` 和 `smooth_ln_fcs_llama_like()` 两个入口；scale 因子计算 `s = act_scale^alpha / weight_scale^(1-alpha)`；迁移强度 alpha 参数 | 整个项目的核心，面试必问 |
| 10 min | 理解数学变换 | ⭐⭐⭐⭐⭐ | `Y = XW = (X · diag(s)^(-1)) · (diag(s) · W)`；scale 因子如何平滑异常值 | Day 3 和代码实现对照 |
| 10 min | 理解 alpha 参数 | ⭐⭐⭐⭐ | alpha=0：全部迁移到权重；alpha=1：全部保留在激活；通常 alpha=0.5~0.9 | Day 14 调参实验 |

**产出**：能在白板上推导 SmoothQuant 的数学变换，解释每一部分的意义

### Day 3：伪量化实现 ⭐⭐⭐⭐⭐

**目标**：理解量化如何模拟（模拟推理时会遇到什么精度损失）

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `smoothquant/fake_quant.py` | ⭐⭐⭐⭐⭐ | `W8A8Linear`：权重 per-channel INT8 + 激活 per-token INT8；反量化 + FP16 计算；三种量化粒度（per-channel/per-token/per-tensor） | Day 12 对比真实量化 |
| 15 min | 理解 per-token 动态量化 | ⭐⭐⭐⭐ | 每个 token 独立计算 scale，无需校准数据；运行时开销小（只需求 absmax） | 为什么激活量化选 per-token |
| 10 min | 理解 static vs dynamic 量化 | ⭐⭐⭐ | per-tensor static 需要校准数据统计 scale；per-token dynamic 运行时计算 | 两种量化模式的适用场景 |

**产出**：能用代码实现一个简易的 W8A8 伪量化模块

### Day 4：校准数据 + 激活统计收集 ⭐⭐⭐⭐

**目标**：理解激活 scale 如何获得

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `smoothquant/calibration.py` | ⭐⭐⭐⭐ | `get_act_scale()`：在校准数据上跑前向 → 记录每层激活幅度的通道级统计（per-channel max） | Day 2 的 smooth() 使用这些 scale |
| 15 min | 看 `examples/generate_act_scales.py` | ⭐⭐⭐⭐ | 批量生成所有模型的激活 scale；按模型名保存到 `act_scales/` 目录 | Day 13 自己为新模型生成 scale |
| 10 min | 看 `act_scales/` 目录 | ⭐⭐⭐ | 预计算好的 scale 文件（Llama/Mistral 等）；理解文件格式 | 直接用现成的做实验 |

**产出**：能为一个新模型生成激活 scale 文件

### Day 5：不同 LLM 架构的适配 ⭐⭐⭐⭐

**目标**：理解 SmoothQuant 如何适配不同 transformer 变体

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 对比 `smooth_ln_fcs()` vs `smooth_ln_fcs_llama_like()` | ⭐⭐⭐⭐ | OPT 架构：LayerNorm→QKV project + OutFc；Llama 架构：RMSNorm→QKV project；不同位置需要不同平滑组合 | 新架构适配 |
| 15 min | 理解需要平滑的组合 | ⭐⭐⭐ | LN({输入}) → QKV/Wi 层({输出})：通过 scale 把 LN 的输出缩放和 QKV 的输入缩放配对 | 理解"平滑"的作用域 |
| 10 min | 看 Falcon/BLOOM/Mistral 的特殊处理 | ⭐⭐⭐ | 各架构的 LN 位置和映射关系不同 | 知道适配新模型的思路 |

**产出**：能为一种新 LLM 架构写出 SmoothQuant 的适配代码

---

## 第 2 周：真实 INT8 部署 + 评估（5 天）

### Day 6：困惑度评估系统 ⭐⭐⭐⭐

**目标**：理解如何衡量量化模型的质量

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `smoothquant/ppl_eval.py` 详细逻辑 | ⭐⭐⭐⭐ | 数据集（WikiText-2/PTB/C4）、stride 滑动窗口、per-token PPL 计算 | Day 14 做全评估时对照理解 |
| 10 min | 理解 perplexity 指标 | ⭐⭐⭐ | `PPL = exp(cross_entropy)`；越低越好；不同模型/dataset 的基线不同 | 论文中量化的主要指标 |
| 15 min | 理解评估脚本的命令行接口 | ⭐⭐⭐ | `--model-path`、`--smoothquant`、`--per-token`、`--per-channel` 等参数含义 | Day 14 跑实验时调参 |

**产出**：能解释困惑度指标，能运行完整的 PPL 评估

### Day 7：量化 OPT 模型类 ⭐⭐⭐

**目标**：理解 HuggingFace OPT 模型的 SmoothQuant 实现

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 25 min | 读 `smoothquant/opt.py` | ⭐⭐⭐ | `QuantOPTModel` 继承 HuggingFace OPT；替换线性层为 W8A8 量化版本；引入 "bundle scales" 机制 | 理解量化如何嵌入框架 |
| 10 min | 理解 bundle scales | ⭐⭐⭐ | 平滑后 weight scale 和 absmax 绑在一起，不需要额外存储 | 存储开销几乎为零 |
| 10 min | 看模型加载方式 | ⭐⭐ | `smooth_and_quant()` 直接对已平滑模型进行伪量化 | 走读即可 |

**产出**：能说明伪量化模型是如何嵌入 HuggingFace 框架的

### Day 8：真实 INT8 推理部署 ⭐⭐⭐⭐

**目标**：理解从伪量化到真实硬件加速的桥接

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 看 `examples/export_int8_model.py` | ⭐⭐⭐⭐ | 导出真正的 INT8 模型（不是伪量化）：权重 INT8 存储 + 运行时反量化 | Day 11 理解真实加速 |
| 15 min | 看 `examples/smoothquant_opt_real_int8_demo.ipynb` | ⭐⭐⭐ | 真实 INT8 推理 demo：延迟对比、内存对比；用 CUTLASS INT8 GEMM 代替模拟量化 | Day 10 性能分析 |
| 10 min | 理解 CUTLASS 集成的角色 | ⭐⭐⭐ | `torch-int` 包将 CUTLASS INT8 GEMM 包装为 PyTorch 操作 | 真加速 vs 伪量化的区别 |

**产出**：能说出伪量化和真实 INT8 推理的区别

### Day 9：Jupyter Demo 阅读 ⭐⭐⭐

**目标**：通过 Notebook 理解实践用法

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 读 `examples/smoothquant_llama_demo.ipynb` | ⭐⭐⭐ | LLaMA 完整流程：加载模型 → 生成 act scale → smooth → 伪量化 → 评估困惑度 | Day 13 做实验时的模板 |
| 15 min | 读 `examples/smoothquant_opt_real_int8_demo.ipynb` | ⭐⭐⭐ | OPT 真实 INT8：加载平滑后模型 → 导出 INT8 → 对比延迟 | 理解端到端速度收益 |
| 10 min | 对比两个 Notebook | ⭐⭐ | 伪量化（精度验证）vs 真实 INT8（速度验证） | 知道两种验证方式 |

**产出**：能独立跑通 LLaMA 的 SmoothQuant 伪量化 + 评估

### Day 10：性能分析 + 精度对比 ⭐⭐⭐

**目标**：理解量化后的速度和精度权衡

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 理解 W8A8 的计算特性 | ⭐⭐⭐ | INT8×INT8 矩阵乘法：速度理论上 2x（INT8 vs FP16），内存 2x 节约（INT8 vs FP16）；实际接近 1.5-1.8x | Day 11 硬件的理论峰值 |
| 15 min | 精度损失分析 | ⭐⭐⭐ | WikiText-2 perplexity 损失 <0.5（6.7B→13B→30B pattern）；alpha 参数的影响曲线 | 论文主要结果 |
| 15 min | 对比其他量化方法的精度 | ⭐⭐ | vs LLM.int8()（混合精度）、vs GPTQ（W4A16）、vs AWQ（W4A16） | 知道 SmoothQuant 的定位 |

**产出**：能画出 SmoothQuant 在不同 alpha 值下的精度曲线

---

## 第 3 周：实践 + 深入（5 天）

### Day 11：真实 INT8 GEMM 内核 ⭐⭐⭐

**目标**：理解真实硬件上的 INT8 加速原理

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 理解 CUTLASS INT8 GEMM | ⭐⭐⭐ | W INT8 + A INT8 → 累加器 INT32 → 反量化 FP16；数据搬运和计算的 pipeline | Day 8 的底层实现 |
| 15 min | 理解 tensor core INT8 | ⭐⭐⭐ | NVIDIA A100/H100 的 tensor core 支持 INT8；理论峰值 624 TOPS (A100 INT8) | 理解硬件上限 |
| 10 min | 与 W4A16 的对比 | ⭐⭐ | W8A8 = 权重+激活都 INT8 = 2× 带宽；W4A16 = 权重 INT4 + 激活 FP16 = 4× 权重压缩 | 理解 AWQ vs SmoothQuant |

**产出**：能解释 tensor core 上 INT8 GEMM 的 tile 计算模式

### Day 12：代码走读 — 端到端流程 ⭐⭐⭐⭐

**目标**：阅读完整代码，理解从输入到输出的每一步

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 追踪 LLaMA smooth 流程 | ⭐⭐⭐⭐ | 加载模型 → hook 注册（捕获每层输入）→ 校准 → get_act_scale() → smooth_ln_fcs() | Day 13 自己改代码 |
| 15 min | 追踪伪量化替换过程 | ⭐⭐⭐ | 替换 nn.Linear → W8A8Linear（权重 per-channel INT8 + 激活 per-token INT8） | 理解模型修改模式 |
| 15 min | 追踪评估流程 | ⭐⭐⭐ | 数据集加载 → 逐步推理 → 累积 cross entropy → 计算 PPL | Day 14 理解评估结果 |

**产出**：能手动追踪一遍完整的 smooth+quant+eval 代码执行路径

### Day 13：动手修改参数实验 ⭐⭐⭐⭐

**目标**：通过修改参数加深对核心算法的理解

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 15 min | 修改 alpha 参数（0.3 / 0.5 / 0.7 / 0.9） | ⭐⭐⭐⭐ | 观察 PPL 变化；alpha 越大→更多量化难度留在激活→精度下降 | 验证 Day 2 的理论 |
| 10 min | 切换量化粒度 | ⭐⭐⭐ | per-token vs per-tensor 对激活量化；per-channel vs per-tensor 对权重量化 | 理解粒度的影响 |
| 10 min | 跳过某些层的平滑 | ⭐⭐⭐ | 观察某些层不平滑对整体精度的影响 | 理解哪些层是关键 |
| 10 min | 记录实验结果 | ⭐⭐⭐ | 总结最佳的参数组合 | 量化项目的通用流程 |

**产出**：有一份完整的参数对比实验结果表

### Day 14：完整模型评估 ⭐⭐⭐⭐

**目标**：完整跑通一个模型的评估，产出报告

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 10 min | 选择评估模型 | ⭐⭐⭐ | Llama-7B/13B 或 Mistral-7B（推荐 7B 级别，方便实验） | Day 13 的实验基础 |
| 15 min | 跑完整 PPL 评估 | ⭐⭐⭐⭐ | WikiText-2 / PTB / C4 三个数据集 | 论文级评估 |
| 10 min | 对比 FP16 基线 | ⭐⭐⭐ | 量化前后困惑度差异 | 量化质量判断 |
| 10 min | 总结报告 | ⭐⭐⭐ | 精度损失 <0.5 PPL 即达标 | 输出 |

**产出**：一份完整的模型量化评估报告

### Day 15：复习 + 自测 ⭐⭐⭐⭐⭐

**目标**：检验理解程度

| 时间段 | 内容 | 重要性 | 知识点 | 后续用到的地方 |
|--------|------|--------|--------|--------------|
| 20 min | 自测基础 + 进阶面试题 | ⭐⭐⭐⭐⭐ | 前 10 道面试题 | 核心验收 |
| 25 min | 设计题 + 综合 | ⭐⭐⭐⭐⭐ | 13 道题全部 | 全面验收 |

**产出**：确认自己掌握了 SmoothQuant 的核心内容

---

## 附录：关键文件速查表

| 文件 | 核心函数 | 重要性 | 你的进度 |
|------|---------|--------|---------|
| `smoothquant/smooth.py` | `smooth_ln_fcs()` / `smooth_ln_fcs_llama_like()` | ⭐⭐⭐⭐⭐ | |
| `smoothquant/fake_quant.py` | `W8A8Linear` / `quantize_weight/activation` | ⭐⭐⭐⭐⭐ | |
| `smoothquant/calibration.py` | `get_act_scale()` | ⭐⭐⭐⭐ | |
| `smoothquant/ppl_eval.py` | 困惑度评估入口 | ⭐⭐⭐⭐ | |
| `smoothquant/opt.py` | `QuantOPTModel` | ⭐⭐⭐ | |
| `examples/generate_act_scales.py` | 生成激活 scale | ⭐⭐⭐ | |
| `examples/export_int8_model.py` | 导出 INT8 模型 | ⭐⭐⭐ | |
| `examples/smoothquant_llama_demo.ipynb` | LLaMA 完整 demo | ⭐⭐⭐ | |

## 附录：常见误区

| 误区 | 正解 |
|------|------|
| SmoothQuant 量化权重 | 同时量化权重和激活（W8A8），不是只量化权重 |
| SmoothQuant 需要重新训练 | 完全训练无关的后训练量化（PTQ），不需要任何微调 |
| SmoothQuant = AWQ | SmoothQuant 是 W8A8（激活也量化），AWQ 是 W4A16（只量化权重）。原理也不同：SmoothQuant 数学变换迁移难度，AWQ 缩放保护显著通道 |
| alpha=0.5 一定最优 | alpha 是超参数，不同模型最优值不同（0.5-0.7 通常较好）。需要在验证集上调参 |
| 伪量化和真实 INT8 一样 | 伪量化用 FP16 计算模拟量化误差（验证精度）；真实 INT8 用 CUTLASS 硬件加速（验证速度）。计算和存储方式完全不同 |

## 附录 A：面试常问题目

### 基础题（必须答对）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 1 | SmoothQuant 是什么？解决什么问题？ | 训练无关的 W8A8 PTQ 方法。解决 LLM 激活中存在通道级异常值导致量化困难的问题。数学变换 Y=X·diag(s)^(-1)·diag(s)·W 将激活的量化难度移给权重。 | Day 1, 2 |
| 2 | 为什么 LLM 激活量化比权重量化难？ | 权重值分布均匀，易量化。激活中有通道级系统性异常值（某些通道的值比其他大 100-1000x），用统一的 scale 量化会损失精度。 | Day 2 |
| 3 | smooth 变换的数学原理？ | `Y = XW = (X·diag(s)^(-1))·(diag(s)·W)`。激活乘以缩放因子的逆（变小→更平滑），权重乘以缩放因子（变大→但分布均匀）。整体计算等效。 | Day 2 |
| 4 | scale 因子 s 怎么计算？ | `s = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)`。alpha 控制迁移强度：alpha→1 保留在激活；alpha→0 全部移给权重。 | Day 2 |
| 5 | per-token dynamic 量化 vs per-tensor static 量化？ | Per-token：每个 token 独立 scale，运行时计算 absmax，不需要校准数据。Per-tensor：整个张量统一 scale，需要校准数据，离线计算。 | Day 3 |

### 进阶题（区分水平）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 6 | 不同模型架构（OPT vs Llama）的 smooth 处理有何不同？ | OPT：LN 后自注意力(输入)和 FFN(输入) 需要独立平滑。Llama：RMSNorm 后自注意力，需要分别处理 QKV 投影和 output 投影。核心思路一致但映射关系不同。 | Day 5 |
| 7 | SmoothQuant 的精度如何随模型规模变化？ | 模型越大，激活异常值越严重，SmoothQuant 的平滑效果越明显。对 100B+ 模型 W8A8 精度损失可忽略，因为权重分布也更均匀。 | Day 10 |
| 8 | 为什么选择 per-channel 权重量化 + per-token 激活量化？ | 权重 per-channel：每个输出通道独立 scale，精度最高。激活 per-token：运行时动态计算，无需校准数据，避免了校准数据偏移问题。 | Day 3 |
| 9 | SmoothQuant 比 LLM.int8() 好在哪里？ | LLM.int8() 用混合精度（大部分 INT8 + 异常值列 FP16），SmoothQuant 纯 INT8 计算，无混合精度开销，速度更快（1.5-1.8x vs 1.2x）。 | Day 10 |
| 10 | 伪量化时为什么用 FP16 累加？ | 模拟真实 INT8 推理的精度：INT8×INT8→INT32 累加→反量化到 FP16/FP32。FP16 累加接近 INT32 累加精度，但无法完全模拟截断误差。 | Day 3, 11 |

### 设计题（展示架构能力）

| # | 问题 | 参考回答要点 | 学习日 |
|---|------|------------|--------|
| 11 | 如果要适配 Mixtral MoE 架构，怎么改？ | 核心不变（同一数学变换）。MOE 的 expert 层需要用 `mixtral` 变量名模式适配。Router 层不需要量化（计算量小）。 | Day 5 |
| 12 | 在新模型上，如何选择最优 alpha？ | 网格搜索 alpha=[0.1, 0.3, 0.5, 0.7, 0.9]，每个值跑一次 pseudo-quant eval，选困惑度最低的。也可针对不同层用不同 alpha（更精准但更复杂）。 | Day 13 |
| 13 | SmoothQuant W8A8 和 AWQ W4A16 在 ARM 端侧的适用场景？ | W8A8 激活也要量化→内存带宽需求更低；W4A16 激活保持 FP16→计算精度更高但内存稍大。ARM NEON 支持 INT8 SIMD，W8A8 的矩阵乘法可以全 INT8，速度更快。 | Day 11 |

## 附录 B：学习达标标准

### 第 1 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能手写 smooth 变换的数学推导；能解释 alpha 参数的作用；能区分三种量化粒度 |
| **优秀** | 能适配新模型写 smooth 代码；能解释为什么 per-token dynamic 比 per-tensor static 更适合 LLM |

### 第 2 周结束标准

| 级别 | 标准 |
|------|------|
| **达标** | 能运行困惑度评估；能区分伪量化和真实 INT8 推理；能导出 INT8 模型 |
| **优秀** | 能分析伪量化和真实 INT8 的精度差异来源 |

### 第 3 周结束标准（最终验收）

| 级别 | 标准 |
|------|------|
| **达标** | 独立完成一个模型的完整量化+评估；回答 5 道基础题 |
| **优秀** | 能回答全部 13 道面试题；能独立为新模型进行 SmoothQuant 量化和调参 |
