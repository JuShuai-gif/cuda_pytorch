# 第 13 章：深度学习模型的可解释性与透明度

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 13 章，第 515–551 页。

---

## 目录

1. [章节概述](#章节概述)
2. [文件索引](#文件索引)
3. [编译与运行](#编译与运行)
4. [技术速查](#技术速查)
5. [PDF 完整内容对照](#pdf-完整内容对照)
6. [注意事项](#注意事项)

---

## 章节概述

当模型影响医疗、信贷、招聘或安全决策时，仅知道"预测是什么"远远不够。利益相关者还需要理解**为什么**做出该预测、对其置信度有多高，以及哪些因素可能改变结果。**可解释性**正是将复杂函数转译为人类可评估和行动的理由。

### 可解释性三大轴

| 轴 | 维度 | 说明 |
|------|------|------|
| 局部 vs 全局 | 范围 | 解释单个决策 vs 整个模型行为 |
| 事后 vs 内嵌 | 时机 | 分析已训练模型 vs 从设计阶段就构建可解释性 |
| 模型无关 vs 模型特定 | 通用性 | 跨架构适用（LIME/SHAP） vs 针对特定模型族（Grad-CAM for CNN） |

### 三大方法对比

| 方法 | 类型 | 原理 | 适用场景 |
|------|------|------|---------|
| LIME | 局部、模型无关、事后 | 在输入邻域扰动采样，拟合稀疏线性代理模型 | 快速、单预测解释，设置简单 |
| SHAP | 局部+全局、模型无关、事后 | 基于合作博弈Shapley值，估算加性特征贡献 | 可审计归因，支持全局汇总 |
| Grad-CAM | 局部、模型特定（CNN）、事后 | 对目标卷积层激活和梯度做加权和，生成类特定热力图 | 视觉模型的空间定位解释 |

### 四大利益相关者需求

| 受众 | 需求 | 解释形式 |
|------|------|---------|
| 临床医生/领域专家 | 个案级解释，领域特征贡献，校准风险 | 显著图 + 置信区间 + OOD 警告 |
| 审计/合规/风险 | 模型卡片、数据沿袭、公平性、决策日志 | 全局汇总 + 已校准度量 |
| 终端用户 | 简短、易懂、可行动 | 主要因素摘要 + 改进建议 |
| 工程师/运维 | 忠实反映模型行为，支持调试 | 显著图、归因模式、分歧信号、队列汇总 |

---

## 文件索引

### 一、可解释性概念框架 — PDF 第 516–519 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `01_explainability_concepts.cpp` | 516–519 | 三轴分类（局部/全局、事后/内嵌、模型无关/特定）、利益相关者需求矩阵、信任/安全/监管连接 | STL |

### 二、LIME 实现 — PDF 第 520–531 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `02_lime_tabular.cpp` | 520–531 | 扰动生成（标准化+高斯噪声）、接近度核加权、加权岭回归（Eigen LDLT解）、top-K 系数提取、局部 R² | Eigen3 |

### 三、KernelSHAP 实现 — PDF 第 531–534 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `03_shap_kernelshap.cpp` | 531–534 | 联盟采样、Shapley 核权重、掩码批量构建、加权回归、基线集选择、加性验证 | Eigen3 |

### 四、Grad-CAM — PDF 第 536–542 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `04_gradcam_cnn.cpp` | 536–542 | SmallCNN（暴露最后卷积激活）、retain_grad、反向传播类得分、通道权重（全局平均梯度）、ReLU加权和、双线性上采样、热力图叠加 | LibTorch, OpenCV |

### 五、不确定性与弃权 — PDF 第 543–544 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `05_uncertainty_and_abstain.cpp` | 543–544 | 校准概率、预测区间、OOD 检测（距背景集距离）、弃权网关、人审路由 | STL |

### 六、模型卡片与伦理 — PDF 第 544–547 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `06_model_card_and_ethics.cpp` | 544–547 | 模型卡片结构、审计日志、原因码、人审覆盖策略、避误导归因 | STL |

---

## 编译与运行

### 环境要求

```bash
C++17 编译器（GCC 11+ / Clang 14+）
CMake 3.22+
LibTorch → $HOME/Downloads/libtorch           # 04_gradcam_cnn.cpp 需要
Eigen 3.4+ → apt install libeigen3-dev            # 02_lime, 03_shap 需要
OpenCV 4.x  → apt install libopencv-dev           # 04_gradcam_cnn.cpp 需要
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行示例

```bash
# 概念
./build/chapter13/explainability_concepts

# LIME & SHAP (需要 Eigen)
./build/chapter13/lime_tabular
./build/chapter13/shap_kernelshap

# Grad-CAM (需要 LibTorch + OpenCV)
./build/chapter13/gradcam_cnn <image_path> <target_class>

# 伦理与治理
./build/chapter13/uncertainty_and_abstain
./build/chapter13/model_card_and_ethics
```

---

## 技术速查

### LIME 核心公式

```
β̂ = argmin Σ π(x_i, x_0) · (f(x_i) - g_β(x_i))² + Ω(β)
```

- `π(x_i, x_0) = exp(-dist²(x_i, x_0) / (2σ²))` — 接近度核
- `g_β` — 稀疏线性代理模型
- `Ω(β)` — 正则化（L1 for 稀疏，L2 for 稳定）

### KernelSHAP 核心公式

Shapley 值（特征 j 的边际贡献）：
```
φ_j = Σ (|S|!(M-|S|-1)! / M!) · [f(S ∪ {j}) - f(S)]
```

核权重（偏向中等规模联盟）：
```
ω(z) = (M-1) / (C(M,k) · k · (M-k))
```

### Grad-CAM 核心公式

通道权重（全局平均梯度）：
```
α_k = (1/Z) · Σ Σ ∂y^c / ∂A^k_{ij}
```

类特定热力图：
```
L^c = ReLU(Σ α_k · A^k)
```

### 可解释性陷阱与防护

| 陷阱 | 防护措施 |
|------|---------|
| 扰动偏离数据流形 | 限制扰动范围，夹紧到有效区间 |
| 特征共线性转移重要性 | 分组相关特征，使用条件变体 |
| 梯度饱和 | 使用 Grad-CAM++ 或 SmoothGrad 叠加 |
| 错误归因（捷径学习） | 随机化权重/标签后的健全性检查 |
| 泄露敏感属性 | 聚合解释，使用粗粒度原因码 |
| 解释不稳定 | 固定 RNG 种子，记录核宽度和背景集 |

### 伦理支柱

| 支柱 | 关键操作 |
|------|---------|
| 传达不确定性 | 校准概率、预测区间、OOD 标记、弃权 |
| 避免误导归因 | 说明范围限制、健全性检查、一致性可视化 |
| 记录已知失效 | 模型卡片、数据合约、边界测试、失效→响应映射 |
| 人工覆盖与审计 | 升级路径、不可变审计日志、原因码、用户追索 |

---

## PDF 完整内容对照

以下是 PDF 第 515–551 页（原书页码）的完整纲要：

| PDF 页（书） | PDF 页（文件） | 内容 | 实现文件 |
|-------------|--------------|------|---------|
| 515–516 | 548 | 章节概述、三轴框架（局部/全局、事后/内嵌、模型无关/特定） | `01_explainability_concepts.cpp` |
| 517 | 549 | 何时需要解释、不同利益相关者需求 | `01_explainability_concepts.cpp` |
| 518–519 | 549–550 | 信任/安全/监管、设计原则（延迟预算、一致性优于聪明） | `01_explainability_concepts.cpp` |
| 520–521 | 550–551 | LIME 概述：邻域近似、代理模型拟合、LIME 公式 | `02_lime_tabular.cpp` |
| 521 | 551 | LIME 示例：表格信用评分、文本情感、图像超像素 | `02_lime_tabular.cpp` 注释 |
| 522–526 | 551–552 | SHAP 概述：Shapley 值、加性归属、联盟机制、基线选择 | `03_shap_kernelshap.cpp` |
| 526 | 552 | Grad-CAM 概述：类特定热力图、通道权重、ReLU 阈值 | `04_gradcam_cnn.cpp` |
| 526–527 | 552–553 | 真实约束下的方法选择与组合 | `note.md` |
| 527–528 | 553 | 从预测函数到解释器：Model 接口（opaque） | `02_lime_tabular.cpp` |
| 528–531 | 553–554 | LIME C++ 实现：LimeConfig、样本生成、加权岭回归（Eigen）、top-K 提取 | `02_lime_tabular.cpp` |
| 531–535 | 554–555 | KernelSHAP C++ 实现：ShapConfig、联盟掩码、Shapley 核权重、掩码批量构建、加权回归 | `03_shap_kernelshap.cpp` |
| 535–536 | 555–556 | 将解释保持在 SLO 内：批量所有查询、预分配、工作上限、缓存、稳定输出 | `note.md` |
| 536 | 556 | 展示什么（与不展示什么） | `06_model_card_and_ethics.cpp` |
| 536 | 556 | 故障排除忠实度：核宽度、背景集、共线性、加性验证 | `02_lime_tabular.cpp` 验证 |
| 536–541 | 556–558 | Grad-CAM 端到端实现：SmallCNN、activate/retain_grad、gradcam_map、热力图叠加 | `04_gradcam_cnn.cpp` |
| 541–542 | 558 | 真实网络上的 Grad-CAM：目标层选择、eval 设置、性能、缩放、限制 | `04_gradcam_cnn.cpp` 注释 |
| 543–544 | 558–559 | 伦理：传达不确定性、校准概率、预测区间、弃权 | `05_uncertainty_and_abstain.cpp` |
| 544 | 559–560 | 避免误导归因：范围限制、健全性检查、防泄露/防博弈 | `06_model_card_and_ethics.cpp` |
| 545 | 560 | 记录已知失效：模型卡片、数据合约、边界测试、失效→响应映射 | `06_model_card_and_ethics.cpp` |
| 545–546 | 560–561 | 人工覆盖与审计：升级/覆盖策略、不可变日志、原因码、PII 隔离 | `06_model_card_and_ethics.cpp` |
| 546–547 | 561 | 付诸实践：API 字段、OOD/不确定性网关、文档、运营监控 | `05_uncertainty_and_abstain.cpp` |
| 547– | 562 | 章节总结、问题、进一步阅读、答案 | — |

---

## 注意事项

### 外部库依赖

| 文件 | 需要的外部库 | 未安装时的行为 |
|------|-------------|---------------|
| `01_explainability_concepts.cpp` | 无（纯 STL） | 始终可编译运行 |
| `02_lime_tabular.cpp` | Eigen 3.4+ | 必需，否则链接失败 |
| `03_shap_kernelshap.cpp` | Eigen 3.4+ | 必需，否则链接失败 |
| `04_gradcam_cnn.cpp` | LibTorch + OpenCV 4.x | CMake 自动跳过 |
| `05_uncertainty_and_abstain.cpp` | 无（纯 STL） | 始终可编译运行 |
| `06_model_card_and_ethics.cpp` | 无（纯 STL） | 始终可编译运行 |

### PDF 中提及但未独立实现的内容

| 知识点 | PDF 页 | 说明 |
|--------|--------|------|
| 图像超像素分割（LIME 图像） | 521 | 需要额外库（如 SLIC），未实现；只提供表格版本 |
| ONNX Runtime 后端 | 参考 | 所有示例使用 Eigen 或 LibTorch；ONNX 集成模式相同 |
| 坐标下降 L1（LASSO） | 528 | `02_lime_tabular.cpp` 使用岭回归（L2）；LASSO 在注释中说明可互换 |

### 其他注意事项

- `02_lime_tabular.cpp` 和 `03_shap_kernelshap.cpp` 中的 Eigen 路径为 `/usr/include/eigen3/`（通过 `libeigen3-dev` 安装）。
- `04_gradcam_cnn.cpp` 使用 LibTorch C++ API，路径为 `$HOME/Downloads/libtorch`。
- LIME 的 `LimeConfig.kernel_width` 和 SHAP 的 `ShapConfig.n_coalitions` 参数需根据实际数据调优。
- 生产环境应缓存 LIME/SHAP 结果；在线 UI 可提供"快速模式"（较少样本），离线审计使用"完整模式"。
- 所有解释 API 应包含 `calibration_score`、`uncertainty` 和 `abstain` 字段。
