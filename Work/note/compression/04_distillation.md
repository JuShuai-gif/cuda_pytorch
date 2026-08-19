# 04｜模型蒸馏：Teacher → Student 的 soft label 信息增益

## 本模块解决的问题

大模型精度高但上不了边缘设备，小模型能部署但精度不够。蒸馏是"让大模型把知识教给小模型"的方法。本章回答：

```text
蒸馏到底在教什么？（不是 hard label，是 soft logits）
温度 T 起什么作用？为什么 T 越大信息越"软"？
为什么蒸馏在 few-shot 场景下增益最大？
large model → small edge model 的精度/延迟/显存权衡是多少？
```

配套代码：`src/compression/distillation/`（MNIST 上 logit distillation）。

---

## 1. Teacher-Student 框架

```text
训练期：
  Teacher（大模型）  ──soft logits──►  蒸馏损失（KL + 温度）
        │                                    │
        └──────────── hard labels ────────────┤
                                              ▼
                                    Student（小模型）

部署期：只部署 Student（小、快、省显存）
```

两种蒸馏损失：

```text
Logit distillation   对齐输出 logits（本章实现）
Feature distillation 对齐中间层特征（更细，见第 5 节）
```

---

## 2. 核心：soft label 比 hard label 信息更多

hard label（one-hot）只说"这是数字 3"。soft label（teacher 的软化 logits）还说"这个 3 有 70% 像 3、20% 像 8、8% 像 5"。**类间相似性是 hard label 完全没有的信息**，而它正是小模型泛化需要的：

```text
hard label：只给正确答案，其余全 0
soft label：给整个概率分布（类间相似性、歧义、边界）
```

**温度 T 控制"软硬程度"**：

```text
soft_i = softmax(z_i / T)

T → 1   接近原始 softmax（接近 hard）
T → 大   分布更平坦（类间差异被放大 → 相似性信息更突出）
```

### 本机实测（温度 sweep，few-shot 2000 样本）

```text
T=1     acc 0.914
T=2     acc 0.924
T=4     acc 0.939
T=8     acc 0.940
T=20    acc 0.946
```

温度越高，soft label 越平坦，student 学到的类间相似性越充分，精度越高（但过高也会让正确类信号过弱，存在最优区间）。

---

## 3. 关键发现：蒸馏在 few-shot 下增益最大

这是本模块最有价值的实测洞察。我最初用**全量 MNIST（60000 样本）**做蒸馏，结果蒸馏**没有收益**（甚至更差）：

```text
全量数据：student direct 98% ≈ teacher 98%，蒸馏增益 ≈ 0（甚至为负）
```

原因：全量数据下，小模型直接训练就到 98%，teacher 的 soft label 没有额外信息可给（甚至把 teacher 的 2% 错误蒸馏进 student）。

换到 **few-shot（2000 样本）**：

```text
n_train=1000: direct 88.1% → distill 90.4%   (+2.4%)
n_train=2000: direct 90.5% → distill 93.9%   (+3.4%)
n_train=4000: direct 93.3% → distill 95.7%   (+2.4%)
```

**结论：soft label 本质是一种"数据增强/正则化"**。当训练数据少、小模型会过拟合时，teacher 的 soft label 提供了额外的泛化信号；当数据充足时，这个信号就没用了。这就是蒸馏的工业定位——**它解决的是"数据少 + 模型小"的边缘场景，不是"无脑提精度"**。

---

## 4. 实测权衡（MNIST，2000 样本）

| 模型 | 精度 | 参数量 | latency |
|---|---|---|---|
| Teacher（512×3） | 97.7% | 932k | 94.2us |
| Student direct | 90.5% | 269k | 68.6us |
| Student distilled | **93.9%** | 269k | 68.6us |

蒸馏让 student 在**参数量和 latency 都不变**的情况下，精度从 90.5% 提到 93.9%。这是"免费"的精度提升——代价只在训练期（需要 teacher 跑一遍）。

---

## 5. Feature Distillation（工业补充）

Logit distillation 只对齐最终输出。Feature distillation 对齐**中间层特征**：

```text
L_feature = || f_student(x) - f_teacher(x) ||²   （中间特征对齐）
```

用途：teacher 和 student 架构差异大（如 CNN → Transformer）时，中间特征对齐能传递更细粒度的知识。代价是实现复杂（需要对齐 feature map 的维度），且不稳定。工业上 logit distillation 是主力，feature distillation 是补充。

---

## 6. 工业意义总结

蒸馏在 AI Infra 里的真实位置：

```text
大模型（云端训练，精度高）
   ↓ 蒸馏
小模型（边缘部署，低延迟、低功耗）
```

典型场景：
1. **VLM/VLA 蒸馏到机器人端**：云端大模型蒸馏出小 policy，机器人端低延迟执行。
2. **few-shot 边缘微调**：边缘数据少，用 teacher 的 soft label 辅助。
3. **知识继承**：新小模型直接继承大模型的"经验"。

和量化的区别：**量化是"同一个模型变轻"，蒸馏是"训练一个新的小模型"**。两者常叠加（蒸馏出小模型 + 再量化）。

---

## 7. 本模块闭环小结

```text
问题：大模型精度高但上不了边缘，小模型精度不够
      ↓
原理：soft label 编码类间相似性（hard label 没有的信息）
      ↓
实测：few-shot 下蒸馏 +3.4% 精度，参数/延迟不变；全量数据下无增益
      ↓
结论：蒸馏解决"数据少 + 模型小"的边缘泛化问题
      ↓
下一步：Stage 11 LLM 推理（Prefill/Decode、KV Cache、TTFT/TPOT）
```

要继续就说「继续」。
