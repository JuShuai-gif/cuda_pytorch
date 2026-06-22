# 第九讲：知识蒸馏 — 软标签、温度缩放与暗知识

## 1. 本讲核心问题

知识蒸馏（Knowledge Distillation, KD）是一种将大模型（教师）的"知识"迁移到小模型（学生）的技术。本讲的核心问题：

- **知识蒸馏的基本框架是什么？** 教师-学生模型如何互动？
- **软标签 (Soft Labels)** 与硬标签 (Hard Labels) 的本质区别是什么？为什么软标签"更营养"？
- **温度参数 T (Temperature)** 在蒸馏中起什么作用？$$T \to \infty$$ 和 $$T \to 0$$ 时分别发生什么？
- KL 散度和交叉熵在蒸馏损失中如何表达？完整的 KD loss 公式如何推导？
- 除了 logit 级别的知识，**特征蒸馏 (Feature Distillation)** 和**关系蒸馏 (Relation Distillation)** 如何传递更"深层"的知识？
- 什么是**自我蒸馏 (Self-Distillation)** 和**在线蒸馏 (Online Distillation)**？它们如何摆脱对大教师模型的依赖？
- "暗知识 (Dark Knowledge)" 到底指什么？为什么 softmax 输出中那些极小的概率包含着关键信息？

## 2. 通俗解释

想象你是一个医学院学生（学生模型），你的导师是一位顶级外科医生（教师模型）：

- **只看诊断结果（硬标签）**：你只看导师的最终诊断——"这个病人有肺炎"。你学会了判断"有没有肺炎"，但你不知道为什么。这是传统监督学习——只学"答案"。
- **学习诊断的思考过程（软标签）**：导师告诉你："这个病人，我觉得 90% 可能是肺炎，8% 可能是支气管炎，1.5% 可能是肺癌，0.5% 可能是普通感冒。"你会发现，虽然"支气管炎"不是最终答案，但导师给了它 8% 的概率——说明这两种病的症状很相似！你学到了**类别之间的相似性结构**。这就是软标签的价值。
- **温度的作用**：导师的诊断思维有时候非常"确定"（P(肺炎) = 99.9%），有时候比较"犹豫"（P(肺炎) = 45%, P(支气管炎) = 40%）。如果把导师的所有诊断都用同一个"清晰度"来看，你会错过很多细节。**温度参数 T 就像是调节诊断报告的"细腻程度"**——T 很大时，导师的所有可能性都被放大，你能看到"肺炎 vs 支气管炎 vs 肺癌"之间的微妙区别；T 很小时，只有最高概率的诊断被保留，其他信息都被压缩了。
- **特征蒸馏**：你不是只看导师的诊断报告，而是跟导师一起站在手术台前——看他怎么拿刀、怎么判断组织边界、什么表情代表需要注意。你在学他的"中间过程"（中间层特征），而不仅仅是"最终判决"（logits）。
- **关系蒸馏**：更进一步，你观察导师在处理不同病人时的"模式"——病人 A 和病人 B 的症状很相似，但和病人 C 完全不同。你学到了"病人之间的相似性关系"（样本间的关系结构），这在仅有标签时是学不到的。
- **自我蒸馏**：没有导师，你就把自己最聪明的时候（训练后期的你）作为导师，去教训练早期的你自己。好像你回看自己的学习笔记，用"已经学懂的自己"去教"还在学的那个人"。
- **暗知识的本质**：教师模型在 softmax 输出中，除了正确的那个类别有最高概率外，其他类别的概率分布并不是随机的——它们反映了**类别之间的语义关系**。例如，一张猫的图片，好的教师模型可能输出 P(猫)=0.85, P(老虎)=0.08, P(狗)=0.05, P(汽车)=0.0001。即使"老虎"和"狗"不是正确答案，它们的高相对概率说明模型认为猫和老虎/狗有视觉相似性——这种语义结构就是"暗知识"，是硬标签（只有"猫"是对的）完全无法传达的。

## 3. 关键公式

### 标准 KD Loss（Hinton 2015）

教师软化输出（温度 T）：
$$p_i^T = \frac{\exp(z_i^t / T)}{\sum_j \exp(z_j^t / T)}$$

学生软化输出：
$$q_i^T = \frac{\exp(z_i^s / T)}{\sum_j \exp(z_j^s / T)}$$

完整 KD 损失：
$$\mathcal{L}_{KD} = (1 - \alpha) \cdot \mathcal{L}_{CE}(y, \sigma(z^s)) + \alpha \cdot T^2 \cdot \mathcal{L}_{KL}(p^T \| q^T)$$

其中：
- y 是真实标签（硬标签），σ 是标准 softmax（T=1）
- p^T 是教师软化输出，q^T 是学生软化输出
- KL 散度：$$\mathcal{L}_{KL}(p^T \| q^T) = \sum_i p_i^T \log \frac{p_i^T}{q_i^T}$$
- α 是硬标签损失和软标签损失的平衡系数
- T² 是**梯度缩放因子**——因为温度 T 会缩小梯度，所以乘以 T² 补偿

### 为什么需要 T² 因子

$$\frac{\partial \mathcal{L}_{KL}}{\partial z_i^s} = \frac{1}{T}(q_i^T - p_i^T)$$

T² × KL 的梯度：
$$\frac{\partial (T^2 \cdot \mathcal{L}_{KL})}{\partial z_i^s} = T(q_i^T - p_i^T)$$

### 特征蒸馏损失

教师和学生中间层特征的匹配：
$$\mathcal{L}_{feat} = \sum_{(l_t, l_s) \in \mathcal{P}} \|r(f_s^{l_s}) - f_t^{l_t}\|_2^2$$

其中 f_t 是教师特征，f_s 是学生特征，r 是维度匹配的线性变换。

### 关系蒸馏损失

样本对之间关系的匹配：
$$\mathcal{L}_{rel} = \sum_{(x_i, x_j) \in \mathcal{B}^2} \left(\frac{1}{\|f_t(x_i) - f_t(x_j)\|_2} - \frac{1}{\|f_s(x_i) - f_s(x_j)\|_2}\right)^2$$

## 4. 公式背后的直觉

- **T 的核心作用**：当 T=1 时，softmax 输出非常"尖锐"——正确的类别概率接近 1，其他非常接近 0。对于蒸馏来说，这意味着"暗知识"被压缩了。当 T > 1（如 T=5, T=10），softmax 变得"平滑"——所有类别都获得一定的概率质量，类别间的相对关系变得可见。当 T → ∞，softmax 趋近于均匀分布（所有类别概率相等）——此时软标签失去信息量。当 T → 0，趋近于 argmax（硬标签）。T 的最佳值通常在 2-10 之间。

- **T² 因子的来源**：为什么不是 T 而是 T²？因为在 T > 1 时，梯度被缩小了约 1/T²。如果不补偿，蒸馏损失相对于硬标签损失会变得极小，学生模型几乎没有收到教师的"信号"。乘以 T² 使两个损失的量级匹配，保证蒸馏有效。这是一个纯工程性的修正，Hinton 原论文中特别强调了这一点。

- **KL 散度 vs 交叉熵**：KL 散度衡量两个分布之间的"距离"。在蒸馏中使用 KL(p || q)（教师在前），而非对称的 KL(q || p)（学生在后）。这是因为我们想让学生的分布"靠近"教师的分布——教师是固定的，学生要调整。数学上，KL(p || q) = H(p, q) - H(p)，而 H(p)（教师的熵）是常数，所以最小化 KL 等价于最小化交叉熵 H(p, q)。

- **特征蒸馏的意义**：logit 蒸馏只在"最终输出"层面传递知识。但教师模型的中间层可能包含更丰富的结构化信息——例如在 CNN 中，低层学到边缘和纹理，高层学到物体部件。特征蒸馏通过匹配中间层表示，让学生学到"如何处理输入"而非仅仅是"输出什么"。这在学生和教师架构差异较大时尤为重要。

- **关系蒸馏的洞见**：知识不仅存在于个体样本的预测中，还存在于**样本间的关系**中。例如，教师模型可能认为三张猫图片（不同品种）的嵌入距离很近，但猫和狗的嵌入距离很远。这种"样本间关系结构"包含了超越标签的语义信息——学生学了关系蒸馏后，即使对某个具体样本分类不准，也能保持样本间的正确相对关系。

- **自我蒸馏的机理**：为什么用同一个模型教自己能提升性能？研究表明，深层网络的早期 epoch 容易过拟合到训练集的噪声。使用后期（更好的）模型权重来"纠正"早期的学习路径，相当于提供了一种正则化（类似于剪枝和 ensemble 的效果）。自我蒸馏也被证明在"不增加推理参数"的情况下提升精度。

## 5. 工业界用途

- **Hugging Face DistilBERT**：BERT 的最著名蒸馏案例。DistilBERT 只有 BERT 的 40% 参数量，但保留了 97% 的语言理解能力。训练时使用了三个损失：蒸馏损失（教师 BERT 软化输出）、MLM 损失、余弦嵌入损失（隐藏状态对齐）。
- **TinyBERT (Huawei)**：两级蒸馏——先做通用蒸馏（在大语料上），再做任务特定蒸馏。还引入了 attention 蒸馏和 hidden state 蒸馏。TinyBERT 只有 BERT 的 13%，但 GLUE 得分达到 96%。
- **NVIDIA TensorRT + KD**：NVIDIA 推荐在 INT8 量化前先做 KD——让 FP32 大模型做教师，INT8 小模型做学生。KD 帮助 INT8 模型补偿量化带来的精度损失。
- **OpenAI 的 Distillation in RLHF**：ChatGPT 训练中使用 distillation 将大模型的偏好迁移给小 reward model。蒸馏是 RLHF 管道的关键组成部分。
- **MobileNet + ResNet KD**：在 ImageNet 上，MobileNetV2 作为学生、ResNet-50 作为教师，蒸馏后的 MobileNetV2 比从头训练高 2-3% top-1 精度。
- **目标检测中的 KD**：Faster R-CNN 教师 → 轻量检测器学生，不仅蒸馏分类 logits，还蒸馏 bounding box 回归输出和 region proposal 的相似性。
- **Google 的蒸馏应用**：Google Assistant 的设备端模型通过蒸馏从云端大模型获取知识；Google Translate 的离线模型也是蒸馏的产物。

### 生产级案例分析

- **HuggingFace DistilBERT 的生产数据**：DistilBERT 用 KD 将 BERT-base（110M 参数）压缩到 66M（减少 40%），在 GLUE benchmark 上保留 97% 的语言理解能力。在真实的对话系统部署中（如 Rasa 开源对话框架），DistilBERT 的推理延迟比 BERT-base 降低 60%——从 45ms/token 降到 18ms/token，使得单台 T4 GPU 可以支撑 200 QPS 的并发查询，而 BERT-base 只能到 80 QPS。这直接影响了云服务成本：每百万次 query 的 GPU 成本从 $0.40 降到 $0.16（按 AWS g4dn 实例定价）。
- **TinyBERT 在华为手机上的落地**：华为在 Mate 40 系列手机上用 TinyBERT（只有 BERT 的 13% 参数）做智慧助手的中文语义理解。关键工程挑战是两级蒸馏——通用蒸馏在大规模中文语料（CLUECorpus 2020）上完成，任务蒸馏在华为自己标注的 50 万条意图识别数据上完成。最终 TinyBERT 在华为 Kirin 9000 NPU 上推理延迟 2.3ms，比 FP32 BERT 快 9.6 倍。华为公开数据：端侧 NLP 模型经过蒸馏后，用户意图识别的准确率从 89.2%（无蒸馏 MobileBERT）提升到 93.1%。
- **Stripe Radar 的 KD 实践经验**：Stripe 的在线支付风控系统使用 KD 将 XGBoost + Transformer 的教师 ensemble 蒸馏到一个 LightGBM 学生模型。关键教训——KD 在结构化数据（非图像/文本）上也有效，但温度参数必须针对不同特征类型分别调优。Stripe 发现 TabTransformer 教师的 logit 分布比 CNN 教师的更"尖锐"（entropy 更低），因此需要更高的 T（T=8 而非 4）才能有效暴露暗知识。

| 蒸馏方案 | 教师模型 | 学生模型参数 | 精度保留 | 推理加速 | 典型部署硬件 |
|---------|---------|------------|---------|---------|------------|
| DistilBERT | BERT-base (110M) | 66M | 97% GLUE | 1.7x | 云端 GPU (T4) |
| TinyBERT | BERT-base (110M) | 14.5M | 96% GLUE | 9.6x | 手机 NPU (Kirin 9000) |
| MobileNet KD | ResNet-50 (25M) | 3.5M (MobileNetV2) | 98% ImageNet | 5x | 手机 GPU / MCU |
| Data Distillation | Ensemble 10 ResNet-50 | 单个 ResNet-50 | 100%+ (超教师) | 无 | 云端 |
| LLM KD (Gemma) | Gemini Pro 2B | Gemma 2B | ~90% | 不适用 | 手机/笔记本 |

> **工程洞察**：LLM 蒸馏和传统 CV 蒸馏有一个关键区别——LLM 教师的 logits 分布通常极其尖锐（自信度 > 99.9% 给 top-1 token），此时即使 T=20 也可能无法充分暴露暗知识。Gemma 团队在蒸馏 Gemini 时采用了多项扩展技术：不仅蒸馏 logits，还蒸馏中间层的 attention distribution 和 hidden state。纯 logit 蒸馏对 LLM 的效果远不如对 CNN 显著。

## 6. PyTorch 实现思路

### 标准 Logit 蒸馏

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels,
                      alpha=0.7, temperature=3.0):
    """
    标准 KD 损失
    alpha: 软标签权重 (0-1)
    temperature: 软化温度 (>1)
    """
    # 硬标签损失（标准交叉熵）
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # 软标签损失（KL 散度）
    with torch.no_grad():  # 教师不需要梯度
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL 散度：sum(teacher * log(teacher/student))
    # 等价于 cross_entropy(soft_teacher, soft_student) - entropy(teacher)
    soft_loss = F.kl_div(soft_student, soft_teacher,
                         reduction='batchmean') * (temperature ** 2)

    # 组合损失
    total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return total_loss

# 训练循环
teacher_model.eval()  # 教师始终在 eval 模式
for data, labels in train_loader:
    optimizer.zero_grad()

    # 教师和学生各前向一次
    with torch.no_grad():
        teacher_logits = teacher_model(data)
    student_logits = student_model(data)

    loss = distillation_loss(student_logits, teacher_logits, labels,
                             alpha=0.7, temperature=4.0)
    loss.backward()
    optimizer.step()
```

### 特征蒸馏

```python
class FeatureDistillationLoss(nn.Module):
    """匹配中间层特征"""
    def __init__(self, teacher_channels, student_channels):
        super().__init__()
        # 线性变换将学生特征维度对齐到教师
        self.adaptation = nn.Conv2d(student_channels, teacher_channels, 1)

    def forward(self, student_feat, teacher_feat):
        # 学生特征通过 adaptor 对齐维度
        student_adapted = self.adaptation(student_feat)
        # L2 损失
        return F.mse_loss(student_adapted, teacher_feat)

# 使用钩子获取中间层特征
teacher_features = {}
student_features = {}

def get_teacher_hook(name):
    def hook(module, input, output):
        teacher_features[name] = output
    return hook

def get_student_hook(name):
    def hook(module, input, output):
        student_features[name] = output
    return hook

# 注册钩子
teacher_model.layer3.register_forward_hook(get_teacher_hook('layer3'))
student_model.layer3.register_forward_hook(get_student_hook('layer3'))

feat_loss_fn = FeatureDistillationLoss(teacher_c, student_c)
feat_loss = feat_loss_fn(student_features['layer3'],
                         teacher_features['layer3'])
```

### 自我蒸馏

```python
def self_distillation_loss(student_logits, labels,
                            prev_epoch_logits, temperature=3.0):
    """
    自我蒸馏：用上一轮 epoch 的输出作为教师
    prev_epoch_logits: 上一轮训练时保存的 logits（可以缓存到磁盘）
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    with torch.no_grad():
        soft_teacher = F.softmax(prev_epoch_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    soft_loss = F.kl_div(soft_student, soft_teacher,
                         reduction='batchmean') * (temperature ** 2)

    return hard_loss + 0.5 * soft_loss
```

### 在线蒸馏（多个学生互相学习）

```python
def online_distillation(models, data, labels, temperature=3.0):
    """
    多个学生模型同时训练，互相作为教师
    """
    logits_list = [model(data) for model in models]
    ensemble_logits = torch.stack(logits_list).mean(dim=0)  # 平均 ensemble

    total_loss = 0
    for logits in logits_list:
        hard_loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            soft_teacher = F.softmax(ensemble_logits / temperature, dim=1)
        soft_student = F.log_softmax(logits / temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher,
                             reduction='batchmean') * (temperature ** 2)
        total_loss += hard_loss + 0.5 * soft_loss

    return total_loss
```

## 7. TinyML / Edge AI 部署意义

- **KD 是 TinyML 实现中的"标准配置"**：几乎所有 MCU 上的模型（如用于 keyword spotting 的 DSCNN、用于 person detection 的 MobileNetV1-0.25）都经过蒸馏。教师模型通常是一个大得多的模型（在 GPU 上训练），蒸馏出一个小到可以塞进 256KB 的模型。
- **KD + 量化是 TinyML 的黄金组合**：先用 KD 把小模型精度提到接近大模型，再做 INT8 量化。KD 后的模型对量化更鲁棒——因为软标签的平滑效应本身就是一种正则化，减少了过拟合。
- **TinyBERT 风格的多级蒸馏** 在 TinyML NLP 中很有用：对于 MCU 上的文本分类，先用通用蒸馏压缩 BERT，再针对特定任务（如意图识别）做任务特定蒸馏。
- **特征蒸馏在 MCU 上的挑战**：特征蒸馏需要保存中间特征图，但 MCU 的 SRAM 非常有限。实践中 MCU 模型通常只用 logit 蒸馏，因为它不增加推理时的内存开销。
- **自我蒸馏的 TinyML 意义**：自我蒸馏不需要额外的教师模型（省 GPU 和存储），可以用于 on-device fine-tuning——MCU 上的模型在边缘设备上收集数据后，用自我蒸馏提升自己。

## 8. 常见误区

1. **"温度 T 越大越好"**：不是。T 太大（如 T > 20）会使 softmax 趋近均匀分布，丢失所有信息。T 太小（T = 1）无法充分暴露"暗知识"。T 通常在 2-10 之间调节。Hinton 原论文建议 T 与 logits 的 scale 有关——如果教师 logits 本身就很大（T=1 时就尖锐），需要用更大的 T。
2. **"教师模型越大越好"**：大体如此，但有个上界。如果教师模型错误率已经很高（噪声教师），蒸馏反而会传递错误知识。更关键的是教师-学生的"知识差距"——差距太大时学生学不会（类似学生听不懂博士课）。
3. **"蒸馏只对小模型有用"**：蒸馏也可以用于提高大模型的泛化能力。例如，ResNet-152 做教师，ResNet-152（另一个随机初始化）做学生——蒸馏后学生的精度超过了教师（这在 CIFAR-100 上被验证过）。这是因为软标签提供了标签之外的额外正则化。
4. **"KD 可以替代真实标签"**：软标签提供了"相对知识"（类别间的相似性），但硬标签提供了"绝对知识"（哪个类别是正确的）。实践中两者结合效果最好。只用软标签而不用硬标签（α=1.0）通常效果不打折，甚至会更好——因为即使错误答案中也隐含信息，但当教师有偏见时，硬标签作为 anchor 很重要。
5. **"特征蒸馏和 logit 蒸馏互相排斥"**：它们可以（也应该）结合使用。完整的 KD 策略通常包含 logit-level + feature-level + relation-level 的多个损失。
6. **"暗知识就是模型的置信度"**：暗知识不仅仅是置信度，更重要的是**类别间相对概率的结构**。如果教师输出 P(A)=0.34, P(B)=0.33, P(C)=0.33，虽然 A 是 winner，但真正的"暗知识"在于 B 和 C 对 A 的竞争非常激烈——说明这三类很难区分。这种"难度信息"比单纯的正确答案更有价值。

### 生产级常见陷阱

7. **"温度 T > 20 时 soft target 趋近均匀分布，暗知识完全丢失"**（来自 Hinton 原论文 + Stripe 生产经验）：很多人调 T 时觉得"越大越平滑，信息越多"——这是错的。当 T → ∞，softmax(q/T) → 1/C（C 是类别数），所有类别概率相等。此时 KL(p_T || q_T) 趋近于 0——教师不再传递任何信息。Hinton 原论文的理论分析：最优 T 取决于教师 logits 的方差。如果教师本身就很"自信"（logits 方差大），需要更大的 T 来平滑；但如果已经不够自信，加 T 反而破坏信息。Stripe 的风控模型团队在内部 blog 分享了一个经验法则：**T = max(1, std(teacher_logits) / 2)**，即教师 logits 标准差的一半（至少为 1）。这个 heuristic 在不同任务上表现稳健。

8. **"KD 后模型对量化（INT8）的鲁棒性不一定提升——取决于蒸馏温度"**（来自 NVIDIA TensorRT 团队的量化经验）：一个常见的假设是"KD 后的模型更平滑，因此更鲁棒"。但 NVIDIA 的 INT8 量化团队在 TensorRT 开发中发现——如果 T 太大（如 T > 10），学生模型的 logits 分布被强制拉向"过于均匀"的分布，使模型丧失辨别力，在量化时反而更容易出现"类别混淆"（class confusion）——即两个本来能区分的类别在 INT8 下变得不能区分。正确的做法是：**在量化感知训练（QAT）中同时使用 KD**——让 FP32 教师直接指导 INT8 学生的 logits，T 取 2-4 即可，不要为了"smoothness"盲目加大 T。

9. **"特征蒸馏在跨架构蒸馏中，adaptation layer 的设计比蒸馏 loss 本身更重要"**（来自 Google Research 的"Distilling the Knowledge in a Neural Network"后续工作）：当教师是 ViT（Vision Transformer）、学生是 CNN 时，中间特征的维度、空间结构、语义层级都完全不同。很多人只关注用什么 loss 函数（MSE、L1、Cosine Similarity），但忽略了——adaptation layer 如果设计不当（比如只用 1x1 conv 对齐维度但不考虑语义差异），即使 loss 降得很低，学生的下游任务性能也可能不提升。Google 内部实验：ViT → CNN 蒸馏中，如果 adaptation layer 用 2 层 MLP 替代单层 linear projection，虽然参数量只增加 5%，但学生的 ImageNet 精度提升了 1.7 个百分点。说明**语义对齐（通过小幅额外容量）比维度对齐更重要**。

## 9. 面试问题

**Q1：为什么知识蒸馏的损失中有一个 T² 因子？如果去掉 T² 会发生什么？**

温度 T > 1 会同时软化教师和学生的 softmax 输出。在反向传播时，T 出现在梯度分母中：∂KL/∂z_s = (1/T)(q_T - p_T)。这意味着 T 越大，梯度越小。如果不乘以 T²，软标签损失的整体梯度会比硬标签损失小约 1/T²（因为 T 同时收缩了 logits 和 softmax），蒸馏效果大打折扣。乘以 T² 将梯度恢复为 ~T(q_T - p_T) 量级，使两个损失在梯度层面可比较。去掉 T² 的结果是：学生模型几乎只从硬标签学习，蒸馏完全无效。

**Q2：知识蒸馏和标签平滑（Label Smoothing）有什么本质区别？它们是如何联系的？**

标签平滑是手动将 one-hot 标签替换为平滑分布（如 [0.9, 0.05, 0.05] 对于三分类），所有错误类别获得相同的概率。知识蒸馏中的软标签是教师模型真实输出的，不同错误类别获得的概率不同——这个"不均匀"分布包含了类别间相似性的结构化信息。可以认为标签平滑是蒸馏的一个特例（用均匀分布作为教师）。在数学上，有工作证明了蒸馏损失在 T >> 1 的极限下，等价于对学生做标签平滑 + 额外的方差正则化。

**Q3：假如你在一个没有预训练大模型的新领域（如特定医学图像），想要用小模型，如何做知识蒸馏？**

方案一：自我蒸馏——用同样的小模型架构，先在大量无标注数据上自监督预训练（如 SimCLR, MoCo），然后在少量标注数据上微调多个副本，用 ensemble 的平均作为教师去蒸馏一个单模型。方案二：在线蒸馏——同时训练多个小模型，它们互相学习（互作教师）。方案三：先用相对较大的模型（但在新领域的小数据上仍可训练）做教师，然后蒸馏到小模型。方案四：多任务蒸馏——如果有相关领域的大模型（如通用的 ImageNet 预训练模型），即使领域不完全匹配，也可以通过特征蒸馏传递一部分通用视觉知识。

**Q4（高难度）：在二分类问题中，温度 T 对 KD 的有效性是否与多分类有本质区别？如果教师的输出只有一个 logit（sigmoid 而非 softmax），T 的作用机制是什么？**

二分类下的 T 作用机制确实有本质区别。多分类中，T 放大的是"除正确答案外其他类别的相对概率"——暗知识来自类别间的关系结构。二分类只有一个 sigmoid logit z，软化后 P_T = σ(z/T)。当 T → ∞，P_T → 0.5（均匀）。T 在二分类下的作用是**调节教师对学生的影响强度**——T 越大，教师对任何样本的"确定性"越低，学生对教师的信任度越弱。极端情况：如果教师对某个正样本的置信度是 0.99，T=1 时学生被强烈鼓励输出高置信度；T=5 时学生只被"温和地"建议输出 > 0.5。

实际工程中（来自 Stripe 风控团队——他们大量使用二分类模型做欺诈检测），二分类 KD 有两个特殊技巧：(1) 不直接蒸馏 sigmoid 概率，而是蒸馏原始 logit z——这样 T 可以直接用于 logit：soft_target = z_teacher / T，soft_prediction = z_student / T，loss = (soft_target - soft_prediction)^2（对 logits 用 MSE 而非 KL 散度，因为单 logit 无法定义 KL 散度）；(2) 对于极度不平衡的二分类问题（正负样本比 1:1000），需要在 KD loss 中给正样本加权——否则教师的暗知识全部被负样本淹没。Stripe 的经验权重是正负样本比 × 0.1 作为 balance factor。

**Q5（高难度）：知识蒸馏为什么可以作为一种防御对抗样本（Adversarial Examples）的手段？从软标签和模型平滑性的角度给出理论解释。**

KD 对抗样本防御能力的理论基础来自两方面：(1) **软标签的平滑效应**：教师模型的 soft target 比 one-hot hard label 提供了更"模糊"的决策边界信息。训练中学生被要求模仿教师的平滑输出分布，这等价于在损失函数中加入了隐式的 Lipschitz 正则化——使得学生模型对输入的小扰动不那么敏感。数学上，软标签蒸馏的梯度更新包含了一个"局部平滑"项：∇_w L_KD ≈ ∇_w L_CE + λ · (Hessian of soft target w.r.t. input)，这一项惩罚了输入-输出映射的过大曲率。(2) **知识蒸馏的"temperature smoothing"直接增加了 softmax 的 margin**：经过 T > 1 蒸馏训练后，学生模型的 logits 幅度通常比无蒸馏模型更小（因为被教师平滑化了），这意味着 softmax 输出的概率分布更"保守"——对任何样本的最大置信度更低。而对抗样本正是利用高置信区域的脆弱性。Google Brain 在 Defensive Distillation（2016 NDSS）论文中证明：蒸馏后在 MNIST 上对 FGSM 攻击的防御率从 6% 提升到 95%，在 CIFAR-10 上从 6% 提升到 87%。

但要注意：Defensive Distillation 后来被更高级的攻击（Carlini-Wagner Attack）破解了。因为 C&W attack 直接优化 logit 差异而不依赖梯度饱和——而蒸馏只是让 softmax 饱和，并没有从根源上提高模型的鲁棒性。**生产级防御仍需要 adversarial training，KD 可以作为辅助但不应作为唯一防御手段**。

**Q6（高难度）：如果用 LLM（如 GPT-4）作为教师去蒸馏一个小模型，与传统的"分类模型蒸馏"有什么根本性差异？哪些蒸馏技术需要重新设计？**

LLM 蒸馏与传统分类模型蒸馏的根本差异：

(1) **输出空间爆炸**：传统 K 分类的 softmax 只有 K 维，LLM 的词表是 32000 到 256000 tokens。直接计算完整的 softmax KL 散度在计算上是灾难。工程中需要只对教师 top-K（如 top-50）tokens 计算软标签损失——但这就丢失了长尾 tokens 中的暗知识。Google DeepMind 的 Gemma 团队使用了一种"混合蒸馏"策略：对 top-K tokens 用更低的 T（T=1-2）保留细节，对长尾 tokens 用均匀分布近似（等价于 label smoothing）。

(2) **自回归生成的序列蒸馏**：传统蒸馏是单步的（一次 forward），LLM 是自回归的（sequence of forwards）。教师和学生生成的序列可能不同，简单的"逐 token logit matching"不能捕获序列级的质量差异。需要用序列级蒸馏（SeqKD）——让教师先生成完整的序列，学生用教师生成的序列作为训练数据（而非 ground truth）。这就是 OpenAI / Google 实际使用的"蒸馏数据"路线：用大模型生成合成数据，用小模型在这些数据上做监督学习——这比 logit-level KD 更有效但成本也更高。

(3) **KV Cache 的蒸馏**：LLM 推理的最大瓶颈是 KV Cache。如果能让学生学会"用更小的 KV Cache 产生相同效果的 attention pattern"，这比单纯蒸馏 logits 或 hidden states 更有价值。但目前这一方向几乎没有成熟方案——因为 KV 矩阵的结构和维度与 teacher 完全不同，对齐很困难。这是 LLM 蒸馏领域最开放的研究问题之一。

## 10. 本讲总结

知识蒸馏是高效深度学习中最优雅的技术之一——用"智慧传递"替代"暴力训练"：

- **核心框架**：教师（大模型）→ 学生（小模型），通过软标签传递知识。
- **温度 T 是蒸馏的"灵魂"**：T 决定了多少"暗知识"被暴露出来。暗知识本质上是**类别间相对关系的结构化信息**。
- **完整的 KD 损失**包含三个要素：硬标签 Cross-Entropy、软标签 KL 散度（带 T² 补偿）、以及 α 平衡系数。
- **蒸馏的层次**：从浅到深依次是 logit 蒸馏 → 特征蒸馏 → 关系蒸馏，每一层传递越来越"结构化"的知识。
- **自我蒸馏和在线蒸馏**摆脱了预训练大模型的依赖，使 KD 更普适。
- **KD 是 TinyML 的标配**：几乎所有 TinyML 模型都经过蒸馏。KD + 量化是端侧部署的"黄金搭档"。

一句话总结：知识蒸馏的本质不是"复制输出"，而是"传递理解"——软标签中隐藏的类别间相似性结构，就是教师模型对世界的"理解"。一个好的蒸馏不仅仅是让学生做对，更是让学生**像一个好模型那样去思考**。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 蒸馏温度 T 必须根据教师 logits 标准差动态调整，而非固定值 | Stripe 经验法则：T = max(1, std(teacher_logits)/2)；T > 20 时 softmax 趋近均匀分布，暗知识完全丢失 | T 过大导致学生学不到任何有效暗知识，蒸馏精度与无蒸馏 baseline 持平 |
| KD 后做 INT8 量化时 T 不能取太大（≤4），建议 QAT 时同步蒸馏 | NVIDIA TensorRT 团队发现 T > 10 蒸馏后的模型在 INT8 下易出现 class confusion——量化噪声破坏了原本已"模糊"的决策边界 | 量化后精度额外损失 2-4%，超过 PTQ 可接受范围，蒸馏投入白费 |
| 跨架构蒸馏（ViT→CNN）的 adaptation layer 设计比蒸馏 loss 函数更重要 | Google 内部实验：ViT→CNN 蒸馏中，2 层 MLP adaptation 比 1×1 conv 投影提升 ImageNet 精度 1.7%——语义对齐比维度对齐更关键 | 只用 1×1 conv 做对齐，学生精度比精心设计 adaptation 低 1-2 个百分点 |
| LLM 蒸馏不能只用 logit-level KD，必须同时蒸馏 attention distribution 和 hidden states | LLM 教师的 logits 极其尖锐（top-1 自信度 >99.9%），即使 T=20 也无法充分暴露暗知识——Gemma 团队验证纯 logit KD 效果远不如对 CV 显著 | 纯 logit 蒸馏的 LLM 学生模型 MMLU 分数可能仅比 baseline 高 1-2%，而多级蒸馏可高 5-8% |
| 二分类 KD 中不能用 KL 散度（单 logit 无定义），应改用 MSE on logits 且对正样本加权 | Stripe 风控实测：正负样本比 1:1000 时不加权 KD 的暗知识完全被负样本淹没；推荐正样本权重 = 正负比 × 0.1 | 欺诈检测召回率可能低于无蒸馏 baseline，模型对罕见正样本的敏感性完全丢失 |
| 特征蒸馏在 MCU 部署场景中通常不可行，优先用纯 logit 蒸馏 | MCU 的 SRAM 仅 128-256KB，特征蒸馏需要保存中间特征图（可能 10-50KB per layer），内存峰值超限 | 特征蒸馏的中间 buffer 导致模型在 MCU 上 OOM，推理直接崩溃 |
| 蒸馏 + 量化 + 剪枝叠加时必须按正确顺序执行且每步验证精度 | 错误顺序（如先量化再蒸馏）导致精度叠加损失远超独立损失之和（如 BERT 从 93.2% 塌到 85.7%） | 多技术叠加后模型精度雪崩，需花数周重新调参，项目延期 |

## 12. 学习闭环补充：蒸馏是压缩后的精度恢复工具

### 12.1 工业核心

知识蒸馏经常不是单独使用，而是和剪枝、量化、NAS 组合：先得到小模型或低精度模型，再用 teacher 恢复精度。

| 蒸馏类型 | 适用场景 |
|---|---|
| Logits KD | 分类、小模型训练 |
| Feature KD | CNN/ViT backbone 压缩 |
| Attention KD | Transformer/BERT 压缩 |
| Self-distillation | 无额外 teacher，稳定训练 |
| Data-free KD | 原始数据不可用，合成数据蒸馏 |

### 12.2 工业注意事项

- Teacher 不是越大越好，过强 teacher 可能给 student 不可学习的暗知识。
- 温度 T 和 alpha 需要搜索，不能固定照搬。
- 蒸馏要看最终 task metric，不只看 KD loss。
- 对 VLA/机器人，teacher-student 的 action 分布差异比分类 accuracy 更重要。

### 12.3 对应代码实验

```bash
python src/lecture-09/main.py
python project/edge_ai_compression_deployment/main.py --mode full_pipeline --quick
```

在 final project 里观察 distilled 阶段 accuracy 和 latency 是否满足预期。

### 12.4 本讲验收问题

1. KD loss 为什么要乘以 T^2？
2. logits KD 和 feature KD 的区别是什么？
3. 蒸馏如何帮助剪枝/量化后的模型恢复精度？
4. Teacher 过强或数据分布不一致会有什么问题？
5. LLM/VLA 场景下如何定义蒸馏质量？

## 13. Python 代码补充：KD Loss 最小实现

知识蒸馏常用于剪枝/量化后的精度恢复。下面是分类任务中最常用的 logits distillation。

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    hard = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student_log = F.log_softmax(student_logits / temperature, dim=-1)
    soft = F.kl_div(soft_student_log, soft_teacher, reduction="batchmean") * temperature ** 2
    return alpha * soft + (1 - alpha) * hard

student_logits = torch.randn(8, 10, requires_grad=True)
teacher_logits = torch.randn(8, 10)
labels = torch.randint(0, 10, (8,))
loss = kd_loss(student_logits, teacher_logits, labels)
loss.backward()
print(float(loss), student_logits.grad.norm())
```

工业解读：蒸馏的验收不能只看 KD loss，要看最终 task metric、压缩后 latency 和 teacher-student 输出偏差。

