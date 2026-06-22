# Lab 05: Data Pipeline 与 Alignment

## 任务目标

通过本实验，你将：

1. 理解 LLM 训练的数据 pipeline 全流程
2. 实现一个简单的数据质量过滤流水线
3. 理解 DPO 的核心公式并与 RLHF 进行对比
4. 实现 DPO loss 函数

## 实验任务

### Task 1: 数据 Pipeline 设计 (30%)

在 `starter.py` 中实现一个数据预处理流水线：

1. **Text cleaning**：去除 HTML tags、URLs、特殊字符
2. **Quality filtering**：基于以下规则过滤低质量文本
   - 最小/最大长度过滤
   - 重复率检测（n-gram duplication ratio）
   - 语言检测（简单的启发式方法）
3. **De-duplication**：实现 MinHash-based 近似去重
4. **Data mixing**：按给定比例混合多个数据源

### Task 2: DPO Loss 实现 (40%)

在 `starter.py` 中实现 Direct Preference Optimization (DPO) loss：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

其中：
- $\pi_\theta$ 是当前策略模型
- $\pi_{\text{ref}}$ 是 reference 模型（frozen）
- $y_w$ 是 preferred response，$y_l$ 是 dispreferred response
- $\beta$ 是 temperature 参数

### Task 3: RLHF vs DPO 对比 (30%)

1. 解释 RLHF 的三步流程（SFT → Reward Model → PPO）
2. 解释 DPO 如何通过 reparameterization 简化 RLHF
3. 分析 DPO 的优缺点（何时用 RLHF，何时用 DPO）
4. 讨论 GRPO (DeepSeek) 与 DPO 的区别

## 验收标准

- [ ] 数据 pipeline 能正确处理包含 HTML、重复文本的输入
- [ ] DPO loss 实现与参考值误差 < 1e-6
- [ ] 正向样本 (chosen) 的 log probability 高于负向样本 (rejected)
- [ ] 回答 RLHF vs DPO 的对比题
- [ ] 代码有清晰的注释

## 参考资料

- [DPO 论文 (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [RLHF 论文 (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [GRPO (DeepSeekMath, 2024)](https://arxiv.org/abs/2402.03300)
- [Data pipeline for LLMs (CommonCrawl 处理)](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

## 时间估计

约 3 小时
