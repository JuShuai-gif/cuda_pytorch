# Lab 04: Scaling Laws

## 任务目标

通过本实验，你将：

1. 理解 Kaplan 和 Chinchilla scaling laws 的核心结论与区别
2. 亲手拟合 scaling law 曲线（loss vs N, loss vs D）
3. 绘制 IsoFLOP curves 并找到 compute-optimal 配置
4. 理解 scaling law 对实践训练的指导意义

## 实验任务

### Task 1: 回顾论文 (20%)

阅读并回答以下问题：

1. Kaplan (2020) 的核心结论是什么？$L(N)$ 和 $L(D)$ 的幂律指数分别是多少？
2. Chinchilla (2022) 如何修正了 Kaplan 的结论？
3. 什么是 "compute-optimal" training？Kaplan 和 Chinchilla 给出的 optimal compute budget 分配有什么不同？
4. 为什么 Chinchilla 的训练 token 数远大于 Kaplan 的预测？

### Task 2: 拟合 Scaling Law (40%)

在 `starter.py` 中完成：

1. 实现 power-law fit 函数：$L(N) = a \cdot N^{-\alpha} + b$
2. 使用提供的实际数据点拟合参数 $a, \alpha, b$（使用最小二乘法或 `scipy.optimize.curve_fit`）
3. 绘制拟合曲线与实际数据点的对比图
4. 外推更大的模型规模，预测 loss

### Task 3: IsoFLOP Curves (40%)

1. 给定一系列 $(\log N, \log D)$ 的扫描结果，绘制 IsoFLOP loss contour
2. 在每条 IsoFLOP 曲线上找到 optimal $N$ 和 $D$
3. 绘制 optimal $N$ 与 FLOPs 的关系：$N_{opt} \propto C^a$ 和 $D_{opt} \propto C^b$
4. 对比你的拟合结果与 Chinchilla 论文的结论

## 验收标准

- [ ] 正确回答论文回顾题目
- [ ] Power-law fit 的 R² > 0.95
- [ ] 拟合出的指数与论文一致（$\alpha \approx 0.05$ for Kaplan, $\approx 0.07$ for Chinchilla）
- [ ] IsoFLOP curves 正确绘制，能识别 compute-optimal 区域
- [ ] 拟合结果保存为图像文件

## 数据

你可以在 `starter.py` 中找到预先生成的 synthetic 数据（模拟了 Kaplan 和 Chinchilla 的趋势）。

## 参考资料

- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Chinchilla 论文解读 (DeepMind 博客)](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training)

## 时间估计

约 3 小时
