# 实验 3：神经架构搜索实验 (NAS)

## 实验目标
本实验旨在让学生理解神经架构搜索（Neural Architecture Search, NAS）的基本概念和实现方法。你将实现随机搜索、进化搜索以及一个简单的精度预测器，并分析和对比不同搜索策略的效率与效果。

## 实验内容
1. **定义搜索空间**：构建 CNN 架构的搜索空间（卷积核大小、通道数、深度等）
2. **实现随机搜索**：随机采样架构并评估性能
3. **实现进化搜索**：基于种群、变异、交叉的进化算法
4. **实现精度预测器**：训练一个简单的 MLP 来预测架构精度（OFA 简化版）
5. **寻找 Pareto 最优架构**：在精度和 MACs 之间进行权衡

## 关键概念
- **搜索空间**：定义了所有可能架构的集合
- **随机搜索 (Random Search)**：最简单的 NAS 基线方法
- **进化搜索 (Evolutionary Search)**：通过变异、交叉和选择来优化架构
- **精度预测器 (Accuracy Predictor)**：使用代理模型快速估计架构性能
- **Pareto 最优**：在多目标优化中无法在不损害其他目标的情况下改进某一目标的解

## 运行方式
```bash
python starter_code.py
```

## 提交要求
1. 完成 `starter_code.py` 中的所有 TODO 标记部分
2. 填写 `report_template.md` 中的实验报告
3. 将代码和报告打包提交

## 参考资源
- Zoph & Le, "Neural Architecture Search with Reinforcement Learning" (ICLR 2017)
- Real et al., "Regularized Evolution for Image Classifier Architecture Search" (AAAI 2019)
- Cai et al., "Once-for-All: Train One Network and Specialize it for Efficient Deployment" (ICLR 2020)
- MIT 6.5940 Lecture 7-8: Neural Architecture Search
