# 实验 1：剪枝实验 (Pruning)

## 实验目标
本实验旨在让学生理解神经网络剪枝的基本概念和实现方法。你将实现幅度剪枝（magnitude pruning）、敏感性扫描（sensitivity scan）以及剪枝后的微调（fine-tuning），并分析不同剪枝比例对模型精度、参数量和推理速度的影响。

## 实验内容
1. 加载预训练的 VGG11 模型并在 CIFAR-10 上评估基线性能
2. 实现 **幅度剪枝**（magnitude pruning）：按权重大小将最小的权重置零
3. 实现 **敏感性扫描**（sensitivity scan）：逐层分析每层的剪枝敏感性
4. 实现 **微调循环**（fine-tuning loop）：恢复剪枝后的精度损失
5. 测量并对比剪枝前后的精度、参数量、稀疏度和推理延迟

## 关键概念
- **非结构化剪枝 vs 结构化剪枝**：本实验实现非结构化剪枝（细粒度），将单个权重置零
- **稀疏度（Sparsity）**：被剪枝（置零）的权重占总权重的比例
- **敏感性扫描**：不同层对剪枝的敏感度不同，第一层和最后一层通常更敏感
- **微调的重要性**：剪枝后必须进行微调以恢复精度

## 运行方式
```bash
python starter_code.py
```

## 提交要求
1. 完成 `starter_code.py` 中的所有 TODO 标记部分
2. 填写 `report_template.md` 中的实验报告
3. 将代码和报告打包提交

## 参考资源
- Han et al., "Learning both Weights and Connections for Efficient Neural Networks" (NeurIPS 2015)
- MIT 6.5940 Lecture 3: Pruning and Sparsity
