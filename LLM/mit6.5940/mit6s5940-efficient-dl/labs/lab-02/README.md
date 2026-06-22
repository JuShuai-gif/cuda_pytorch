# 实验 2：量化实验 (Quantization) - 起始代码

## 实验目标
本实验旨在让学生理解神经网络量化的基本原理，包括线性量化、K-means 量化以及量化感知训练（QAT）。你将实现不同的量化方法，并比较不同位宽下的精度和量化误差。

## 实验内容
1. 实现 **线性量化**（linear quantization）：支持 int8/int4 量化
2. 实现 **K-means 量化**：基于聚类的非均匀量化
3. 实现 **激活值校准**（activation calibration）：确定激活值范围
4. 实现 **量化推理模块**（quantized inference module）
5. 比较不同位宽下的模型精度

## 关键概念
- **量化公式**: q = round((x - zero_point) / scale), x' = q * scale + zero_point
- **对称量化 vs 非对称量化**
- **逐层量化 vs 逐通道量化**
- **PTQ（训练后量化）vs QAT（量化感知训练）**
- **量化误差**：由舍入操作引入的信息损失

## 运行方式
```bash
python starter_code.py
```

## 提交要求
1. 完成 `starter_code.py` 中的所有 TODO 标记部分
2. 填写 `report_template.md` 中的实验报告
3. 将代码和报告打包提交

## 参考资源
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
- Nagel et al., "A White Paper on Neural Network Quantization" (2021)
- MIT 6.5940 Lecture 5: Quantization
