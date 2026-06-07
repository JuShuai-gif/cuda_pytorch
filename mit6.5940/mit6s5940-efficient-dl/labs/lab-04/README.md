# 实验 4：LLM 量化实验 (AWQ)

## 实验目标
本实验旨在让学生理解大语言模型（LLM）量化中的关键挑战和 AWQ（Activation-aware Weight Quantization）方法。你将实现伪量化、显著性通道识别和保护、以及自动缩放搜索，理解如何在不显著损失精度的情况下对 LLM 进行低位宽量化。

## 实验内容
1. **实现伪量化**：模拟权重的低位宽量化操作
2. **识别显著性通道**：通过激活值统计识别对精度影响大的通道
3. **保护显著性通道**：将显著性通道保持在 FP16 精度
4. **实现缩放操作**：对显著性权重进行 scale-up/scale-down 操作
5. **自动缩放搜索**：通过搜索找到最优的缩放因子

## 关键概念
- **AWQ (Activation-aware Weight Quantization)**：基于激活值统计的权重量化方法
- **显著性通道 (Salient Channels)**：对输出贡献最大的通道，量化时需特殊处理
- **缩放等效性 (Scaling Equivalence)**：对权重放大、对输入相应缩小可保持输出不变
- **Per-channel 量化 vs Per-group 量化**
- **困惑度 (Perplexity)**：衡量语言模型质量的重要指标

## 运行方式
```bash
python starter_code.py
```

## 提交要求
1. 完成 `starter_code.py` 中的所有 TODO 标记部分
2. 填写 `report_template.md` 中的实验报告
3. 将代码和报告打包提交

## 参考资源
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (MLSys 2024)
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)
- MIT 6.5940 Lecture 12-13: LLM Quantization
