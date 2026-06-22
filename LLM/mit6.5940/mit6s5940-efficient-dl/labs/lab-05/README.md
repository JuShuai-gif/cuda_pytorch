# 实验 5：LLM 边缘部署实验

## 实验目标
本实验旨在让学生理解将大语言模型（LLM）部署到资源受限的边缘设备（如手机、IoT 设备）时的核心挑战和解决方法。你将模拟 ONNX 导出流程、实现 INT8 量化、进行 CPU 推理基准测试，并生成部署报告。

## 实验内容
1. **模拟 ONNX 导出**：将微型 Transformer 以可追踪的方式"导出"
2. **实现 INT8 量化**：对导出模型进行 INT8 权重量化
3. **CPU 推理基准测试**：模拟边缘设备上的推理性能
4. **FP32 vs INT8 对比**：比较延迟、模型大小和精度
5. **生成部署报告**：总结部署指标和建议

## 关键概念
- **ONNX (Open Neural Network Exchange)**：跨框架的模型交换格式
- **边缘部署 (Edge Deployment)**：在资源受限设备上运行模型
- **模型压缩率**：量化后模型大小与原始大小的比率
- **推理延迟**：单个推理请求的端到端延迟
- **吞吐量**：单位时间内能处理的请求数

## 运行方式
```bash
python starter_code.py
```

## 提交要求
1. 完成 `starter_code.py` 中的所有 TODO 标记部分
2. 填写 `report_template.md` 中的实验报告
3. 将代码和报告打包提交

## 参考资源
- ONNX Runtime 官方文档: https://onnxruntime.ai/
- NVIDIA TensorRT: https://developer.nvidia.com/tensorrt
- MIT 6.5940 Lecture 14-15: Efficient Deployment
