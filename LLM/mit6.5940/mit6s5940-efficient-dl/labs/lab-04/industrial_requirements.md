# lab-04 工业验收要求：LLM/AWQ

本实验不以“函数写完”为最终目标，而以能否形成工程闭环为验收标准。

## 必须产出

1. Baseline 指标：模型大小、参数量、核心质量指标和 latency。
2. 优化后指标：同一硬件、同一输入设置下的对比结果。
3. Delta 分析：精度/误差变化、速度变化、内存变化。
4. 失败原因：如果某一步跳过或未加速，必须写明原因。
5. 上线判断：accept / reject / needs more data。

## 本实验重点指标

`weight-only quant、activation outlier、perplexity/action error、tokens/s`

## 报告最低要求

| 项 | 要求 |
|---|---|
| 可复现命令 | 写清运行命令、seed、device |
| 输入设置 | batch size、shape、calibration samples |
| 延迟 | 至少 report mean 和 P95；有条件 report P99 |
| 质量 | accuracy/MSE/perplexity/action error 中至少一个 |
| 工业判断 | 说明是否值得部署，不能只说“变小了” |

## 常见不合格情况

- 只报告参数量，不报告 latency。
- 只报告 FLOPs，不做 runtime benchmark。
- 压缩后精度下降但没有解释或补救。
- 使用 synthetic data 却声称真实业务可上线。
- 跳过 ONNX/TensorRT/OpenVINO/TFLite 等部署验证但没有说明原因。
