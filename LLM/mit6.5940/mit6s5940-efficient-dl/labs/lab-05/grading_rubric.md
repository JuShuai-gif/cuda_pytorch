# lab-05 评分 Rubric：部署

| 维度 | 分值 | 标准 |
|---|---:|---|
| 算法正确性 | 30 | 核心算法实现正确，边界情况可处理 |
| 指标完整性 | 20 | 同时报告质量、size、latency、memory 或 skipped reason |
| 工业解释 | 20 | 能解释为什么优化有效或无效，特别是硬件/runtime 原因 |
| 可复现性 | 15 | 命令、seed、环境、输入 shape 明确 |
| 报告质量 | 15 | 表格清晰，有结论和后续改进建议 |

## 加分项

- 输出 JSON + Markdown 双格式报告。
- 比较两个 runtime，例如 PyTorch eager vs ONNX Runtime。
- 画 Pareto frontier 或 sensitivity curve。
- 对 P95/P99 latency 做分析。
- 给出上线 reject 的明确技术理由。
