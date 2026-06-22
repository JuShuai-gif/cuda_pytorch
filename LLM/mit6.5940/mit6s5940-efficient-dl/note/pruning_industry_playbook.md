# 剪枝工业实战 Playbook

本文件补充 Lecture 03/04：目标不是“把权重置零”，而是让剪枝方案能在目标 runtime 上产生可验证收益。

## 1. 决策树

| 目标 | 推荐剪枝 | 原因 |
|---|---|---|
| 只减模型大小 | 非结构化 magnitude pruning | 压缩率高，但不保证 latency 下降 |
| CPU/GPU 通用加速 | 结构化通道/hidden 剪枝 | 改变 dense shape，标准 kernel 可受益 |
| NVIDIA Ampere/Hopper | 2:4 sparsity | Sparse Tensor Core 有硬件路径 |
| Transformer/LLM | head/MLP hidden/block sparse/Wanda/SparseGPT | attention/FFN 结构决定收益 |
| MCU/TinyML | channel + operator support aware | Flash/SRAM/算子支持比参数更关键 |

## 2. 工业闭环

```text
baseline -> sensitivity scan -> pruning policy -> structural rewrite -> finetune/distill -> export -> runtime benchmark -> report
```

验收时必须同时报告：accuracy delta、model size、dense FLOPs、actual sparsity、P50/P95/P99 latency、throughput、peak memory、export/runtime status。

## 3. 常见失败

- 只 mask channel，没有真正改 Conv/Linear shape，FLOPs 看似下降但 latency 不变。
- 非结构化稀疏没有 sparse kernel，CSR/COO index overhead 抵消收益。
- 残差分支、concat、BN、GroupNorm 的 channel dependency 没处理，导出失败。
- 敏感度分析只看 overall accuracy，没有看长尾场景。
- 剪枝后没有重新导出 runtime engine，仍在跑旧 engine。

## 4. 代码补齐目标

- `src/model_compression/pruning_policy.py`：保存逐层剪枝率、敏感度曲线和 policy。
- `labs/lab-01/industrial_requirements.md`：要求真实结构化 shape rewrite 和 latency report。
- `project/.../compression/pruner.py`：区分 mask pruning 与 structural pruning，并在 report 中写清楚。
