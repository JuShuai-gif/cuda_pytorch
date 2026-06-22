# NAS 与硬件感知搜索 Playbook

## 核心思想

NAS 的工业价值不是“自动找最高精度网络”，而是在目标硬件和约束下找到 Pareto frontier。

```text
search space -> supernet/weight sharing -> latency lookup table -> search -> retrain -> export -> benchmark
```

## 必须覆盖

- Search space：depth、width、kernel、expand ratio、resolution、attention heads。
- Search strategy：random、evolution、RL、differentiable、Once-for-All。
- Objective：accuracy、latency、memory、energy、operator support。
- Hardware feedback：真实 latency table，不用 FLOPs 替代。

## 工业坑

- FLOPs 更低的模型可能在目标 NPU 上更慢，因为 op 不支持或 layout transform 多。
- 搜索时用 batch=64 latency，上线 batch=1，Pareto frontier 会变。
- 搜索结构必须可导出，不能包含 runtime 不支持的算子。

## 验收

- 至少输出 5 个 Pareto candidate。
- 每个 candidate 有 accuracy、latency、model size、memory。
- 最终模型必须通过 ONNX/TFLite/TensorRT/OpenVINO 之一验证。
