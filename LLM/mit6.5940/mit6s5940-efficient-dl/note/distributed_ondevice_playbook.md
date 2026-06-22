# 分布式训练与端侧训练 Playbook

## 分布式训练

| 技术 | 解决问题 |
|---|---|
| DDP | 数据并行和梯度同步 |
| FSDP/ZeRO | 参数、梯度、optimizer state 分片 |
| Tensor Parallel | 单层矩阵过大 |
| Pipeline Parallel | 层数过多和显存不足 |
| Activation Checkpointing | 用重算换显存 |

工业验收：samples/s、GPU utilization、communication overlap、peak memory、checkpoint time、failure recovery。

## 端侧训练

- Federated averaging 关注通信量和隐私。
- TinyTL/adapter-only training 关注 activation memory。
- On-device personalization 需要小步更新和断点恢复。

工业坑：

- optimizer state 往往比参数更占内存。
- BatchNorm 在小 batch/on-device 场景不稳定。
- 通信压缩会影响收敛，需要和最终 task metric 一起看。
