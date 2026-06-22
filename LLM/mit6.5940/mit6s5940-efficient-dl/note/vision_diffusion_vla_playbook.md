# ViT / Diffusion / VLA 高效部署 Playbook

## ViT

- Token pruning/merging 降低 attention quadratic cost。
- Window/local attention 降低 memory bandwidth。
- Patch embedding 和 layout transform 可能成为实际瓶颈。

## Diffusion

- 最大收益通常来自减少采样步数：DDIM、distillation、consistency model。
- UNet attention 和 cross-attention 是 memory hotspot。
- 量化要谨慎处理 timestep embedding、attention、final decoder。

## VLA / Robotics

VLA 不能只看 classification accuracy。必须看：

- action MSE、max action deviation。
- trajectory drift、rollout success rate。
- control-loop deadline miss rate。
- P99 latency under thermal/load stress。

## 工业坑

- 视觉 encoder 压缩后，语言/action head 指标可能不敏感，但 rollout 会失败。
- action projection 最后一层低比特量化可能造成控制抖动。
- 平均 latency 满足控制周期，不代表 P99 满足。
