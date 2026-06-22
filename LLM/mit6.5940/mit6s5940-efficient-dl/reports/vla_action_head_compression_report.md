# VLA Action Head 压缩实验报告

**生成时间**: 2026-06-12 11:06:59  |  **设备**: cuda  |  **剪枝率**: 50%  |  **量化**: INT8

> 模拟 ACT (Action Chunking Transformer) 风格的 action head 压缩实验
> 模型: VLAEncoder + ActionChunkHead, 输入 vision(256) + state(7), 输出 100 个 action chunks × 7 DoF

## 综合对比

| 方法 | 参数量(M) | 大小(MB) | 延迟(ms) | P99(ms) | 控制频率(Hz) | 吞吐(s/s) | 内存(MB) | Action MSE | Pos MSE | Ori MSE | Gripper MSE | 平滑度 | Cosine Sim | 端侧 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| VLA ActionHead FP32 Baseline | 2.553 | 9.7543 | 0.6758 | 0.8079 | 1238 | 10402.0 | 57.8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | No |
| VLA ActionHead Pruned (50%) | 2.553 | 9.7543 | 0.4030 | 0.7919 | 1263 | 28599.4 | 67.6 | 0.009950 | 0.004965 | 0.009690 | 0.025686 | 0.210432 | 0.9544 | Yes |
| VLA ActionHead PTQ INT8 | 2.553 | 9.7543 | 0.5250 | 1.0682 | 936 | 25406.4 | 77.3 | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.098904 | 1.0000 | Yes |
| VLA ActionHead Prune+INT8 | 2.553 | 9.7543 | 0.3135 | 0.3675 | 2721 | 29836.8 | 87.0 | 0.010038 | 0.004979 | 0.009777 | 0.026002 | 0.209973 | 0.9544 | Yes |

## 各自由度精度退化分析

| 方法 | Pos MAE (mm) | Ori MAE (deg) | Gripper MSE | Max Deviation | Chunk Variance |
|------|-------------|--------------|-------------|--------------|---------------|
| VLA ActionHead FP32 Baseline | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 |
| VLA ActionHead Pruned (50%) | 55.7098 | 4.7479 | 0.025686 | 0.334931 | 0.024522 |
| VLA ActionHead PTQ INT8 | 1.0877 | 0.0653 | 0.000002 | 0.004747 | 0.042484 |
| VLA ActionHead Prune+INT8 | 55.7716 | 4.7720 | 0.026002 | 0.333333 | 0.024414 |

## 延迟分析（控制回路适用性）

| 方法 | P99 延迟 (ms) | 最大控制频率 (Hz) | 满足 30Hz? | 满足 100Hz? |
|------|-------------|------------------|-----------|------------|
| VLA ActionHead FP32 Baseline | 0.8079 | 1238 | Yes | Yes |
| VLA ActionHead Pruned (50%) | 0.7919 | 1263 | Yes | Yes |
| VLA ActionHead PTQ INT8 | 1.0682 | 936 | Yes | Yes |
| VLA ActionHead Prune+INT8 | 0.3675 | 2721 | Yes | Yes |

## 端侧部署评估

| 设备 | 内存限制 | 推荐方案 |
|------|---------|---------|
| Jetson Orin (8GB) | 58 MB OK | VLA ActionHead FP32 Baseline |
| Jetson Orin (8GB) | 68 MB OK | VLA ActionHead Pruned (50%) |
| Jetson Orin (8GB) | 77 MB OK | VLA ActionHead PTQ INT8 |
| Jetson Orin (8GB) | 87 MB OK | VLA ActionHead Prune+INT8 |

## 方法说明

- **VLA ActionHead FP32 Baseline**: Baseline FP32 model
- **VLA ActionHead Pruned (50%)**: Unstructured pruning: 50% weights zeroed. Requires sparse kernel for real speedup on edge devices.
- **VLA ActionHead PTQ INT8**: PTQ INT8 quantization. Action output layer should be validated with rollout MSE on real hardware.
- **VLA ActionHead Prune+INT8**: Combined pruning + quantization maximizes compression for edge deployment.

## VLA 部署建议

1. **Action head 压缩优先级**: MLP 中间层 (residual blocks) > 输入投影 > action 输出层。输出层量化需谨慎验证 rollout MSE。
2. **延迟预算**: 机器人控制通常需要 ≤10ms (100Hz) 或 ≤33ms (30Hz)。P99 延迟必须满足控制回路周期。
3. **Action chunk consistency**: 压缩后 chunk 间的 smoothness 应保持与 baseline 一致。smoothness 急剧增大说明模型输出不稳定。
4. **验收指标**: 不能只看总体 MSE。必须分解为 position (mm)、orientation (deg)、gripper (开合) 分开验收。
5. **真实部署**: 本实验使用合成数据。真实部署需在机器人硬件上做 rollout evaluation，计算 success rate 和 trajectory error。

---
*报告由 vla_action_head_compress.py 自动生成于 2026-06-12 11:06:59*