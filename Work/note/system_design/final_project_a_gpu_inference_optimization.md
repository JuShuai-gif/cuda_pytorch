# 最终项目 A：GPU Inference Optimization Report

## 目标

对一个模型走完优化阶梯，记录每一层的 Before/After 完整指标，证明"每一步为什么快"，而不是一句"提速 X%"。

## 模型与硬件

```text
模型：残差 MLP（hidden=1024, layers=4, batch=1, seq=16）
硬件：NVIDIA Thor（sm_110，统一内存，20 SM）
测量：CUDA Event，p50/p95/p99，多次取中位数（clock/thermal 有波动）
```

## 优化阶梯（Before → After）

| stage | p50(us) | p95(us) | p99(us) | 显存(MB) | kernel 数 | 加速比 |
|---|---|---|---|---|---|---|
| eager fp32 | 441.2 | 481.2 | 496.3 | 43.5 | 20 | 1.00x |
| torch.compile | 419.0 | 455.8 | 472.8 | 43.5 | 17 | 1.05x |
| eager fp16 | 151.5 | 394.0 | 2442.7 | 26.6 | 20 | 2.91x |
| Triton RMSNorm | 138.2 | 285.7 | 451.7 | 43.4 | 20 | 3.19x |
| CUDA Graph | 108.4 | 136.2 | 139.5 | 62.4 | 1 | 4.07x |
| TensorRT FP16（Stage 7） | ~90 | - | - | - | - | ~4.9x |

## 每一步为什么有效

### 1. fp32 → fp16：2.91x（最大收益）

Tensor Core 的 fp16 吞吐是 fp32 CUDA core 的数倍（Stage 8 实测 14x），GEMM 是 compute-bound，收益直接兑现。显存从 43.5MB 减到 26.6MB（权重减半）。**代价**：精度从 fp32 的 `1e-4` 降到 fp16 的 `1e-3`（可接受）。

### 2. fp16 → Triton RMSNorm：1.09x（小收益）

把 `nn.LayerNorm` 换成 fused Triton RMSNorm。收益小，因为 `nn.LayerNorm` 本身已是 fused（Stage 6 结论：无 GEMM 的融合收益大，但这里替换的是已 fused 算子）。**真实收益在**：去掉 mean 中心化（RMSNorm 比 LayerNorm 少一步 reduction），且为后续自定义算子铺路。

### 3. → CUDA Graph：1.27x（batch=1 关键）

batch=1 时 20 个 kernel 的 launch 开销占相当比例。CUDA Graph 把 20 次 launch 折叠成 1 次（kernel 数 20 → 1），p99 从 451 降到 139（**同时消除了 jitter**，呼应 Stage 14/26 实时性）。显存增加（62.4MB）是因为 graph 静态 buffer 预分配。

### 4. torch.compile：只快 5%（小模型）

torch.compile 的融合收益被编译/调度 overhead 抵消（batch=1 小模型），且 kernel 数只从 20 降到 17。**结论：小模型上 torch.compile 不是银弹**（Stage 7 结论一致）。

### 5. TensorRT FP16：~4.9x（自动化最优）

TensorRT 的 layer fusion + tactic selection 全自动达到最优，接近但超过手写优化的 CUDA Graph 结果（~90 vs 108us）。**手写优化的价值在 TensorRT 覆盖不到的算子**（Plugin，Stage 7）。

## 关键结论

```text
1. 总加速 4.07x（手写阶梯）→ 4.9x（TensorRT）
2. fp16 贡献最大（2.91x）：Tensor Core 是免费午餐
3. CUDA Graph 的 p99 从 451→139：不只快，还稳（实时关键）
4. torch.compile 在小模型上收益有限：不是银弹
5. 手写优化（Triton/CUDA Graph）的价值在 TensorRT 覆盖不到的算子
```

## 测量注意事项（本项目的诚实记录）

```text
1. edge 设备的 clock 随温度动态调整（Stage 15）：长时间 benchmark 后温度升到 65°C，
   clock 降频，导致 fp16 的 p50 在 151-326us 之间波动。
2. 用多次测量取中位数（p50 维度），p99 仍有长尾（clock 抖动/allocator）。
3. 结论基于 p50 的相对加速比，绝对延迟是"本机当前状态"的快照。
```

## 复现

```bash
export PYTHONPATH="$PWD/Work/src"
python -m projects.inference_optimization.benchmark --device cuda --output /tmp/opt.json
python -m projects.inference_optimization.report --input /tmp/opt.json --output /tmp/report.md
```
