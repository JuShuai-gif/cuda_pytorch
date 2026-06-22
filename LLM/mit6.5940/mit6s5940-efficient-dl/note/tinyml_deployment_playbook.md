# TinyML / TinyEngine / MCU 部署 Playbook

TinyML 的核心瓶颈通常是 SRAM activation memory，而不是参数量。

## MCU 约束

| 资源 | 典型约束 |
|---|---|
| Flash | 权重和代码常驻空间 |
| SRAM | activation、arena、stack、IO buffer |
| 算子 | CMSIS-NN/TFLite Micro/TinyEngine 支持范围 |
| 功耗 | duty cycle 和 thermal budget |

## 工业流程

```text
model design -> int8 quantization -> operator legalization -> memory planning -> C code/kernel selection -> target benchmark
```

## 必须讲清楚

- im2col 可能让 activation buffer 膨胀 10-20x。
- per-channel int8 weight 需要 requantization 和 accumulator scale。
- depthwise conv 不一定比 regular conv 快，取决于 kernel 实现。
- 静态 arena allocator 比动态 malloc/free 更可靠。

## 验收

- Flash bytes、peak SRAM bytes、arena bytes。
- 每个 op 的 latency 和 kernel path。
- INT8 accuracy delta。
- 目标板卡上实测，而不是桌面 CPU 模拟。
