# 07 · 量化：PTQ / MinMax / KL / Percentile / 混合精度

> 对应代码：`tools/edge-quantize/edge-quantize.cpp`、自定义类型 `!edge.qtensor`、属性 `#edge.quant_params`
> 验证：`ninja -C build check-edge`（quantize 测试通过）；`edge-quantize --edge-out=reports`

---

## 1. 中文原理讲解

后训练量化（PTQ）把 FP32 模型转成 INT8：用少量校准数据估计每个张量的数值范围，定出量化参数
`scale`（对称量化：`real ≈ scale * q`，`q∈[-127,127]`），从而用 INT8 算子换取吞吐与显存。本模块实现：

1. **校准数据集加载器**：合成激活样本（高斯 N(0,1) + 0.1% 极端离群点 @ ±15σ，模拟真实激活长尾）。
2. **三种校准算法**（决定阈值 → `scale = threshold/127`）：
   - **MinMax**：`threshold = max|x|`。简单，但对离群点敏感（被极端值拉大）。
   - **Percentile**：取 `|x|` 的 p 百分位（如 99.9%）。裁掉离群点，鲁棒。
   - **KL 散度（熵校准）**：枚举候选阈值，在 `[0,threshold)` 内把分布量化到 128 级再展开（消除粗网格空隙），
     对尾部 `[threshold, max)` 计入裁剪惩罚，取 KL(P‖Q) 最小的阈值。这是 TensorRT 的默认校准。
3. **INT8 量化模拟**：round-trip（量化→反量化）测 MSE 与 SQNR；同时报告 **full SQNR**（全体数据，含离群点）
   与 **body SQNR**（仅主体 `|x|<=4`）。
4. **混合精度决策**：INT8 的 SQNR 低于阈值（默认 30 dB）的张量保留 FP16。
5. **报告**：quantization / accuracy / latency 三份。

**关键洞见（验证数据）**：对"高斯 + 0.1% @ ±15σ"：
| 方法 | 阈值 | full SQNR | body SQNR |
|------|------|-----------|-----------|
| MinMax | 15.96 | 29.62 | 28.86 |
| Percentile(99.9) | 3.66 | 10.37 | **41.52** |
| KL | 15.97 | 29.61 | 28.85 |

MinMax/KL 覆盖全范围 → full SQNR 高，但主体量化粗 → body SQNR 低；Percentile 裁掉极端离群点
→ 牺牲少量 full SQNR，换来主体 **+12.7 dB** 的量化精度。**真实模型精度由主体精度主导**，故百分位/熵校准
通常优于 MinMax——这正是 TensorRT 默认用熵(KL)校准、而非 MinMax 的根本原因。

量化参数最终用自定义类型 `!edge.qtensor<tensor<...xi8>, scale, zp>` 和属性 `#edge.quant_params<scale, zp>`
承载（Module 03 已实现），把"已量化"在类型系统层面显式化。

## 2. 工业背景

量化是边缘部署降延迟/降显存最有效的手段之一（INT8 相对 FP32 约 2-4x 吞吐、4x 显存）。难点不在"乘除"，
而在**如何选 scale 使精度损失最小**——这就是校准。TensorRT/TPU-MLIR/CANN 都有完整 PTQ + 校准流程。

## 3. TensorRT 对应模块

- KL 校准 ≈ `IInt8EntropyCalibrator2`（TensorRT 默认，本模块对标实现）。
- MinMax 校准 ≈ `IInt8MinMaxCalibrator`。
- 混合精度 ≈ builder 的 per-layer 精度选择（INT8/FP16/FP32 按精度-性能权衡）。

## 4. TVM 对应模块

≈ TVM 的 `relay.quantize`（calibrate → realize）/ AutoTVM 量化；KL 校准同思路。

## 5. TPU-MLIR 对应模块

≈ `run_calibration`（产出 calibration table，默认 KL）+ `model_deploy` 带表 lowering 到量化 `tpu` 算子；
支持 per-channel 权重量化、混合精度。本模块的"校准与部署分离"理念即源于此。

## 6. Ascend CANN 对应模块

≈ AMCT（Ascend Model Compression Toolkit）的 PTQ：校准收集分布 → INT8 → 部署。

## 7. 性能收益

- INT8 算子吞吐约为 FP32 的 2-4x，显存/带宽降 4x；本模块 latency 报告给出粗略端到端加速估算。
- per-channel 权重量化 + 熵校准可把精度损失压到 <1%（常见 CV/检测模型）。

## 8. Trade-off（核心）

- **full SQNR vs body SQNR**：覆盖范围 vs 主体精度的权衡（见上表）。校准方法的选择本质是这个权衡。
- 对称 vs 非对称：对称简单（无 zero-point 偏移），非对称（带 zp）对非零中心分布（如 ReLU 后）更准。
- per-tensor vs per-channel：权重 per-channel 精度高但实现复杂；激活通常 per-tensor。
- 量化对**机器人/安全攸关**任务要格外谨慎：动作精度退化可能危险，需混合精度保留敏感层。

## 9. 常见 Bug（本模块真实踩坑）

1. **KL 退化**：候选阈值接近量化级数时 Q≈P 使 KL→0，算法会退化地选最小阈值（裁掉几乎所有信号）。
   修复：在 `[0,threshold)` 内展开 Q 到与 P 同支撑 + 对尾部计入裁剪惩罚，使两端均非退化。
2. **SQNR 指标误导**：只看全体 SQNR 会得出"MinMax 最好"的错误结论；必须区分 full/body SQNR，
   因为模型精度由主体主导。这是本模块最重要的工程认知。
3. **离群点处理**：MinMax 被极端值拉大 scale → 主体量化粗。百分位/熵校准是标准应对。
4. **splat/per-channel**：权重 dense 可能是 splat，读取要广播；per-channel 要按输出通道分别定 scale。

## 10. 调试方法

- `edge-quantize`：对比三种校准的 full/body SQNR，直观看权衡。
- 用合成分布（高斯 + 可控离群点）做单测，校准阈值与 SQNR 可定性预测。
- 真实模型：先量化权重（数据已知、确定性），再量化激活（需校准数据）。

## 11. Profiling 方法

- accuracy 报告：MSE / full SQNR / body SQNR。
- latency 报告：INT8/FP16 张量计数 + 估算加速；真实加速需在目标硬件实测。
- 端到端精度：量化前后跑验证集对比 top-1/mAP（本模块用 SQNR 作代理）。

## 12. 在机器人 / VLA 中的应用

VLA 模型大、attention 多，量化是上边缘 SoC 的关键。策略：视觉主干/线性层 INT8（body SQNR 高即可），
对动作头等**安全敏感层**用混合精度保留 FP16（避免动作精度退化带来的安全风险）。熵校准保证主体精度，
混合精度兜住敏感层——这是机器人量化部署的标准权衡，本模块的 full/body SQNR + 混合精度决策正为此服务。

> 至此量化闭环：校准 → 量化参数(`!edge.qtensor`/`#edge.quant_params`) → 混合精度 → 报告，
> 与图优化(05)、lowering(10)、内存(09)、运行时(11/12)共同构成完整的边缘 AI 编译栈。
