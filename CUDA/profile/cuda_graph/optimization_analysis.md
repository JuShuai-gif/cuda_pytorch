# 四个 VLA 推理库的优化策略分析

> realtime-vla / realtime-vla-v2 / realtime-vla-flash / learn_flashRT

---

```
                       Triton 路线                          CUDA 路线
              ┌─────────┼─────────┐                  ┌─────────┴──────────┐
         realtime-vla    v2     flash             learn_flashRT (手写 PTX)
              │           │        │                   │
         41+ kernel   31 kernel  35+ kernel        187 .cu + 126 .cuh
         BF16 only   +RTC预计算  +推测解码+FA多后端   FP8/FP4/INT8/INT4
         无CUTLASS   无CUTLASS  多种Flash Attn        CUTLASS全覆盖
         SDPA        SDPA        FA2/FA3             FA2/FA4/FMXA/Sage2
         预分配KVCache            全缓存快照           Paged KV+FP8 KV
          ~60ms       ~50ms       ~30ms (草稿命中)    ~45ms (FP8)
                                                    Megakernel+TMA+StreamK
```

**结论：flashRT 是另外三个的终极形态。** 它们在 Triton 层做的融合、预计算、CUDA Graph、Flash Attention —— flashRT 全部以手写 CUDA + 量化做到了更极致。

---

## 各库简析

### realtime-vla（基线）

Triton 手写 kernel + CUDA Graph + BF16。三个模型：DM0（15 kernel）、Pi0（22 kernel）、Pi05（5 kernel）。
- 最多的是**算子内融合**：matmul+bias+residual、QKV+RoPE+split、LayerNorm+matmul+GELU
- 权重离线融合：RMSNorm 缩放吸收进 GEMM 权重、归一化 + 输出投影 + dt 三合一
- Split-K MatMul（Pi0 专用，vision FFN `M=512, K=4304`, SPLIT_K=4）

### realtime-vla-v2（+RTC）

在 v1 基础上增加 `pi05rtc_infer.py`：
- AdaRMS 调制预计算（`_build_adarms_mod_bases`）：初始化时一次性算完 10 步 × 18 层的时间调制向量，省在线 MLP
- 动作预填充掩码：RTC 流式推理的块间衔接
- MPC 轨迹后处理（`optimizer.py`）

### realtime-vla-flash（+推测解码）

在 v2 基础上增加：
- **Flash Attention 多后端**：FA2/FA3 + CUTLASS FMHA + SDPA
- **推测 VLA 解码**：DraftChunkHead（单层 Gemma decoder）廉价预测 → BK 并行验证 → 半径接受准则。接受率好的时候 **3-5x 吞吐**
- 全缓存快照、语言嵌入缓存、LoRA 合并、roofline 分析

### learn_flashRT（终点站）

手写 PTX MMA + CUTLASS 3.x TMA + 全精度量化 + Megakernel。
- **Kernel 融合**：5→1、6→1 融合，launch ~21,000 → ~2,840（-85%）
- **量化**：FP8/NVFP4/INT8/INT4 全覆盖，静态校准消除 630 个动态 quant kernel
- **Megakernel**：Norm+GEMM_up+GELU+GEMM_down+Residual 单次 launch
- **FA**：FA2(vendored) + FA4(自研 CUTE TMA) + FlashInfer XQA + CUTLASS FMHA + Sage2
- **MoE**：7 种 tiling 策略
- **3D Conv**：Motus FP4/FP8 手写

---

## 对 Pi05 而言：flashRT 已做到饱和

以 Pi05 为目标模型，逐一验证另外三个库的独有技术：

| 技术                   | 来源  | Pi05 适用？ | flashRT 状态 | 结论               |
| ---------------------- | ----- | ----------- | ------------ | ------------------ |
| 时间偏置表预计算       | v1    | ❌           | N/A          | Pi05 用 AdaRMS，不走 bias |
| Split-K MatMul         | v1    | ❌           | N/A          | 解码器 M=10 太小        |
| AdaRMS 调制预计算      | v2    | ✅           | **已有**         | `pi05_rtx.py:369`，比 v2 更完备 |
| 动作预填充掩码         | v2    | ✅           | 部分         | 不省推理时间         |
| 轨迹 MPC               | v2    | ✅           | 缺失         | 纯 CPU 后处理       |
| 推测 VLA 解码          | flash | ✅           | 缺失         | **唯一有量级提升的** |
| 全缓存快照/嵌入缓存    | flash | ✅           | 部分         | 工程优化，非性能瓶颈 |

**Pi05 kernel 层面的性能优化已饱和。** 融合省的是微秒级 launch，量化已到 FP4，Megakernel 已覆盖 FFN 全路径。剩余 3 项融合（bias_add 替换、Vision bias+LN、QKV+RoPE）合计 ~1.5-3.5ms，**加起来不到 10%。**

**唯一有量级提升的方向：推测 VLA 解码（3-5x 吞吐），但需训练草稿模型 + 重写推理流程，不是 kernel 工程范畴的项目。**

---

## 演进路线

```
Phase 1   Triton 手写 kernel + CUDA Graph + BF16        (realtime-vla)
Phase 2   + 预计算时间调制 + RTC 流式                     (realtime-vla-v2)
Phase 3   + Flash Attention 多后端                        (realtime-vla-flash)
Phase 4   + 手写 PTX + CUTLASS + FP8/FP4/INT8 + Megakernel (learn_flashRT)
── kernel 优化已到天花板 ──
Phase 5   推测 VLA 解码（DraftChunkHead + BK 验证）        (realtime-vla-flash 提出)
         → flashRT 尚未实现，3-5x 吞吐潜力
```
