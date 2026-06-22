# 附录 A：算力-板卡-模型效率全景图

> 本附录是 MIT 6.5940 的硬件基础参考。不讲理论公式，只讲工程真相：你手上有某块板卡，到底能跑什么模型？为什么标称 TOPS 是骗人的？内存带宽才是真正的瓶颈？

---

## 1. 一张表看懂所有边缘 AI 硬件

| 级别          | 代表芯片/板卡                       | 标称算力          | 内存带宽            | SRAM/显存     | 功耗     | 能跑的模型                                    |
| ------------- | ----------------------------------- | ----------------- | ------------------- | ------------- | -------- | --------------------------------------------- |
| **MCU**           | STM32F746 (Cortex-M7)               | ~0.1 GOPS         | 0.3 GB/s (SRAM)     | 320 KB SRAM   | <1W      | MCUNet, 简单 KWS, 2 层 CNN                    |
| **MCU+**          | STM32H743 (Cortex-M7 + DSP)         | ~0.5 GOPS         | 1 GB/s (SRAM)       | 1 MB SRAM     | <1W      | MCUNetV2, 人脸检测(精简), 异常检测             |
| **MCU Pro**       | Arduino Portenta H7                 | ~0.5 GOPS         | —                   | 8 MB SDRAM    | <2W      | TinyML 全系列, 简单视觉分类                    |
| **跨界**          | NXP i.MX RT1060 (Cortex-M7 600MHz)  | ~1 GOPS           | 1.5 GB/s            | 1 MB SRAM     | <3W      | MCUNetV2 full, MobileNetV1-0.25x              |
| **IoT SoC**       | ESP32-S3 (Xtensa LX7)               | ~0.5 GOPS（FPU弱） | —                   | 512 KB SRAM   | <1W      | TFLite Micro, 关键词检测, 手势识别(小)         |
| **轻量 NPU**      | Google Coral Edge TPU               | 4 TOPS (INT8)     | —                   | 8 MB SRAM     | <5W      | MobileNetV2 SSD, EfficientDet-Lite            |
| **手机 NPU**      | Apple A17 Pro ANE                   | 35 TOPS (INT8)    | —                   | 共享 8GB LPDDR | <5W (ANE) | CoreML 模型, MobileNetV3, ViT-Small           |
| **手机 NPU**      | 骁龙 8 Gen 3 Hexagon NPU            | 45 TOPS (INT8)    | ~77 GB/s (LPDDR5)   | 共享 12GB      | <5W (NPU) | SNPE 模型, EfficientViT, YOLOv8-nano          |
| **边缘 GPU**      | NVIDIA Jetson Orin Nano 8GB         | 40 TOPS (sparse)  | 68 GB/s             | 8 GB LPDDR5   | 7–15W    | YOLOv8-s, ResNet50, Whisper-tiny, 7B LLM(慢)  |
| **边缘 GPU**      | NVIDIA Jetson Orin NX 16GB          | 100 TOPS (sparse) | 102 GB/s            | 16 GB LPDDR5  | 10–25W   | BEVFusion, VLA-RT1, 7B LLM(AWQ INT4, 可用)    |
| **边缘 GPU+**     | NVIDIA Jetson AGX Orin 64GB         | 275 TOPS (sparse) | 204 GB/s            | 64 GB LPDDR5  | 15–60W   | 自动驾驶全栈, 70B LLM(AWQ INT4, 勉强)         |
| **桌面 GPU**      | NVIDIA RTX 4090                     | 661 TOPS (INT8 TC)| 1008 GB/s           | 24 GB GDDR6X  | 450W     | 70B LLM INT4, Stable Diffusion XL, 一切        |
| **数据中心 GPU**  | NVIDIA H100                          | 1979 TOPS (FP8)   | 3350 GB/s (HBM3)    | 80 GB HBM3    | 700W     | GPT-4 训练, 70B-405B LLM, 千卡集群            |
| **数据中心 GPU**  | NVIDIA B200                          | 4500 TOPS (FP4)   | 8000 GB/s (HBM3e)   | 192 GB HBM3e  | 1000W    | GPT-5 训练, MoE 万亿参数                       |
| **国产 NPU**      | 华为昇腾 910B                        | 320 TOPS (FP16)   | 1228 GB/s (HBM2e)   | 64 GB HBM2e   | 310W     | LLaMA-70B, 千卡训练                            |
| **车载芯片**      | 特斯拉 FSD Chip 2 (HW4.0)            | ~50 TOPS (INT8)×2  | —                   | 32 GB LPDDR5  | ~100W    | 全栈自动驾驶 (8 路视频 + BEV + Transformer)     |
| **车载芯片**      | 高通 Snapdragon Ride Flex            | ~200 TOPS (INT8)  | —                   | —             | ~65W     | 自动驾驶 + 座舱 AI 合二为一                    |

> **关键备注**: 标称 TOPS 是在 "100% 利用率" 下的理论峰值。实际利用率看下一节。

---

## 2. 标称 TOPS 的残酷真相：实际利用率

### 2.1 为什么你的 40 TOPS 板卡实际只有 4 TOPS？

```
标称算力（Theoretical Peak TOPS）
    ↓
    × 算子效率（~30–70%）  ← 矩阵乘法 VS 逐元素操作，差异巨大
    ↓
    × 内存带宽利用率（~40–80%）  ← 数据搬进来才算，这步常常是瓶颈
    ↓
    × batch/并发利用率（~20–100%）  ← 边缘推理通常是 batch=1
    ↓
    = 实际有效 TOPS
```

**典型场景**：

| 硬件            | 标称 TOPS | 实际有效 TOPS | 利用率 | 瓶颈在哪里                          |
| --------------- | --------- | ------------- | ------ | ----------------------------------- |
| GPU (大batch)   | 275       | 120–180       | 45–65% | 算子调度 + 内存带宽                 |
| GPU (batch=1)   | 275       | **15–40**     | 5–15%  | **batch=1 完全吃不饱 GPU**              |
| NPU (batch=1)   | 45        | 20–35         | 45–80% | NPU 对小 batch 比 GPU 友好          |
| MCU (batch=1)   | 0.1       | 0.05–0.08     | 50–80% | MCU 没有并行浪费，但算力基数太低    |
| TPU Edge (INT8) | 4         | 3–3.5         | 75–90% | TPU 利用率极高（专为 INT8 小模型设计） |

### 2.2 batch=1 杀伤力最大

GPU 的并行度依赖 batch 填充。当你只推理 1 张图：

| 操作          | batch=32 利用率   | batch=1 利用率     |
| ------------- | ----------------- | ------------------ |
| Conv2D 3×3    | 85%               | **15%** ← 张量核吃不饱 |
| GEMM (FC 层)  | 90%               | **8%** ← 更惨          |
| Depthwise Conv | 30%（本来就不高） | 5%                 |
| Attention     | 95%               | 25%（FlashAttention 有改善） |

**这就是为什么手机 NPU 在小模型上比 GPU 快——它专门为 batch=1 优化了流水线。**

---

## 3. 内存带宽才是真正的王（Memory Wall）

### 3.1 一个模型推理发生了什么

```
每次推理 = 把所有权重从内存读出来 + 做计算 + 把中间结果写回去

对于 ResNet50 (25M params, ~4G FLOPs):

  数据搬运量 ≈ 100 MB (权重) + 200 MB (中间激活读写) = 300 MB
  计算量 ≈ 4G FLOPs

  如果内存带宽 = 68 GB/s (Jetson Orin Nano):
    搬运时间 = 300 MB / 68 GB/s ≈ 4.4 ms

  如果 GPU 算力 = 40 TOPS (标称), 实际 batch=1 利用率 15% = 6 TOPS:
    计算时间 = 4G FLOPs / 6 TOPS ≈ 0.67 ms

→ 数据搬运时间 (4.4ms) >> 计算时间 (0.67ms)
→ 模型是 "memory-bound" 而非 "compute-bound"
→ 提升算力没用，瓶颈在内存带宽
```

### 3.2 不同模型的内存-计算比

| 模型类型                       | 计算/带宽比  | 瓶颈在         | 适用硬件              |
| ------------------------------ | ------------ | -------------- | --------------------- |
| 大卷积 (ResNet, VGG)           | 高（计算密集）  | 计算为主       | GPU/NPU               |
| Depthwise Conv (MobileNet)     | 低（搬运密集）  | **带宽为主**       | NPU（NPU 的 DW Conv 效率好） |
| Transformer (Attention)        | 中            | **KV Cache 带宽** | HBM/大带宽            |
| LLM decode (自回归)            | **极低**         | **纯带宽瓶颈**     | HBM 必须，GDDR 不够   |
| 小 MLP / 1×1 Conv              | 低            | 带宽为主       | 任何硬件都慢          |

> **直觉**: GEMM/大卷积 = 计算密集（多数据做多次运算），适合 GPU。逐元素操作/depthwise/小矩阵 = 带宽密集（读出来只做一次运算就扔了），**什么硬件都慢**，唯一的办法是**减少这些操作或融合它们**。

### 3.3 LLM Decode 为什么是极端的 Memory-Bound

```
LLaMA-7B 单 token 推理：
  读取权重: 7B × 2 bytes (FP16) = 14 GB
  读取 KV Cache (4096 tokens): ~0.5 GB
  计算量: ~14 GFLOPs

  在 H100 (3350 GB/s) 上:
    搬运时间 = 14.5 GB / 3350 GB/s = 4.3 ms
    计算时间 = 14 GFLOPs / 1979 TOPS = 0.007 ms

→ 搬运时间是计算时间的 600 倍！
→ LLM 推理的本质是 "在 HBM 上不停翻书"，不是计算
→ 这就是为什么量化（减少搬运数据量）= 立竿见影加速
→ 这也是为什么 vLLM/PagedAttention 的核心优化是 "怎么少搬数据"
```

---

## 4. 不同尺寸模型的实际可运行硬件门槛

### 4.1 视觉模型

| 模型                     | 参数量   | FLOPs     | 最低 SRAM/显存 | 推荐硬件                                           |
| ------------------------ | -------- | --------- | -------------- | -------------------------------------------------- |
| MCUNet (ImageNet 分类)   | 0.7M     | 150M      | 256 KB SRAM    | STM32F746/H743                                     |
| MobileNetV1-0.25x        | 0.5M     | 40M       | 512 KB SRAM    | ESP32-S3, i.MX RT1060                              |
| MobileNetV2-1.0x         | 3.5M     | 300M      | 32 MB 以上     | Coral TPU, 手机 NPU                                |
| EfficientNet-B0          | 5.3M     | 390M      | 50 MB+         | 手机 NPU, Jetson Nano                              |
| YOLOv8-nano              | 3.2M     | 8.7G      | 128 MB+        | 手机 NPU, Orin Nano                                |
| YOLOv8-small             | 11.2M    | 28.6G     | 256 MB+        | Orin NX 以上                                       |
| EfficientViT-B0          | 3.5M     | 0.5G      | 64 MB+         | 骁龙 8 Gen 3, Orin Nano（比 MobileNet 快 3x）        |
| ViT-L/14                 | 304M     | ~200G     | 4 GB+          | Orin AGX, RTX 4090                                 |
| Stable Diffusion XL      | 2.6B     | ~500G     | 12 GB+         | RTX 4090 (SVDQuant 4-bit 可降到 4GB)               |
| SANA (Linear DiT, 4-bit) | 0.6B     | ~100G     | 3 GB           | RTX 4090, Orin AGX                                 |

### 4.2 语言/多模态模型

| 模型                          | 参数量 | 最低显存 (FP16) | 最低显存 (INT4) | 推荐硬件 (INT4)                     |
| ----------------------------- | ------ | --------------- | --------------- | ----------------------------------- |
| DistilBERT                    | 66M    | 130 MB          | 35 MB           | 任何设备                            |
| MobileBERT                    | 25M    | 50 MB           | 15 MB           | MCU+（勉强）, 手机                   |
| GPT-2 Small                   | 124M   | 250 MB          | 65 MB           | 手机, Orin Nano                     |
| Whisper-tiny                  | 39M    | 80 MB           | 25 MB           | 手机, Orin Nano                     |
| LLaMA-3.2-1B                  | 1.2B   | 2.5 GB          | 0.6 GB          | 手机 NPU, Orin Nano                 |
| LLaMA-3.2-3B                  | 3.2B   | 6.5 GB          | 1.6 GB          | Orin NX, 手机（勉强）                 |
| Qwen2.5-7B                    | 7.6B   | 15 GB           | 4 GB            | Orin NX (AWQ), Orin AGX             |
| LLaMA-3-70B                   | 70B    | 140 GB          | 35 GB           | Orin AGX 64GB (AWQ INT4, 勉强), H100 |
| GPT-4 量级                    | ~1.8T (MoE) | 3.6 TB         | —               | 百卡 H100 集群                      |

### 4.3 VLA / 机器人策略模型

| 模型                   | 参数量 | 部署要求               | 推荐硬件                                      |
| ---------------------- | ------ | ---------------------- | --------------------------------------------- |
| RT-1 (robotics)        | 35M    | 150 MB                 | Orin NX 以上                                  |
| Octo-small             | 27M    | 100 MB                 | Orin Nano                                     |
| Octo-base              | 93M    | 400 MB                 | Orin NX                                       |
| π0 (Physical Intelligence) | 3.3B | 13 GB (FP16) | Orin AGX (AWQ INT4 可到 2GB)                  |
| RT-2 (PaLI-X 基座)     | ~10B   | 40 GB (FP16)           | 必须云端推理，蒸馏后才上边缘                  |

---

## 5. 选型决策树

```
你要部署的设备功耗预算是多少？
│
├── <1W (电池/纽扣电池/能量采集)
│   └── MCU 级 (STM32, nRF52)
│       └── 模型必须 <500 KB, FLOPs < 100M
│       └── 用 MCUNet + TinyEngine + INT8
│       └── 只能做: KWS, 异常检测, 简单分类
│
├── 1-5W (手机协处理器 / 智能传感器)
│   └── 低功耗 DSP / 小 NPU
│       └── 模型 < 5 MB, FLOPs < 500M
│       └── 用 MobileNet/ShuffleNet + INT8
│       └── 能做: 人脸检测, 手势识别, 简单 VAD
│
├── 5-15W (Jetson Orin Nano / 高端手机)
│   └── 边缘 GPU/NPU
│       └── 模型 < 500 MB, FLOPs < 20G
│       └── 用 YOLOv8 + TensorRT INT8 + 通道剪枝
│       └── 能做: 实时检测+分割, 1B LLM, VLA(RT-1)
│
├── 15-30W (Orin NX / 工业 PC)
│   └── 中等算力
│       └── 模型 < 4 GB, FLOPs < 100G
│       └── 用 YOLO + BEVFusion + AWQ 7B LLM
│       └── 能做: 自动驾驶感知, 仓库机器人全栈, 本地 LLM
│
├── 30-60W (Orin AGX / 车载)
│   └── 高端边缘
│       └── 模型 < 16 GB, FLOPs < 500G
│       └── 用 TensorRT + SVDQuant + 多模型并行
│       └── 能做: 全栈自动驾驶, 70B LLM(勉强), 多路视频
│
└── >100W (GPU 集群 / 云端)
    └── 不做边缘讨论
        └── 但注意：大模型训练完 → 蒸馏/量化 → 才能上边缘
```

---

## 6. 判断模型能否在某硬件上跑的通用的三步法

### Step 1: 显存/SRAM 检查（一票否决）

```
模型权重大小 + 峰值激活大小 + KV Cache(LLM) + 运行时开销 < 可用内存

示例：MobileNetV2-1.0x 在 STM32F746 (320KB SRAM):
  权重: 3.5M × 1 byte (INT8) = 3.5 MB → 超过 320KB → 直接否决
  必须用 MCUNet 替代（专为 <256KB 设计的架构）
```

### Step 2: 带宽检查（决定实际延迟）

```
延迟下限 = (权重大小 + 中间激活) / 内存带宽

示例：ResNet50 INT8 (12.5 MB) 在 Orin Nano (68 GB/s):
  延迟下限 = 12.5 MB / 68 GB/s = 0.18 ms（纯搬运）
  实际延迟 ≈ 2–5 ms（加上计算和调度）
  
→ 如果你需要 <2ms，Orin Nano 可能慢 2-3x
→ 必须用模型剪枝 + INT8 + TensorRT fusion
```

### Step 3: 算子兼容性检查（决定能否跑）

```
模型中的算子 → 目标推理引擎是否支持？

常见陷阱：
  - PyTorch 的 interpolate (双线性插值) → ONNX 版本敏感
  - LayerNorm → TFLite Micro 不支持（需用 RMSNorm 或手写 kernel）
  - MultiHeadAttention → 需要 ONNX opset ≥ 14
  - GELU → 有些老 NPU 不支持，需替换为 ReLU/SiLU
  - GroupNorm → TFLite 不原生支持
  - grid_sample → 绝大多数边缘推理引擎不支持

解决：用 ONNX Runtime 的检查工具（onnx.checker.check_model）+ 目标平台测试
```

---

## 7. 常见选型错误与真实事故

**事故 1**：团队买了 Jetson Orin NX 部署 13B LLM，结果 1 token/s

> 原因：13B FP16 需要 26GB 显存，Orin NX 只有 16GB → 用 swap（系统内存），速度掉 100 倍。
> 教训：**先算显存，再买硬件。AWQ INT4 之后 13B→~7GB，才能勉强跑。**

**事故 2**：用 MobileNetV3-large 在 Coral TPU 上，结果比 CPU 还慢

> 原因：MobileNetV3 有大量的 squeeze-and-excitation (SE) 模块 + hard-swish 激活。Coral TPU 对 SE 块的 `GlobalAveragePool → FC → ReLU → FC → sigmoid` 路径没有优化，每次都回退到 CPU 执行。
> 教训：**看标称 TOPS 之前，先查目标加速器对模型算子的白名单支持列表。**

**事故 3**：在 ESP32-S3 上跑 MobileNetV2-0.35x 推理一张图要 8 秒

> 原因：ESP32-S3 的 FPU 很弱，但团队没有做 INT8 量化（TFLite 默认 FP32）。
> 教训：**MCU 必须 INT8，FP32 推理慢 10–50 倍。**

**事故 4**：Stable Diffusion 在 Orin AGX 上 "能跑" 但 OOM 后 swap 导致系统卡死

> 原因：SDXL UNet (~2.6B FP16 = 5.2GB) + VAE + CLIP = 合计 >10GB。Orin AGX 64GB 看起来够了，但默认共享显存配置只给 GPU 分配了 16GB。
> 教训：**Jetson 的显存是共享的，需要用 `sudo jetson_clocks` + 调整 `/sys/devices` 中的 carveout。**

---

## 8. 理想部署流程模板

```
1. 分析需求
   ├── 目标延迟: P50 < Xms, P99 < Yms
   ├── 功耗上限: Z Watts
   └── 准确率底线: Acc/mAP > W%

2. 初步选型
   ├── 计算模型的最小内存需求 (INT8/INT4)
   ├── 查带宽算出延迟下限
   └── 确认算子白名单

3. 原型验证
   ├── ONNX 导出 → onnx.checker 验证
   ├── 在目标硬件上跑 ONNX Runtime / TFLite / TRT
   ├── 测量实际延迟、内存、功耗
   └── 如果不过 → 回到 Step 2 换板卡或 Step 4 压缩

4. 模型压缩迭代
   ├── 结构化剪枝 → 通道数减少 → 延迟线性下降
   ├── INT8 量化 → 带宽需求减半 → 延迟~减半
   ├── 算子融合 → Conv+BN+ReLU 合并 → 减少内存往返
   ├── 如果还不够 → NAS 搜一个天生小的架构
   └── 如果还不够 → 换更贵的板卡

5. 生产部署
   ├── CI/CD 中自动化 ONNX 验证 + 精度回归测试
   ├── 监控 P99 延迟，设 SLO
   ├── 灰度发布 (1% → 10% → 100%)
   └── 准备回滚方案（FP32 模型在更强硬件上的备选）
```

---

## 9. 推荐工具链速查

| 目标平台          | 推荐工具链                               | 关键命令                                          |
| ----------------- | ---------------------------------------- | ------------------------------------------------- |
| 通用 ONNX         | onnx + onnxruntime                       | `onnx.checker.check_model(model)`                 |
| NVIDIA GPU        | TensorRT + Polygraphy                     | `trtexec --onnx=model.onnx --int8 --saveEngine`   |
| NVIDIA Jetson     | Jetson-specific TensorRT + jetson-stats  | `sudo jetson_clocks && sudo tegrastats`           |
| ARM CPU (手机)    | TFLite / QNN / ONNX Runtime              | `benchmark_model --graph=model.tflite`            |
| Apple (iPhone)    | CoreML + coremltools                     | `coremltools.convert(model, source='pytorch')`    |
| Qualcomm (骁龙)   | SNPE / QNN                               | `snpe-onnx-to-dlc`                                |
| MCU (ARM Cortex-M)| TFLite Micro / TinyEngine                | `xxd -i model.tflite > model.cc` (嵌入C数组)      |
| FPGA              | FINN (Xilinx) / hls4ml                   | ONNX → FINN compiler → bitstream                  |
| 国产 NPU          | 各厂商 SDK (昇腾 CANN / 寒武纪 / 瑞芯微) | 参考厂商文档                                      |
| 通用 CPU 基准测试 | onnxruntime benchmark                     | `onnxruntime_perf_test -m model.onnx -I`          |

---

## 10. 核心公式速记

```
实际延迟 ≈ max(计算时间, 搬运时间) + 调度开销

计算时间 = FLOPs / (标称TOPS × batch_利用率 × 算子_效率)

搬运时间 = (权重大小 + 激活大小) / 内存带宽

内存需求 = 权重(FP16: 2×params | INT8: 1×params | INT4: 0.5×params)
          + 峰值激活（取决于输入分辨率和模型架构）
          + KV Cache（LLM 专用: 2 × layers × hidden_dim × seq_len × 2 bytes）
          + 运行时框架开销（~100-500MB for GPU, ~1-10KB for MCU）
```

---

> **本附录的核心信息**:
>
> 1. **标称 TOPS 是市场营销数字，看实际利用率**——GPU batch=1 时利用率暴跌至 5-15%
> 2. **内存带宽决定延迟下限**——在边缘设备上，搬运数据的时间 >> 计算时间
> 3. **选硬件 = 先算内存需求，再看算子兼容性**——内存不够直接一票否决
> 4. **MCU 必须 INT8，手机必须 NPU 引擎，LLM 必须 INT4**——硬件特性决定压缩方案
> 5. **同一块板卡，量化+剪枝后的模型 vs 裸模型，速度差 3-10 倍不等**
