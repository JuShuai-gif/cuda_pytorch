# MIT 6.5940 项目源码精读指南

7 个项目的整体关系：**从 TinyML（MCU 推理/训练）到 LLM 量化部署**，涵盖 MIT HAN Lab 在端侧深度学习领域的核心工作。

```
第一阶段: TinyML 基础 (MCU 优化)
tinyml/ ──→ ofa (弹性NAS) + mcunet (MCU专用NAS) + netaug (训练增强) + tinytl (迁移学习)

第二阶段: MCU 部署引擎
tinyengine/ ──→ TFLite→C 代码生成器 + MCU 推理/训练运行时
tiny-training/ ──→ 256KB 内存内 MCU 训练 (编译期自动微分+稀疏更新)

第三阶段: LLM 量化 (算法→系统)
smoothquant/ ──→ W8A8 训练无关量化 (Scale 迁移消除激活异常值)
llm-awq/     ──→ AWQ W4A16 量化 + TinyChat 端侧推理引擎
TinyChatEngine/ ──→ 纯 C++ 零依赖 LLM 推理引擎 (跨平台)
```

---

## 1. llm-awq (AWQ 量化 + TinyChat 推理)

**做了什么**：AWQ (Activation-aware Weight Quantization) — 4-bit 仅权重量化，在边缘设备部署 LLM。

### 项目结构

```
llm-awq/
├── awq/                    # 核心 AWQ 库
│   ├── entry.py            # 【主入口】量化+评测流水线
│   ├── quantize/
│   │   ├── pre_quant.py    # AWQ 流程: run_awq() → apply_awq()
│   │   ├── auto_scale.py   # 激活感知缩放搜索 (20个网格点找最优α)
│   │   ├── auto_clip.py    # MSE 剪切 (搜索最优per-channel max)
│   │   ├── quantizer.py    # 伪量化 / 真量化函数
│   │   ├── qmodule.py      # WQLinear (4-bit 打包权重)
│   │   └── smooth.py       # 视觉塔平滑 (SmoothQuant 的视觉版)
│   └── utils/
│       ├── calib_data.py   # 校准数据加载
│       └── module.py       # 模型子模块操作工具
├── tinychat/               # TinyChat 推理引擎
│   ├── demo.py             # 文本聊天 demo
│   ├── benchmark.py        # 延迟 benchmark
│   ├── models/             # 模型实现 (Llama/Falcon/MPT/Qwen2/LLaVA/InternVL3)
│   ├── modules/            # 融合量化模块 (attn/mlp/norm)
│   ├── stream_generators/  # 流式文本生成
│   └── serve/              # 分布式服务 (FastAPI+Gradio)
└── examples/
```

### 精读流程（从简到深）

```
1. 先看量化原理 (1h)
   awq/quantize/quantizer.py → pseudo_quantize_tensor()
   ─ 理解均匀量化的 scale/zero_point 计算和伪量化流程

2. AWQ 关键：预量化 (2h)
   awq/quantize/auto_scale.py → auto_scale_block() + _search_module_scale()
   ─ 理解激活感知缩放：在 α∈[0,1] 网格搜索最优缩放，使量化输出 MSE 最小
   awq/quantize/pre_quant.py → run_awq()
   ─ 理解完整流水线：标定→缩放→剪切→量化

3. AWQ 入口 (1h)
   awq/entry.py
   ─ 理解 CLI 参数和整体调用链

4. 推理引擎 (3h)
   tinychat/models/llama.py → 自定义 Llama (WQLinear + fused attn + flash attn)
   tinychat/modules/fused_attn.py → 融合量化注意力
   tinychat/modules/fused_mlp.py → 融合量化 MLP
   tinychat/demo.py → 完整推理 demo

5. 多模态扩展 (2h, 可选)
   tinychat/models/llava_llama.py → LLaVA 架构
   tinychat/models/nvila_qwen2.py → NVILA (VILA 2.0)
```

**核心知识点**：
- 激活离群值 → 大激活 x 放大权重量化误差
- 缩放 s_x^α → 网格搜索 α → MSE 最小化 → 融合到 LN/FC 中
- WQLinear：int4 权重打包 + per-group scales + GEMM/GEMV 分发

---

## 2. smoothquant (W8A8 量化)

**做了什么**：训练无关、精度无损的 LLM W8A8 量化。通过数学等价的 scale 变换消除激活异常值。

### 项目结构

```
smoothquant/
├── smoothquant/
│   ├── smooth.py      # 【核心算法】smooth_ln_fcs() — 激活→权重 scale 迁移
│   ├── fake_quant.py  # W8A8 假量化模拟 (per-channel weight + per-token activation)
│   ├── opt.py         # OPT 模型 INT8 真实推理 (torch-int/CUTLASS INT8 GEMM)
│   ├── calibration.py # 激活 scale 标定 (Pile 验证集)
│   └── ppl_eval.py    # 困惑度评测入口
└── examples/
    ├── generate_act_scales.py     # 生成并保存 activation scales
    ├── export_int8_model.py       # 导出 INT8 模型
    └── smoothquant_opt_demo.ipynb # OPT demo notebook
```

### 精读流程

```
1. 理解核心 (30min)
   smoothquant/smooth.py → smooth_ln_fcs()
   ─ 核心只需 20 行: LN.weight /= s, fc.weight *= s
   ─ 本质是 (X / s) * (W * s) = X * W

2. 假量化验证 (1h)
   smoothquant/fake_quant.py → W8A8Linear.forward()
   ─ 激活 per-token 量化 + 权重 per-channel 量化

3. 标定+评测 (1h)
   smoothquant/calibration.py → get_act_scales()
   smoothquant/ppl_eval.py → smooth+quantize → WikiText-2 PPL

4. 真 INT8 推理 (2h, 可选)
   smoothquant/opt.py → Int8OPTForCausalLM.from_float()
   ─ 需要 torch-int Cutlass INT8 GEMM kernel
```

**核心知识点**：
- 激活 outlier 只在少数通道，但每次前向都出现
- `(X / diag(s)) * (diag(s) * W)` — 把量化难度从激活平滑到权重
- α 参数控制迁移比例：0 = 不做平滑，1 = 全部迁移

---

## 3. TinyChatEngine (纯 C++ LLM 推理引擎)

**做了什么**：零依赖、纯 C++ 从零实现的跨平台 LLM/VLM 推理引擎。支持 CPU (x86/ARM) 和 GPU (CUDA/Metal)。

### 项目结构

```
TinyChatEngine/
├── llm/                          # 推理引擎主体
│   ├── application/chat.cc       # 【主入口】聊天程序
│   ├── include/
│   │   ├── model.h              # 模型抽象接口
│   │   ├── nn_modules/          # 模型模块 (按精度 Fp32/Int4/Int8)
│   │   └── ops/                 # 基础算子 (Linear/Attention/LayerNorm/...)
│   ├── src/                     # C++ 实现
│   │   ├── nn_modules/          # ~30 个模块实现
│   │   └── ops/                 # ~15 个算子实现
│   └── tools/                   # Python 离线工具
│       ├── model_quantizer.py   # AWQ 量化
│       ├── llama_exporter.py    # HF → TinyChatEngine 格式转换
│       └── download_model.py    # 模型下载
├── kernels/                      # 各平台高性能 matmul (AVX2/NEON/CUDA/Metal)
└── scripts/                      # Shell 启动脚本
```

### 精读流程

```
1. 构建系统 + 启动 (30min)
   llm/Makefile → 理解后端选择 (x86/ARM/CUDA/Metal)
   llm/scripts/chat.sh → 看启动流程

2. 核心模型 (2h)
   llm/include/model.h → 模型加载接口
   llm/src/nn_modules/LLaMA_int4.cc → INT4 LLaMA 推理实现
   llm/src/ops/Linear_int4.cc → INT4 矩阵乘法调用

3. 高性能 kernels (2h)
   kernels/matmul.h → 统一 matmul 接口
   kernels/neon/matmul_neon_int4.cc → ARM NEON INT4 SIMD 优化
   kernels/cuda/matmul_int4.cu → CUDA INT4 kernel

4. 导出工具 (1h, 可选)
   llm/tools/llama_exporter.py → HF 权重转 TinyChatEngine 二进制格式
```

**核心知识点**：
- 零依赖：不依赖 PyTorch/TF，纯 C++ + SIMD
- INT4 matmul：`qweight` 打包为 int16，解包+乘加 SIMD 并行
- Fused Attention：QKV 合并矩阵乘法 + RoPE + Flash Attention

---

## 4. mcunet (MCU 神经架构搜索)

**做了什么**：系统-算法协同设计，在 MCU (<256KB SRAM) 上运行深度学习。TinyNAS 搜索网络 + TinyEngine 推理。

### 项目结构

```
mcunet/
├── mcunet/
│   ├── model_zoo.py           # 【核心 API】build_model() + download_tflite()
│   ├── tinynas/               # TinyNAS: MCU 专用的架构搜索
│   │   ├── nn/                # ProxylessNAS/MobileNetV2 网络结构
│   │   ├── elastic_nn/        # 弹性网络 (OFA 变体，适配 MCU)
│   │   └── tf_codebase/       # TF → TFLite 转换
│   └── utils/                 # BN 融合、FLOPs 统计
├── eval_torch.py              # 【入口】PyTorch 精度验证
├── eval_tflite.py             # 【入口】TFLite 精度验证
└── eval_det.py                # 【入口】人物检测 demo
```

### 精读流程

```
1. API 试用 (30min)
   mcunet/model_zoo.py → build_model("mcunet-in3") 看如何加载预训练模型

2. 网络结构 (1h)
   mcunet/tinynas/nn/networks/proxyless_nets.py → ProxylessNASNet
   mcunet/tinynas/nn/modules/layers.py → MBConv/DWConv

3. 弹性搜索 (2h)
   mcunet/tinynas/elastic_nn/modules/dynamic_layers.py → 可变宽度/深度
   mcunet/tinynas/elastic_nn/networks/ofa_proxyless.py → OFA 超级网络

4. TFLite 转换 (1h, 可选)
   mcunet/tinynas/tf_codebase/generate_tflite.py → PyTorch→TF→TFLite 量化
```

**核心知识点**：
- MCU 约束：Flash < 2MB, SRAM < 512KB
- TinyNAS：在给定内存/延迟约束下搜索最优网络
- MBConv → depthwise separable conv → 移动端标准算子

---

## 5. tinyengine (MCU 代码生成器)

**做了什么**：TFLite 模型 → C 代码编译器 + MCU 推理/训练运行时。配合 mcunet 使用。

### 项目结构

```
tinyengine/
├── code_generator/              # Python 编译前端
│   ├── TfliteConvertor.py       # TFLite FlatBuffer → IR
│   ├── CodeGenerator.py         # IR → C 源码
│   ├── GeneralMemoryScheduler.py # 跨层生命周期内存分配
│   ├── FusionUtil.py            # 算子融合检测
│   ├── OpGenerator.py           # 生成专用卷积核
│   └── operators/               # 每个算子的代码生成 (~30个)
├── TinyEngine/                  # C 运行时后端
│   ├── include/                 # 头文件
│   └── src/kernels/
│       ├── int_forward_op/      # 量化推理核 (Conv/FC/Pool/...)
│       ├── fp_requantize_op/    # 混合精度推理核
│       └── fp_backward_op/      # 训练反向传播核
└── examples/                    # 使用示例
    ├── vww.py                   # Visual Wake Words 推理
    └── tiny_training.py         # 训练代码生成
```

### 精读流程

```
1. 端到端流程 (1h)
   examples/vww.py → GenerateSourceFilesFromTFlite()
   ─ 理解 TFLite → 解析 → 内存调度 → C 代码 的完整流水线

2. 编译前端 (2h)
   code_generator/TfliteConvertor.py → TFLite 解析为 IR
   code_generator/GeneralMemoryScheduler.py → first-fit 内存分配
   code_generator/CodeGenerator.py → IR→C 代码生成

3. C 运行时 (2h)
   TinyEngine/src/kernels/int_forward_op/ → ARM CMSIS-NN 风格的手写 SIMD 核
```

**核心知识点**：
- 内存规划：跨层张量生命周期分析 → first-fit 分配 → 最小化峰值 SRAM
- 算子融合：conv+bn+scale, conv+depthwise 合并减少内存
- Patch-based 推理：输出分块计算降低峰值 SRAM

---

## 6. tiny-training (256KB 内存 MCU 训练)

**做了什么**：在 <256KB SRAM 的 MCU 上进行深度学习训练。核心是编译期自动微分 + 稀疏更新。

### 项目结构

```
tiny-training/
├── algorithm/                   # GPU 端算法模拟
│   ├── train_cls.py             # 【主入口】QAS (量化感知缩放) 训练
│   ├── quantize/
│   │   ├── quantized_ops.py     # 量化前向 (模拟 MCU INT8)
│   │   └── quantized_ops_diff.py # 量化+反向传播
│   ├── core/
│   │   ├── optimizer/sgd_scale.py   # QAS SGD (自动梯度缩放)
│   │   └── utils/partial_backward.py # 稀疏更新 (只更新关键层)
│   └── configs/                 # YAML 配置
├── compilation/                 # 编译系统
│   ├── mcu_ir_gen.py            # 【编译入口】PyTorch→TVM IR→带反向图的 IR
│   ├── ir2json.py               # IR → JSON 导出
│   └── autodiff/auto_diff.py    # 编译期自动微分核心
└── assets/mcu_models/           # 预训练 MCUNet 模型
```

### 精读流程

```
1. 模拟训练 (2h)
   algorithm/train_cls.py → QAS + 稀疏更新训练循环
   algorithm/quantize/quantized_ops_diff.py → INT8 前向+反向
   algorithm/core/optimizer/sgd_scale.py → 自动缩放梯度补偿量化误差

2. 编译期自动微分 (2h)
   compilation/autodiff/auto_diff.py → 在 TVM Relay IR 上生成反向图
   compilation/autodiff/diff_ops.py → 前向算子的梯度算子映射表
   compilation/mcu_ir_gen.py → 完整编译流水线

3. 稀疏更新 (1h)
   algorithm/core/utils/partial_backward.py → 贡献分析选择更新哪些层
```

**核心知识点**：
- 为什么 MCU 训练难？反向传播需要存储所有激活，内存巨大
- QAS：量化训练中的梯度缩放，使量化感知训练更稳定
- 稀疏更新：只更新偏置和关键层的权重，不更新全部 -> 大幅减少反向图剪枝后的内存
- 编译期自动微分 (compile-time autodiff)：在编译器级别生成反向图，避免运行时记录计算图

---

## 7. tinyml (元仓库：OFA + NetAug + TinyTL)

**做了什么**：MIT HAN Lab TinyML 系列的元仓库，包含多个独立子项目。

### 项目结构

```
tinyml/
├── once-for-all/               # OFA (ICLR 2020): 训练一次，到处部署
│   ├── train_ofa_net.py        # 训练弹性超级网络
│   ├── eval_ofa_net.py         # 评估任意子网络
│   ├── ofa/
│   │   ├── imagenet_classification/
│   │   │   ├── elastic_nn/     # DynamicConv → 可变宽度/深度/核大小
│   │   │   └── networks/       # MobileNetV3/ResNet/ProxylessNAS
│   │   └── nas/                # 进化搜索 + 精度预测器
│   └── tutorial/               # Jupyter 教程
├── mcunet/                     # mcunet 子模块 (同 #4)
├── netaug/                     # NetAug (ICLR 2022): 微型模型训练增强
│   ├── train.py                # 用更宽/更深的辅助分支训练
│   ├── models/base/            # MobileNetV2/V3, MCUNet, ProxylessNAS
│   └── models/netaug/          # 增强后的模型 (训练时加子网络，推理时丢弃)
└── tinytl/                     # TinyTL (NeurIPS 2020): 微型迁移学习
    ├── tinytl_fgvc_train.py    # FGVC 细粒度分类训练
    ├── tinytl/model/
    │   └── modules.py          # LiteResidualModule (轻量残差)
    └── tinytl/utils/           # 推理时内存分析

```

### 精读流程

```
1. OFA — 弹性网络基础 (3h)
   once-for-all/ofa/imagenet_classification/elastic_nn/networks/
   ─ OFAMobileNetV3: 理解一个网络如何包含多种子网
   once-for-all/train_ofa_net.py → 渐进式收缩训练策略

2. NetAug — 训练技巧 (1h)
   netaug/train.py → 辅助分支增强→推理时丢弃
   ─ 核心洞察: 小模型需要更强的监督，用大分支提供

3. TinyTL — 迁移学习 (1h)
   tinytl/tinytl_fgvc_train.py → 冻结 backbone，只训练轻量残差
   ─ 核心洞察: 激活 (不是参数) 是训练内存的主要瓶颈
```

---

## 阅读顺序

### 1. SmoothQuant (1天) — 入门首选

核心代码只有 ~30 行，理解 LLM 量化的基础思想。

```
1. smoothquant/smooth.py            ← 【第1步】读 smooth_ln_fcs()，只有20行
2. smoothquant/calibration.py       ← 【第2步】读 get_act_scales()，看激活 scale 怎么来的
3. smoothquant/fake_quant.py        ← 【第3步】读 W8A8Linear.forward()，看假量化的 forward 流程
4. smoothquant/ppl_eval.py          ← 【第4步】读评测入口，理解 smooth → quantize → evaluate 调用链
5. examples/smoothquant_opt_demo.ipynb  ← 【第5步】跑 notebook，实战感受
```

### 2. llm-awq (3天) — AWQ 量化核心

这是 Lab4 的主角，需要细读。

```
  第一天: 量化基础
1. awq/quantize/quantizer.py        ← 读 pseudo_quantize_tensor()
                                       理解 uniform quantization: scale/zero_point/q_group 含义
2. awq/quantize/quantizer.py        ← 读 real_quantize_model_weight()
                                       看 WQLinear 是怎么替换原始 Linear 的
3. awq/quantize/qmodule.py          ← 读 WQLinear 类，看 int4 权重怎么打包存储

  第二天: AWQ 核心算法
4. awq/quantize/auto_scale.py       ← 读 auto_scale_block() + _search_module_scale()
                                       理解: s_x^α 搜索 → MSE 最小 → 最优 α → LN/FC 融合
5. awq/quantize/auto_clip.py        ← 读 auto_clip_layer()
                                       理解: 搜索最优 per-channel max，MSE 最小化
6. awq/quantize/pre_quant.py        ← 读 run_awq() 和 apply_awq()
                                       理解完整流程: 标定 → 缩放搜索 → 剪切搜索 → 应用

  第三天: 完整流水线 + 推理
7. awq/entry.py                     ← 读 main()，理解 CLI 参数和整体调用链
8. awq/utils/calib_data.py          ← 读 get_calib_dataset()，看校准数据怎么来的
9. tinychat/models/llama.py         ← 读 LlamaForCausalLM，看 WQLinear 怎么用在推理中
10. tinychat/demo.py                ← 跑一个完整推理 demo，看端到端流程
```

### 3. TinyChatEngine (2天) — C++ 推理引擎

纯 C++ 从零实现，需要 C++ 基础。

```
  第一天: 模型层
1. llm/include/model.h              ← 理解模型抽象接口 (load/sample/generate)
2. llm/src/nn_modules/LLaMA_int4.cc ← 读 INT4 LLaMA 推理实现
3. llm/include/ops/Linear.h         ← 理解 Linear 算子接口
4. llm/src/ops/Linear_int4.cc       ← 看 INT4 矩阵乘法的 dispatch 逻辑
5. llm/application/chat.cc          ← 看聊天主循环: tokenize → generate → detokenize

  第二天: 高性能内核 (选读一个平台)
6. kernels/matmul.h                 ← 统一 matmul 接口定义
7. kernels/cuda/matmul_int4.cu      ← (NVIDIA GPU) CUDA INT4 matmul
  或 kernels/neon/matmul_neon_int4.cc ← (ARM) NEON INT4 matmul
  或 kernels/avx/matmul_avx_int4.cc   ← (x86) AVX2 INT4 matmul
8. llm/tools/llama_exporter.py      ← 看 HuggingFace 权重怎么转成引擎能读的格式
```

### 4. tinyml — OFA (1天)

OFA 是整个 TinyML 系列的基石概念。

```
1. once-for-all/ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py
                                     ← 核心: DynamicMBConvLayer，理解"弹性"怎么做到
2. once-for-all/ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py
                                     ← OFAMobileNetV3，理解一个网络如何包含子网
3. once-for-all/train_ofa_net.py     ← 理解渐进式收缩训练 (full → width → depth → kernel)
4. once-for-all/ofa/nas/search_algorithm/evolution.py
                                     ← 进化搜索: 从超级网络中找最优子网
5. once-for-all/eval_ofa_net.py      ← 评测入口，理解怎么指定子网配置
```

### 5. mcunet (1天)

在 OFA 基础上适配 MCU 约束。

```
1. mcunet/model_zoo.py              ← build_model() + download_tflite()，API 入口
2. mcunet/mcunet/tinynas/nn/networks/proxyless_nets.py
                                     ← ProxylessNASNet，MCUNet 的主干架构
3. mcunet/mcunet/tinynas/nn/modules/layers.py
                                     ← MBConv 模块，理解深度可分离卷积
4. mcunet/mcunet/tinynas/elastic_nn/ ← 看 MCU 适配版弹性网络 (vs OFA 的区别)
5. eval_torch.py                     ← 跑一次精度验证，理解完整评测流程
```

### 6. tinyengine (1天)

模型 → C 代码编译器。

```
1. examples/vww.py                  ← 端到端示例: GenerateSourceFilesFromTFlite()
2. code_generator/TfliteConvertor.py ← TFLite FlatBuffer 解析为 IR
3. code_generator/GeneralMemoryScheduler.py
                                     ← first-fit 内存分配，理解跨层生命周期
4. code_generator/CodeGenerator.py  ← IR → C 源码生成
5. TinyEngine/src/kernels/int_forward_op/
                                     ← 选一个算子 (如 convolve_1x1) 看手写 SIMD 优化
```

### 7. tiny-training (1天)

MCU 端训练。

```
1. algorithm/train_cls.py           ← 训练入口，理解 QAS + 稀疏更新循环
2. algorithm/quantize/quantized_ops_diff.py
                                     ← INT8 前向 + 反向传播实现
3. algorithm/core/optimizer/sgd_scale.py
                                     ← QAS SGD: 自动梯度缩放
4. algorithm/core/utils/partial_backward.py
                                     ← 稀疏更新: 只更新关键层
5. compilation/autodiff/auto_diff.py ← 编译期自动微分核心
```

---

## 关键技术对比

| 技术 | 精度 | 硬件 | 核心思想 |
|------|------|------|----------|
| SmoothQuant | W8A8 | GPU | Scale 迁移消除激活异常值 |
| AWQ | W4A16 | GPU/CPU | 重要通道权重放大→量化→缩小 |
| TinyChatEngine | INT4/FP16 | x86/ARM/CUDA/Metal | 零依赖纯 C++ 推理引擎 |
| MCUNet | INT8 | MCU (STM32) | TinyNAS + TinyEngine 协同设计 |
| TinyEngine | INT8/FP | MCU | TFLite→C 代码生成器 |
| Tiny Training | INT8 | MCU | 编译期自动微分 + 稀疏更新 |
| OFA | FP32 | Mobile/Server | 一个超级网络适配所有硬件 |
