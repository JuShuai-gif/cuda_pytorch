# TinyEngine 项目分析：如何阅读与使用

**一句话定位**：TinyEngine 是 MIT HAN Lab 开源的 MCU 端深度学习推理/训练引擎。它由一个 **Python 编译前端**（将 TFLite/JSON 模型编译为 C 代码）和一个 **C 运行时后端**（手写 SIMD 优化的 ARM 内核）两部分组成，实现了从模型到 STM32 MCU 的端到端部署。

---

## 1. 项目架构总览

```
tinyengine/
│
├── code_generator/               ← 【编译前端】Python → 生成 C 代码
│   ├── CodegenUtilTFlite.py      ← 推理入口：GenerateSourceFilesFromTFlite()
│   ├── TfliteConvertor.py        ← TFLite FlatBuffer 解析 → IR
│   ├── TTEParser.py              ← 训练路径：TVM JSON 图解析 → IR (+ 融合规则)
│   ├── GeneralMemoryScheduler.py ← 核心：跨层 tensor 生命周期 + first-fit 分配
│   ├── CodeGenerator.py          ← IR → C 源码 (genModel.h / genModel.c)
│   ├── OpGenerator.py            ← 为每个 unique conv 配置生成专用 C 内核
│   ├── FusionUtil.py             ← 算子融合模式检测
│   ├── PatchBasedUtil.py         ← Patch 推理 (MCUNetV2)
│   ├── InputResizer.py           ← 输入尺寸重算 (patch 推理)
│   ├── GraphReorder.py           ← 训练图重排序
│   ├── QAS_util.py               ← 量化感知缩放工具
│   ├── operators/                ← 每种算子的 IR 定义 + C 代码生成 (conv2d/dwconv/add/...)
│   └── codetemplate/             ← 深度卷积专用 C 内核模板
├── TinyEngine/                   ← 【运行时后端】C 库
│   ├── include/                  ← 头文件 (tinyengine_function.h 等)
│   └── src/kernels/
│       ├── int_forward_op/       ← INT8 推理内核 (conv/fc/pool/...)
│       ├── fp_requantize_op/     ← 浮点重量化推理内核
│       └── fp_backward_op/       ← 训练反向传播内核 (conv/loss/...)
├── examples/                     ← 使用示例 (vww.py, tiny_training.py, ...)
├── tutorial/                     ← STM32CubeIDE 项目模板 (inference/, training/)
└── mcunet/                       ← 子模块：MCUNet 模型库
```

**数据流**：
```
TFLite 模型 (推理) 或 JSON 图 (训练)
  → TfliteConvertor / TTEParser 解析为 IR (list[basicOperator])
  → GeneralMemoryScheduler 分配 SRAM 地址
  → CodeGenerator 生成 C 源码 (genModel.h / genModel.c)
  → OpGenerator 生成专用深度卷积内核
  → STM32CubeIDE 编译 → .elf → 烧录到 MCU
```

---

## 2. 如何使用

### 环境准备

```bash
git clone --recursive https://github.com/mit-han-lab/tinyengine.git
cd tinyengine
pip install -r requirements.txt    # numpy, torch, flatbuffers, tqdm, matplotlib, torchvision
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```

硬件：STM32F746G-DISCO 探索板 (ARM Cortex-M7)。无需 GPU。

### 2.1 推理路径 (推荐入门)

```bash
# 第1步：运行代码生成（下载 TFLite 模型 → 解析 → 内存调度 → 生成 C 代码）
python examples/vww.py
# 生成 ./codegen/Include/genModel.h 和 ./codegen/Source/genModel.c

# 第2步：将生成的代码 + TinyEngine 运行时合并到 STM32CubeIDE 模板
cp -r ./tutorial/inference ./tutorial/TinyEngine_vww_tutorial
mkdir -p ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine
mv codegen ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/
cp -r ./TinyEngine/include ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/
cp -r ./TinyEngine/src     ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/
bash import_arm_inference.sh   # 复制 CMSIS-NN 头文件

# 第3步：STM32CubeIDE 1.5.0 → 导入项目 → 优化级别 -Ofast → 编译 → 烧录
```

板子上 LCD 会显示人物检测结果和 FPS。

### 2.2 Patch 推理路径 (MCUNetV2，更低内存)

```bash
python examples/vww_patchbased.py   # n_patches=2, split_index=5
# 其余步骤同上
```

### 2.3 训练路径 (Tiny Training Engine)

输入格式不同，需要 TVM Relay JSON 图 + pickle 权重文件：

```bash
python examples/tiny_training.py \
    -f assets/49kb-int8-graph.json \        # JSON 计算图
    -D assets/full-int8-params.pkl \        # pickle 权重
    -QAS assets/scale.json \                # 量化感知缩放
    -m -g -d -FR                            # 内存分析 / 代码生成 / 详细输出 / fp 重量化

# 部署到 tutorial/training
cp -r ./tutorial/training ./tutorial/TinyEngine_vww_training_tutorial
# ... 同上，用 import_arm_training.sh
```

---

## 3. 从哪开始阅读源码

### 入门层 (1小时)

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `examples/vww.py` | **端到端入口**。只有 ~30 行，`GenerateSourceFilesFromTFlite(tflite_path)` 一行调用理解完整流程 |
| 2 | `code_generator/operators/basic_utils.py` | **IR 基础**。`basicOperator` 类和 `tensor` 类，所有算子的基类 |
| 3 | `code_generator/operators/conv2d.py` | **基准算子**。理解一个算子如何定义自己的参数、生成 C 调用 |

### 核心层 (3小时)

| 顺序 | 文件 | 说明 |
|------|------|------|
| 4 | `code_generator/CodegenUtilTFlite.py` | **编排入口**。TFLite models 的 `GenerateSourceFilesFromTFlite()` 串联全部步骤 |
| 5 | `code_generator/TfliteConvertor.py` | **解析器**。TFLite FlatBuffer → IR 列表，每个算子映射为 `basicOperator` 子类 |
| 6 | `code_generator/GeneralMemoryScheduler.py` | **核心内存调度**。inplace dwconv、first-fit 分配、tensor 生命周期分析。这是 TinyEngine 的精髓 |
| 7 | `code_generator/CodeGenerator.py` | **代码生成**。IR → genModel.h/genModel.c。缓冲区声明、权重数组、invoke() 函数 |

### 深入层 (可选，3小时)

| 顺序 | 文件 | 说明 |
|------|------|------|
| 8 | `code_generator/TTEParser.py` | **训练路径**。TVM JSON 图 → IR + 14 种融合规则 (cast/tile/where/SGD 等) |
| 9 | `code_generator/codetemplate/depthwiseTemplate.py` | **深度卷积模板**。751 行生成高度优化的 SIMD 深度卷积 C 内核 |
| 10 | `TinyEngine/src/kernels/int_forward_op/convolve_1x1_s8.c` | **手写 SIMD 内核**。看 CMSIS-NN 内部函数 `__SMLAD` / `__PKHBT` 怎么拼出 1 周期 4 MAC |

---

## 4. 核心设计要点（理解这些才算读懂）

### 4.1 内存规划：为什么能在 < 256KB 里跑

1. **Inplace 深度卷积**：dwconv 的输出直接写在输入地址上。因为 CHW 数据流是从 channel-first 遍历，每个 pixel 在被覆盖前就已经被用完了。
2. **First-Fit 分配器**：跨层 tensor 生命周期分析 → 时间不重叠的 tensor 共享同一块 SRAM。按大小降序排列，大 tensor 优先分配。
3. **统一缓冲区**：整个 SRAM 就是一整块 `static char buffer[PEAK_MEM]`。激活在起始 (`buffer0`)，im2col 临时缓冲区紧随其后 (`sbuf`)，深度卷积内核缓冲区再随后 (`kbuf`)。零 malloc。
4. **Patch 推理**：将前面几层拆成多个空间 patch 循环执行，显著降低中间特征图的内存峰值。

### 4.2 算子融合：减内存 + 减开销

`TTEParser.py` 中定义了 14 种融合规则：
- **PAD → CONV**：不生成独立 PAD 层，直接把 padding 信息传给 conv
- **TILE → RESHAPE → CONV**：融合为直接以 tile 输入作权重
- **CAST + TransposeConv**：消除显式类型转换
- **WHERE + zero**：输出写到输入地址 (inplace)
- **SGD 更新**：transpose → abs → max → divide → cast 融合为隐式量化

### 4.3 代码生成：专用的高

生成策略与传统的模板化代码生成不同：
- 为每个 unique 卷积配置（kernel size, stride, channel count）生成**专属 C 文件**
- 深度卷积有 CHW 和 CWH 两种数据流（CWH 用于 kernel_h > kernel_w 的情况）
- 权重以 uint8 十六进制编译到 const 数组中
- 生成的 invoke() 是展开的调用序列，无循环

### 4.4 SIMD 内核：榨干 M7

运行时代码使用 ARMv7E-M DSP 指令：
- `__SMLAD`：一次双 16-bit SIMD 乘加
- `__SXTB16` / `__PKHBT`：符号扩展 + 打包
- 逐点卷积：一次处理 4 行 × 4 列（通道），最大化寄存器利用率
- 各算子有 `ch4`/`ch8`/`ch16`/`ch24` 等不同通道数专用变体

---

## 5. 关键设计决策速览

| 模块 | 决策 | 原因 |
|------|------|------|
| IR 设计 | `basicOperator` 统一接口 | 推理/训练共享同一套 IR |
| 内存 | First-Fit + inplace dwconv | MCU SRAM 极度受限 |
| 融合 | 编译期图模式匹配 | MCU 不能承受冗余算子 |
| 代码生成 | 为每个配置生成专用文件 | 循环展开、寄存器分配最优 |
| 内核 | CHW 数据流用于 dwconv | 支持 inplace 原地计算 |
| 量化 | INT8 输入/权重，INT32 累加 | MCU 无 FPU（或有 FPU 但不高效） |

---

## 6. 快速决策：什么时候看什么

- **我想看完整调用链** → `examples/vww.py` → `CodegenUtilTFlite.py` → `TfliteConvertor.py` → `GeneralMemoryScheduler.py` → `CodeGenerator.py`
- **我想理解内存怎么省到 < 256KB** → `GeneralMemoryScheduler.py`（inplace + first-fit）→ `PatchBasedUtil.py`（patch 推理）
- **我想理解算子融合** → `TTEParser.py` 的 `_findBinMaskPattern` / `_castisFusable` / `_findTileRepAsWeights`
- **我想看 SIMD 优化** → `int_forward_op/convolve_1x1_s8*.c` → `depthwiseTemplate.py`
- **我想跑一次训练 demo** → `examples/tiny_training.py` → `tutorial/training/README.md`
