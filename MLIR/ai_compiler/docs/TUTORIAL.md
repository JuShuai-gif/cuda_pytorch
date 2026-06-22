# 从零读懂这个 AI 编译器（零基础教程）

> 这份文档假设你**完全没学过编译原理**。我们用大量生活化比喻，从"编译器是什么"讲起，
> 一步步带你理解这个项目在做什么，并且每个概念都配一条**你可以亲手敲**的命令。
>
> 读完你会明白：神经网络模型是怎么一步步被"编译"成能在手机/机器人上快速运行的程序的。

---

## 目录

- [第 0 章 先建立直觉：编译器到底是什么](#第-0-章-先建立直觉编译器到底是什么)
- [第 1 章 为什么需要"AI 编译器"](#第-1-章-为什么需要ai-编译器)
- [第 2 章 核心思想：层层翻译（lowering）](#第-2-章-核心思想层层翻译lowering)
- [第 3 章 关键名词，一个一个讲清楚](#第-3-章-关键名词一个一个讲清楚)
- [第 4 章 动手跑一遍（最重要）](#第-4-章-动手跑一遍最重要)
- [第 5 章 名词速查表](#第-5-章-名词速查表)
- [第 6 章 接下来读什么](#第-6-章-接下来读什么)

---

## 第 0 章 先建立直觉：编译器到底是什么

**编译器 = 翻译官。**

你写的代码（人能看懂的"人话"）和电脑芯片能执行的指令（"机器话"）完全是两种语言。
编译器就是中间那个翻译官，把人话翻译成机器话。

- 你写 C 语言：`a + b` —— 编译器（如 GCC）把它翻译成 CPU 的加法指令。
- 你训练一个神经网络：`卷积 → 批归一化 → ReLU` —— **AI 编译器**把它翻译成能在
  GPU / 手机芯片 / 机器人芯片上飞快运行的指令。

这个项目（`edge_ai_compiler_pro`）就是后者：**一个把神经网络翻译成高效程序的翻译官**。

> 类比：你有一篇中文文章（神经网络），要给一个只懂方言的工人（芯片）看。
> 翻译官不仅要翻译，还要**翻得又快又省**——这就是编译器优化。

---

## 第 1 章 为什么需要"AI 编译器"

你可能会问：PyTorch 不是已经能跑模型了吗？为什么还要编译？

因为**"能跑"和"跑得好"是两回事**。

| 场景         | 要求                                         |
| ------------ | -------------------------------------------- |
| 训练（服务器） | 算得对就行，有大显卡、不缺电、不缺内存       |
| 部署（手机/机器人） | 要**快**（实时）、**省内存**（芯片内存很小）、**稳**（不能卡顿） |

举个机器人的例子：机器人要"看一眼 → 决定动作 → 执行"，这个循环每秒要做 10～50 次。
也就是说，从看到画面到算出动作，**必须在 20～100 毫秒内完成**。原始 PyTorch 模型直接跑往往太慢、太占内存，根本塞不进机器人的小芯片。

AI 编译器要解决三件事：
1. **更快**：把多个算子合并、用更低精度计算。
2. **更省内存**：让不同时用到的数据复用同一块内存。
3. **更确定**：提前把内存都规划好，运行时不临时申请内存（避免卡顿）。

这个项目把这三件事都做了。

---

## 第 2 章 核心思想：层层翻译（lowering）

翻译官不会"一步到位"把中文直接翻成机器码，那太难、太容易错。

**正确做法是分很多层，每层只翻译一点点。** 就像：

```
中文（很抽象） → 普通话 → 简单句子 → 单词 → 拼音 → 机器能读的字节
```

每一步都简单、可检查。编译器把这个过程叫 **lowering（逐级下降）**。
"下降"是指：从"人类好理解的高层表达"一步步降到"机器能执行的低层表达"。

本项目的翻译流水线长这样（从上到下，越来越接近机器）：

```
  PyTorch 模型
      │
      ▼
  EdgeDialect            ← 高层：还看得出"这是卷积、这是注意力"（人类视角）
      │   ← 在这一层做优化：算子融合、量化、常量折叠
      ▼
  Linalg                 ← 中层：变成"带循环的数学运算"
      │   ← 把"张量"变成"内存里的格子"（bufferization）
      ▼
  MemRef + 循环(SCF)      ← 低层：变成"对内存的 for 循环读写"
      │
      ▼
  LLVM 方言 → 机器码      ← 最底层：芯片能直接执行
```

**关键点**：每一层叫一个"方言（dialect）"，就像翻译过程中的每种中间语言。
高层方言适合做"看得懂语义"的优化（比如"这俩算子能合并"），低层方言适合生成真正的机器指令。

---

## 第 3 章 关键名词，一个一个讲清楚

下面每个名词都配：**一句话比喻 + 在本项目里对应什么**。别怕，都是大白话。

### 3.1 IR（中间表示，Intermediate Representation）

- **比喻**：翻译官的"草稿纸"。源语言和目标语言之间的过渡写法。
- **本项目**：我们用 MLIR 这套现成的草稿纸系统（不自己造），在上面写神经网络。

### 3.2 Dialect（方言）

- **比喻**：草稿纸上不同层级的"词汇表"。高层方言的词是"卷积/注意力"，低层方言的词是"加载内存/循环"。
- **本项目**：我们造了一个高层方言 `EdgeDialect`，它的"词"是神经网络算子：
  `conv2d`(卷积)、`relu`(激活)、`matmul`(矩阵乘)、`attention`(注意力)……

### 3.3 Operation 和 Value（算子和它的结果）

- **比喻**：像 Excel。每个单元格写一个公式（Operation），算出一个值（Value）；
  下一个单元格可以引用上一个的结果。
- **本项目**：`%1 = edge.relu %0` 意思是"对 `%0` 做 ReLU，结果叫 `%1`"。

### 3.4 Pass（遍 / 一轮处理）

- **比喻**：对草稿纸做"一轮修改"。比如"通读一遍，把所有能合并的工序合并"。
- **本项目**：`edge-shape-inference`（推断尺寸）、`edge-fuse-conv-bn-relu`（融合）等都是 Pass。
  一个个 Pass 串起来，就是优化流水线。

### 3.5 Shape Inference（形状推断）

- **比喻**：填快递单时，知道盒子尺寸才能装箱。编译器要先算出每个张量的尺寸（形状），
  后面才能规划内存、生成循环。
- **本项目**：`edge-shape-inference` 能把"未知尺寸 `?`"推断成具体数字。
  例如卷积输入 `224×224`，步长 2，能算出输出是 `112×112`。

### 3.6 Fusion（算子融合）

- **比喻**：做菜时，"洗菜→切菜→炒菜"如果能一口锅连着做，就省了来回端盘子的功夫。
  把多个算子合成一个，省掉中间结果的"搬来搬去"（省内存带宽）和"开锅"（启动开销）。
- **本项目**：`edge-fuse-conv-bn-relu` 把"卷积 + 批归一化 + ReLU"三个算子合并成一个。
  而且它会用数学把"批归一化"直接折进卷积的权重里——结果完全一样，但少了两个算子。

### 3.7 Constant Folding（常量折叠）

- **比喻**：算式里 `3 × 4` 这种全是常数的部分，编译时就先算成 `12`，别等运行时再算。
- **本项目**：`edge.constant`（常量）能在编译期被提前计算掉。

### 3.8 Quantization（量化）

- **比喻**：用更粗的刻度尺。原来用"毫米尺"（FP32，很精确但占空间），改用"厘米尺"（INT8，
  精度差一点但只占 1/4 空间、算得快 2～4 倍）。关键是**怎么选刻度**才不损失太多精度。
- **本项目**：`edge-quantize` 实现了三种"选刻度"的方法（MinMax / 百分位 / KL 散度），
  并模拟量化后的误差。这正是 TensorRT、华为昇腾做的事。

### 3.9 Bufferization（缓冲化）

- **比喻**：从"数学公式"变成"真实仓库里的货架"。前面我们说"张量 = 数学上的一块数据"，
  这一步把它落实成"内存里具体哪一格"。
- **本项目**：流水线里的 `one-shot-bufferize` 步骤，把 `tensor`（数学的）变成 `memref`（真实内存的）。

### 3.10 Memory Planning（内存规划）

- **比喻**：商场停车场。不是每辆车（张量）都永久占一个车位，车走了车位就能给别人用。
  把"生命周期不重叠"的张量安排到同一块内存，省内存。
- **本项目**：`edge-memplan` 分析每个张量"什么时候生、什么时候死"，让它们复用内存。
  示例里能把峰值内存从 6144 字节降到 4608 字节（省 25%）。

### 3.11 Lowering（逐级下降）

- **比喻**：就是第 2 章说的"层层翻译"，每一步降低一点抽象度。
- **本项目**：`edge-lower-to-llvm` 一条命令把高层 `EdgeDialect` 一路降到 LLVM 方言（最接近机器码）。

### 3.12 Runtime（运行时）

- **比喻**：翻译完了，得有人真的照着译文去干活。运行时就是"照着计算图真正算出结果"的执行器。
- **本项目**：`edge-run` 会真的把模型算一遍，给出输出。

### 3.13 Profiler（性能分析器）

- **比喻**：秒表。看时间到底花在哪个算子上，好针对性优化。
- **本项目**：`edge-run` 自带 Profiler，告诉你每个算子花了多少毫秒、占比多少。

---

## 第 4 章 动手跑一遍（最重要）

光看不练假把式。这一章带你**真的跑一遍**，每条命令都给出**预期输出**和**解释**。

### 4.0 先把项目编译出来

```bash
# 进入项目目录
cd /home/ghr/code/cuda_pytorch/mlir/ai_compiler

# 配置 + 编译（第一次会久一点）
cmake -G Ninja -S . -B build -DMLIR_DIR=/home/ghr/code/llvm-project/install/lib/cmake/mlir
ninja -C build
```

编译完成后，工具都在 `build/bin/` 下。下面用一个最小的"两层 MLP"模型做例子，它在
`examples/end_to_end/mlp.mlir`：

```mlir
func.func @mlp(%x: tensor<8x16xf32>, %w1: tensor<16x32xf32>, %w2: tensor<32x8xf32>)
    -> tensor<8x8xf32> {
  %0 = edge.matmul %x, %w1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %1 = edge.relu %0 : tensor<8x32xf32>
  %2 = edge.matmul %1, %w2 : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  %3 = edge.relu %2 : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}
```

**怎么读这段？**
- `func.func @mlp(...)`：定义一个叫 `mlp` 的函数。
- `%x, %w1, %w2`：三个输入（`%` 开头的都是"值"，类似变量名）。
- `tensor<8x16xf32>`：一个 8 行 16 列的浮点数张量（`f32` = 32 位浮点）。
- 函数体：先做矩阵乘 → ReLU → 再矩阵乘 → ReLU，最后返回结果。

这就是一个最简单的神经网络（两个全连接层）。

---

### 4.1 看结构：`edge-introspect`

> 作用：把计算图的"骨架"打印出来，帮你看清它由哪些部件组成。

```bash
./build/bin/edge-introspect examples/end_to_end/mlp.mlir
```

**预期输出（节选）**：

```
===== IR Structure Tree =====
builtin.module
  Region (1 blocks):
    Block (args=0, ops=1):
      func.func  [operands=0, results=0, regions=1, attrs=2]
        Region (1 blocks):
          Block (args=3, ops=5):
            edge.matmul  [operands=2, results=1, ...]
            edge.relu    [operands=1, results=1, ...]
            edge.matmul  [operands=2, results=1, ...]
            edge.relu    [operands=1, results=1, ...]
            func.return  [...]

===== Op Statistics =====
  edge.matmul : 2
  edge.relu : 2
```

**解释**：它告诉你这个图里有 2 个矩阵乘、2 个 ReLU，以及它们的嵌套结构
（模块里有函数，函数里有一串算子）。这就是"看懂一张计算图"的第一步。

---

### 4.2 看优化：`edge-opt`

> 作用：对计算图做各种优化"遍（Pass）"。

**统计算子和计算量**：

```bash
./build/bin/edge-opt examples/end_to_end/mlp.mlir --edge-statistics
```

会打印一份报告：总共多少算子、估算多少次乘加运算（MAC，衡量计算量的指标）。

**做融合**（这个 MLP 没有 conv+bn+relu，所以不变；换个带卷积的模型就能看到合并）：

```bash
./build/bin/edge-opt examples/end_to_end/mlp.mlir --edge-fuse-conv-bn-relu
```

**串多个 Pass**（先推断形状，再融合）：

```bash
./build/bin/edge-opt examples/end_to_end/mlp.mlir --edge-shape-inference --edge-fuse-conv-bn-relu
```

> 这就是编译器优化的本质：**一串 Pass 依次改写计算图**，每个 Pass 做一件事。

---

### 4.3 看量化：`edge-quantize`

> 作用：演示后训练量化（PTQ），对比三种"选刻度"方法。

```bash
./build/bin/edge-quantize examples/end_to_end/mlp.mlir --edge-out=reports
```

**预期输出（节选）**：

```
| method            | threshold | scale    | full SQNR(dB) | body SQNR(dB) |
| MinMax            | 15.96     | 0.125683 | 29.62         | 28.86         |
| Percentile(99.9)  | 3.66      | 0.028848 | 10.37         | 41.52         |
| KL-divergence     | 15.97     | 0.125714 | 29.61         | 28.85         |
```

**怎么读？**
- `SQNR`（信噪比，dB）：越高表示量化后越接近原始、误差越小。
- `MinMax` 把刻度尺拉到最大值（被极端值带偏），主体精度低。
- `Percentile`（百分位）裁掉极端离群点，**主体精度高很多**（body SQNR 41.52 vs 28.86）。
- 这就是为什么 TensorRT 默认用更聪明的 KL/百分位校准，而不是简单的 MinMax。

> 一句话：量化 = 用精度换速度和内存；**怎么选刻度**决定了你损失多少精度。

---

### 4.4 看内存：`edge-memplan`

> 作用：分析每个张量的"生死时间"，让它们复用内存。

```bash
./build/bin/edge-memplan examples/end_to_end/mlp.mlir --edge-align=64
```

**预期输出（节选）**：

```
- Naive peak (no reuse): 6144 bytes      ← 每个张量各占一块，不复用
- Planned peak (reuse) : 4608 bytes      ← 复用后只需这么多
- Saving: 25%
```

**解释**：就像停车场复用车位。算出每个张量"什么时候不再被用"，把后面的张量安排到
前面已经空出来的内存上，从而降低峰值内存。

---

### 4.5 看翻译到底层：`edge-lower-to-llvm`

> 作用：把高层计算图一路翻译到 LLVM 方言（最接近机器码的一层）。

```bash
./build/bin/edge-opt examples/end_to_end/mlp.mlir --edge-lower-to-llvm | head -40
```

你会看到输出里出现大量 `llvm.func`、`llvm.add`、`llvm.getelementptr` 之类——
这就是"已经翻译成机器能直接执行的低层指令"了。中间它经历了：

```
edge.matmul/relu → linalg（带循环的数学） → memref（真实内存） → scf.for（循环） → llvm
```

这一整条就是第 2 章说的"层层翻译"，现在你亲眼看到了结果。

---

### 4.6 真正算一遍：`edge-run`

> 作用：把模型真的执行一遍，并测每个算子的耗时。

```bash
./build/bin/edge-run examples/end_to_end/mlp.mlir --edge-fill=1.0
```

`--edge-fill=1.0` 表示把所有输入填成 1.0（方便手算验证）。

**预期输出（节选）**：

```
# Edge Runtime Profiling Report
- Ops executed: 4
- Total latency: 0.0068 ms

| op          | latency(ms) | %     |
| edge.matmul | 0.0031      | 45.9  |
| edge.relu   | 0.0005      | 7.5   |
| edge.matmul | 0.0029      | 43.6  |
| edge.relu   | 0.0002      | 3.0   |

## Outputs
- elements=64, checksum=32768.0000
```

**怎么验证它算对了？**
全 1 输入：第一层 `8×16 · 16×32`，每个输出 = 16 个 1×1 相加 = 16；ReLU 不变；
第二层 `8×32 · 32×8`，每个输出 = 32 个 16×1 相加 = 512；ReLU 不变；
共 64 个元素，每个 512，求和 = `64 × 512 = 32768`。✅ 和输出的 checksum 完全一致。

Profiler 还告诉你：时间主要花在两个矩阵乘上（各占 ~45%）。**这就是性能优化的依据**——
知道时间花在哪，才知道优化哪。

---

### 4.7 一键端到端：`edge_compile.py`

> 作用：把上面所有步骤串成一条命令，并自动生成报告。

```bash
python3 scripts/edge_compile.py
```

它会依次跑：优化 → 统计 → 降到 LLVM → 内存规划 → 执行，并把结果写成四份报告到 `reports/`：

- `fusion_report.md`：优化前后算子数量对比
- `compilation_report.md`：是否成功翻译到 LLVM
- `memory_report.md`：峰值内存
- `latency_report.md`：每个算子的耗时

这就是真实编译器（如 NVIDIA 的 `trtexec`、华为的 `atc`）提供的"一键编译 + 体检报告"功能。

---

## 第 5 章 名词速查表

| 名词                | 一句话解释                                       | 本项目对应             |
| ------------------- | ------------------------------------------------ | ---------------------- |
| 编译器              | 把"人话代码"翻译成"机器指令"的翻译官             | 整个项目               |
| AI 编译器           | 把神经网络翻译成高效推理程序                     | 整个项目               |
| IR（中间表示）      | 翻译过程中的"草稿纸"                              | 用 MLIR                |
| Dialect（方言）     | 草稿纸上某一抽象层级的"词汇表"                    | `EdgeDialect`、Linalg、LLVM |
| Operation（算子）   | 一个运算（如卷积、加法）                          | `edge.conv2d` 等       |
| Value（值）         | 一个算子算出的结果，可被后面引用                  | `%0`、`%1`             |
| Tensor（张量）      | 一块多维数组数据（如 8×16 的矩阵）               | `tensor<8x16xf32>`     |
| Pass（遍）          | 对计算图做一轮改写/优化                          | `edge-fuse-conv-bn-relu` 等 |
| Shape Inference     | 推断每个张量的尺寸                                | `edge-shape-inference` |
| Fusion（融合）      | 把多个算子合成一个，省搬运和启动开销             | `edge-fuse-conv-bn-relu` |
| Constant Folding    | 编译期先把全是常数的部分算掉                      | `edge.constant` 折叠   |
| Quantization（量化）| 用更粗刻度省内存、提速（精度换速度）             | `edge-quantize`        |
| Calibration（校准） | 量化时"怎么选刻度尺"                              | MinMax/百分位/KL       |
| Bufferization       | 把数学张量落实成真实内存                          | `one-shot-bufferize`   |
| Memory Planning     | 让生命周期不重叠的张量复用内存                    | `edge-memplan`         |
| Lowering（下降）    | 一级一级把高层翻译到低层                          | `edge-lower-to-llvm`   |
| Runtime（运行时）   | 照着计算图真正算出结果的执行器                    | `edge-run`             |
| Profiler            | 测每个算子耗时的"秒表"                            | `edge-run` 内置        |

---

## 第 6 章 接下来读什么

当你理解了上面这些直觉，可以按这个顺序深入：

1. **[`../README.md`](../README.md)**：项目总览、状态、构建方式。
2. **[`../ARCHITECTURE.md`](../ARCHITECTURE.md)**：设计细节（为什么这样分层）。
3. **[`../notes/`](../notes/)**：17 篇深入笔记，每篇讲一个模块，并和 TensorRT / TVM /
   TPU-MLIR / 华为昇腾对比。建议顺序：
   - `01`（MLIR 基础）→ `02`（怎么定义算子）→ `03`（EdgeDialect）
   - `04`（Pass）→ `05`（图优化/融合）→ `06`（改写模式）
   - `07`（量化）→ `08`（缓冲化）→ `09`（内存规划）
   - `10`（lowering）→ `11/12`（运行时/分析）
   - `13/14/15`（厂商对比）→ `16`（端到端）→ `17`（机器人/VLA 部署）
4. **[`EdgeOps.md`](EdgeOps.md)**：每个算子的精确定义（自动生成的"字典"）。

> 学习建议：**先把第 4 章的命令全部亲手敲一遍**，看到真实输出后再回头读概念，
> 会比纯看理论快得多。编译器是"做出来才懂"的东西。
