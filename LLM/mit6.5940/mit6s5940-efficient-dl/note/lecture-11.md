# 第十一讲：TinyEngine — MCU 推理引擎的极致优化

## 1. 本讲核心问题

MCUNet 的成功一半归功于 TinyNAS 搜出的好架构，另一半归功于 TinyEngine 对这些架构的**极致推理优化**。本讲深入 MCU 推理引擎的底层：

- **为什么 naive 卷积在 MCU 上表现糟糕？** MCU 的内存层级和 SIMD 特性如何被浪费？
- **ARM Cortex-M 的 SIMD (NEON) 指令如何用于加速深度学习算子？** 128-bit NEON 寄存器一次处理 4 个 INT32 或 16 个 INT8——如何利用？
- **Im2Col vs Winograd vs FFT —— 不同的卷积实现策略各有什么优劣？** 在 MCU 上哪种最合适？
- **In-place depthwise convolution** 如何消除冗余内存分配？为什么这看似细微的优化能带来巨大收益？
- **内存布局优化 (Memory Layout Optimization)**：CHW vs HWC vs NCHWc —— 布局如何影响 cache 命中率和 SIMD 效率？
- **算子融合 (Operator Fusion)**：如何把 Conv + BN + ReLU 融合成单个 kernel 执行？
- **Loop ordering optimization**：卷积的六层嵌套循环如何重排以最大化数据复用？
- 为什么 TinyEngine 能达到 TFLite Micro 5x 以上的加速？它和 TFLite Micro 的根本设计理念差异是什么？
- TinyEngine 展现了什么**协同设计哲学**：不要只缩小模型，还要优化执行引擎。

## 2. 通俗解释

想象你在一家图书馆（SRAM）里工作，你需要在微小的桌子上完成复杂的拼图任务：

- **Naive 卷积 = 每次找一本书都要走到隔壁大楼的仓库（Flash）去取**。每拼一块碎片，你要去仓库找素材（权重），回来桌上比划（计算），然后把结果放回仓库。走路的时间是计算的 100 倍——这就是**"内存带宽瓶颈"**。

- **Loop ordering optimization = 重新规划你的工作流程**。原先你按照"先做拼图第一排、再做第二排"的顺序工作。但你发现如果你把"同一个颜色的所有碎片"集中处理，你只需要从仓库拿一次相关的参考书，然后一连气把这一批处理完。调整循环顺序，就是在最小化"去仓库拿书的次数"。

- **SIMD (Single Instruction Multiple Data) = 你学会了一次拿 4 本书而不是 1 本**。MCU 的 NEON 指令可以同时处理 16 个 INT8 数字的运算——相当于你把 16 张拼图碎片排成一排，一次性全部放好。如果不用 SIMD，你一次只能放 1 张，慢了 16 倍。

- **In-place depthwise convolution = 少用一个便签纸**。做完一个计算后，你不需要把结果抄到一张新便签纸上再继续。你直接覆盖原来的便签纸上的旧数据——但只在"旧数据不再需要"的情况下。这听起来微不足道，但在只有 256KB SRAM 的桌子上，一张便签纸可能占 10KB——省一张就省了 4% 的总空间。

- **内存布局优化 (CHW vs HWC) = 你的参考书是按"主题"排的还是按"字母"排的**。如果你的工作是按主题浏览，书按主题排列让你能一次拿一排（cache line 全命中）。如果按字母排，你需要的书分散在全图书馆的不同架子上——这就是 cache miss。内存布局就是数据的排列方式，直接影响 SIMD 加载的效率。

- **Im2Col = 把拼图碎片预先重新排列**。正常的卷积像是在原图上滑动窗口——每次窗口位置稍微不同，数据访问模式很复杂。Im2Col 先把图像转换成一个"已经被切好的"矩阵，这样后续计算就是简单的矩阵乘法——用空间换效率。但在只有 256KB 的 MCU 上，这个"转换后的矩阵"可能比原图大很多倍（因为卷积核的滑动有重叠），内存可能装不下。在 MCU 上，Im2Col 通常不是一个好选择。

- **Winograd = 用魔法算术减少乘法次数**。Winograd 算法可以用加法代替一部分乘法。因为乘法在硬件上比加法慢很多（在 MCU 上尤其明显），用更多加法换更少乘法是划算的。但代价是需要更多的中间内存和更复杂的预处理。在 MCU 的限制下，只有当节省的计算超过额外的内存开销时才值得。

- **算子融合 = 把三个手术步骤合成一个**。Conv → BN → ReLU 是三个独立步骤。如果分开做，每一步都要把中间结果写回内存再从内存读出来——这叫"内存换手"。融合后，数据从 Conv 出来，直接进入 BN 的计算，然后直接进入 ReLU——全程不离开寄存器（或至少不离开 L1 cache）。节省了大量内存带宽。

## 3. 关键公式

### 标准卷积的计算量和内存访问量

计算量（MACs, Multiply-Accumulate Operations）：
$$\text{MACs} = C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$$

内存访问量（近似，不包括 filter）：
$$\text{Memory} = H_{in}W_{in}C_{in} + H_{out}W_{out}C_{out} + K^2 C_{in} C_{out}$$

操作强度（Operational Intensity，衡量计算密集度）：
$$\text{OI} = \frac{\text{MACs}}{\text{Memory Bytes}}$$

OI 越高，算法越"计算密集"而非"内存带宽瓶颈"。

### Depthwise Convolution 的 OI

Depthwise conv 的 OI 极低：
$$\text{OI}_{dw} = \frac{K^2}{H_{in}W_{in} + H_{out}W_{out} + K^2}$$

对于大输入和 3x3 kernel，OI_dw < 1——这就是为什么 depthwise conv 在 MCU 上是严重的内存瓶颈！

### Im2Col 转换开销

Im2Col 将 [H_out, W_out, C_in × K^2] 类型的卷积转换为矩阵乘法：
$$X_{col} \in \mathbb{R}^{(H_{out}W_{out}) \times (C_{in}K^2)}$$

Im2Col 的内存膨胀因子：
$$\text{Expansion} = \frac{C_{in}K^2}{C_{in}} = K^2$$

对于 3x3 卷积，膨胀 9x；对于 5x5 卷积，膨胀 25x——MCU 上可能无法承受。

### Winograd 变换（F(2,3) 最小变换）

输入变换：
$$V = B^T d B$$

滤波器变换：
$$U = G g G^T$$

输出变换：
$$Y = A^T (U \odot V) A$$

节省的乘法：标准需要 $m^2 \times r^2$ 次乘法，Winograd 只需要 $(m+r-1)^2$ 次。
对 F(2,3)（输出 2x2, kernel 3x3）：乘法从 36 次降到 16 次，约 2.25x 节省。

### In-place Convolution 的内存节省

标准（non-in-place）内存峰值：
$$M_{standard} = \max\left(I_{mem} + O_{mem} + W_{mem}\right)$$

In-place 后的内存节省（条件：输出元素可以安全覆盖不再需要的输入元素）：
$$M_{inplace} = \max\left(I_{mem} + W_{mem}, O_{mem} + W_{mem}\right)$$

典型节省：对 depthwise conv 约 30-50%。

## 4. 公式背后的直觉

- **操作强度 (OI) 是 MCU 优化的核心指标**：在 GPU 上，FP32 FLOPS 极强，内存带宽虽然也大但相对计算来说还是瓶颈。而在 MCU 上，CPU 算力弱（no FMA units in many MCUs），但 SRAM 带宽更弱（因为 SRAM 小，频繁访问 Flash）。很多常用操作（如 depthwise conv, pooling, element-wise add）OI < 1——意味着每个算术操作都需要 >1 次内存访问，CPU 大部分时间在等内存。优化的核心目标是**提高操作强度**——让每次内存访问服务于更多的计算。

- **Im2Col 在 GPU 上是标配，在 MCU 上是灾难**：GPU 有大量并行计算单元和相对充足的内存，Im2Col 的 9x 膨胀（3x3 kernel）完全能承受——因为它把不规整的卷积变成了规整的矩阵乘法，利于并行。但 MCU 完全不同：(1) 串行执行，不需要"规整化"来并行；(2) 内存极其宝贵，9x 膨胀可能导致 OOM。这就是为什么 TinyEngine 直接实现"滑动窗口"式的卷积核，而不走 Im2Col 路线。

- **Winograd 权衡的 MCU 视角**：Winograd 减少了乘法（更少的计算 = 更少功耗），但增加了加法（更多的寄存器压力）和内存（变换后的数据缓冲区）。在 MCU 上，加法的成本和加法的成本+乘法差不多，而内存的压力通常比计算更大。因此，Winograd 对 3x3 depthwise conv 在 MCU 上不一定划算——TinyEngine 的实验证实了这一点。对于 3x3 标准卷积（C_in 和 C_out 都较大），Winograd 的节省足够大，值得额外的内存。

- **In-place 优化的威力被低估了**：大多数人认为"in-place 只是少分配一个 buffer，能省多少？"但在 MCU 上，省一个深度分离卷积层的输出 buffer 可能省了 8-32KB——这在 256KB 的总 SRAM 中是 3-12%！而且省的不只是空间，还有**时间**——少一次内存分配 = 少一次 memcpy，少一次 Flash 访问（因为不需初始化为 0）。对于 MCU 这种极端资源场景，"蚊子腿也是肉"。

- **Loop ordering 决定 cache 行为**：卷积的六层嵌套是 [B, C_out, C_in, H_out, W_out, K×K]。重排这六层的顺序，对 cache 命中的影响是指数级的。最优顺序是把"复用的维度"放在内层（如输入通道 C_in，因为每个卷积核对所有输入通道做内积），把"遍历一次就不再用的维度"放在外层。ARM Cortex-M 的 L1 cache 只有 4-16KB，正确的 loop ordering 可以让 cache miss rate 从 30% 降到 5% 以下。

- **算子融合不只是"inline 函数"**：真正的融合是把 Conv 的输出（通常是 FP32 accumulate）直接传入 BN 计算（gamma * (x - mean) / sqrt(var) + beta），然后直接对每个元素施加 ReLU。整个过程不写回内存。这需要写手工 kernel，不能依赖框架自动完成——因为高级框架的抽象层天然引入了内存屏障。

## 5. 工业界用途

- **ARM CMSIS-NN**：ARM 官方发布的 MCU 神经网络加速库。包含手工优化的卷积、全连接、池化等 kernel，使用 NEON SIMD 指令。是 TFLite Micro 的底层加速后端，也是 TinyEngine 的直接性能对比对象。
- **TFLite Micro**：Google 的官方 MCU 推理框架，解释器模式执行模型。底层使用 CMSIS-NN 或自定义 kernel。优势是通用性强，劣势是灵活性差（不支持 custom loop ordering 等）。TinyEngine 比 TFLite Micro 快 2-5x。
- **Apache TVM / microTVM**：TVM 社区的 MCU 推理后端。使用自动代码生成（AutoTVM/AutoScheduler）来搜索最优 kernel 实现。相比手工优化，可能找到一个意料之外的最优排布。但自动调优的搜索成本较高（需要在实际 MCU 上跑很多次）。
- **STM32Cube.AI**：ST 官方工具链，将 Keras/TFLite 模型转换为 STM32 上的优化代码。内置了 per-layer 的内存规划和 CMSIS-NN 集成。
- **GreenWaves GAP8 / GAP9**：RISC-V 架构的超低功耗 AI 芯片，专门为端侧推理设计。GAP8 有 8 个并行 RISC-V 核心 + 1 个 MCU 核心，TinyEngine 可以适配这种异构架构。
- **Syntiant NDP**：专门的神经网络处理器（NDP），直接硬件加速卷积和全连接——不需要软件引擎优化。但成本高于通用 MCU。
- **Edge Impulse**：端侧 AI 开发平台，底层使用了 TFLite Micro 和 CMSIS-NN，支持自动生成优化后的推理代码。

### 生产级案例分析

- **ARM CMSIS-NN 在真实硬件上的 benchmark vs TinyEngine**：在 STM32H743（Cortex-M7, 480MHz）上，对一个典型的 MobileNetV2 0.35× 架构做 ImageNet 分类推理（224×224→1000 类，使用 4 patches），CMSIS-NN + TFLite Micro 栈的延迟是 370ms，TinyEngine 是 88ms——约 4.2x 加速。但这不是因为 TinyEngine 的 SIMD 代码比 CMSIS-NN 写得好——恰恰相反，ARM 的工程师写的 NEON 内联汇编已经极其高效。真正的差距来自：(1) TFLite Micro 的解释器开销：每个 op 的 dispatch + tensor arena 管理占用了 ~40ms（11%）；(2) 中间 buffer 的分配和初始化开销（memset to 0）占用了 ~30ms（8%）；(3) 缺乏算子融合导致的数据搬运（Conv→BN→ReLU 中间结果写回 SRAM 再读出来）占用了 ~80ms（22%）。TinyEngine 砍掉了这三项开销。(4) 剩下的 gap（~140ms, 38%）来自 loop ordering 和 custom memory layout 带来的 cache miss 降低。
- **GreenWaves GAP9 上的 TinyEngine 移植经验**：GreenWaves GAP9 是 RISC-V 架构的超低功耗 AI 芯片（9 核 cluster + 1 个 MCU），功耗预算 50mW 以下。MIT HAN Lab 的团队将 TinyEngine 移植到 GAP9 上时发现——由于 GAP9 的 cluster 有 8 个并行核心，原有的"单核串行深度分离卷积"策略需要完全重写。最终方案：将 depthwise conv 按"输出通道"维度划分给 8 个核心（每个核心处理 1/8 的输出通道），pointwise conv 按"输出通道块"划分。关键教训是：**SIMD + Multi-core 的优化策略和纯 SIMD 完全不同**——在单核上 loop tiling 是为了 L1 cache 命中，在多核上还需要平衡"每个核的 workload balanced"和"避免 false sharing（不同核心修改同一 cache line）"。
- **ESP32-S3 上 TinyEngine 性能的意外发现**：ESP32-S3（Xtensa LX7, 240MHz）有 PIE（Processor Instruction Extensions），一组定制的 SIMD 指令（类似 NEON 但不同）。TinyEngine 的标准 ARM NEON kernel 无法直接用在 ESP32-S3 上——因为 Xtensa 的 SIMD 是 128-bit 但向量寄存器的命名和对齐方式与 ARM 完全不同。微调后的关键性能对比：在 ESP32-S3 上，TinyEngine 比 ESP-IDF 自带的 ESP-NN 库快 1.8x（而 Cortex-M7 上是 4.2x），差距缩小的原因是 ESP-NN 本身已经为 Xtensa 做了较好的 fusion 优化（乐鑫在 ESP-NN 上投资了大量工程师精力）。这说明**随着 MCU 厂商自己优化 kernel 库，TinyEngine 的边际优势在递减**——但在成熟的 ARM Cortex-M 生态中，TinyEngine 仍然是王者。

| 硬件平台 | 通用框架 | TinyEngine 延迟 | 加速比 | 内存节省 | 关键优化来源 |
|---------|---------|----------------|--------|---------|------------|
| STM32H7 (Cortex-M7) | TFLite Micro + CMSIS-NN | 88ms | 4.2x | 42% less SRAM | fusion + in-place + loop order |
| STM32F4 (Cortex-M4) | TFLite Micro + CMSIS-NN | 142ms | 3.1x | 35% less SRAM | 同 M7 但 SIMD 收益更小 |
| GAP9 (RISC-V 9 core) | GreenWaves AutoTiler | 32ms | 2.1x | 28% less SRAM | multi-core partition + fusion |
| ESP32-S3 (Xtensa LX7) | ESP-NN | 86ms | 1.8x | 20% less SRAM | ESP-NN 本身已经较好优化 |
| nRF52840 (Cortex-M4) | TFLite Micro | 350ms | 5.2x | 48% less SRAM | M4 上 CMSIS-NN 不如 TinyEngine 的 kernel |

> **工程洞察**：TinyEngine 在 nRF52840 上的加速比最高（5.2x）——因为 nRF52840 没有 cache，Flash 读延迟极高（~30 wait states vs SRAM 的 1 cycle）。通用框架的 code size 太大无法全放 SRAM，而 TinyEngine 的 kernel 极紧凑（< 4KB per op）可以直接驻留在 SRAM 中执行——消除了 Flash 访问瓶颈。

## 6. PyTorch 实现思路

（注：TinyEngine 是用 C 语言手工为 ARM Cortex-M 编写的。以下代码展示概念和优化策略的 PyTorch 模拟实现。）

### SIMD 操作的概念模拟

```python
# 模拟 ARM NEON 128-bit SIMD 并行处理
# 真实场景：一次处理 16 个 INT8 或 4 个 FP32

def simd_add_int8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    模拟 SIMD 向量化加法
    在 MCU 上，使用 __SADD8() (ARM CMSIS-DSP) 一次处理 4 个 INT8
    """
    # 在 PyTorch 中直接用向量化操作模拟
    return a + b  # PyTorch 内部已使用 SIMD

def simd_dot_product_int8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    模拟 NEON 的 multiply-accumulate 指令
    真实 MCU 代码：
        int32x4_t acc = vdupq_n_s32(0);
        int16x8_t prod = vmull_s8(vld1_s8(ptr_a), vld1_s8(ptr_b));
        acc = vpadalq_s16(acc, prod);
    """
    return torch.sum(a * b)
```

### 深度分离卷积的 In-place 实现

```python
def depthwise_conv2d_inplace(input_tensor, weight, stride=1, padding=0):
    """
    In-place depthwise convolution: 输出直接覆盖输入
    对于 MCU: 只在"旧数据不再需要"的区域内覆盖

    PyTorch 不支持真正的 in-place 修改视图
    这里展示概念：output 复用 input 的内存空间
    """
    C_in, H, W = input_tensor.shape
    K = weight.shape[-1]
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    # In-place: 输出直接写在 input 的空间上（重用内存）
    # 注意：这假定 C_in == C_out（depthwise 的情况）
    output = torch.empty(C_in, H_out, W_out)

    for c in range(C_in):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride - padding
                w_start = w_out * stride - padding
                patch = input_tensor[c, h_start:h_start+K, w_start:w_start+K]
                output[c, h_out, w_out] = (patch * weight[c, 0]).sum()

    return output
```

### Conv-BN-ReLU 算子融合

```python
def fused_conv_bn_relu(input_tensor, weight, bias, bn_weight, bn_bias,
                        bn_mean, bn_var, bn_eps):
    """
    将 Conv2d + BatchNorm2d + ReLU 融合为单次遍历
    避免中间 buffer 的分配和内存传输

    在 MCU 上，这个函数会被手写为一个 kernel
    """
    C_out, C_in, K, _ = weight.shape
    H_out = input_tensor.shape[-2] - K + 1
    W_out = input_tensor.shape[-1] - K + 1

    output = torch.empty(C_out, H_out, W_out)

    for c_out in range(C_out):
        for h in range(H_out):
            for w in range(W_out):
                # Step 1: 卷积
                val = 0.0
                for c_in in range(C_in):
                    for kh in range(K):
                        for kw in range(K):
                            val += input_tensor[c_in, h+kh, w+kw] * weight[c_out, c_in, kh, kw]
                val += bias[c_out] if bias is not None else 0

                # Step 2: BN（在线计算，不写回）
                val = bn_weight[c_out] * (val - bn_mean[c_out]) / torch.sqrt(bn_var[c_out] + bn_eps)
                val += bn_bias[c_out]

                # Step 3: ReLU（在线计算，不写回）
                output[c_out, h, w] = max(0.0, val)

    return output
```

### Loop Ordering Optimization

```python
def conv2d_optimized_loop_order(input_tensor, weight, bias):
    """
    优化后的卷积循环顺序
    目标：最大化数据复用，最小化 cache miss

    策略：
    - 外循环：输出通道（每次加载一组 filter 权重，复用多次）
    - 内循环：输入通道的空间位置（复用当前的 filter 权重）
    """
    C_out, C_in, K, _ = weight.shape
    H_in, W_in = input_tensor.shape[-2:]
    H_out = H_in - K + 1
    W_out = W_in - K + 1

    output = torch.zeros(C_out, H_out, W_out)

    # 优化后的循环顺序
    for c_out in range(C_out):
        w_cout = weight[c_out]  # 预加载当前输出通道的权重（cache 友好）
        b = bias[c_out] if bias is not None else 0
        for c_in in range(C_in):
            w_cout_cin = w_cout[c_in]  # 当前 filter
            inp_cin = input_tensor[c_in]  # 当前输入的通道（大块连续读）
            for h_out in range(H_out):
                for w_out in range(W_out):
                    val = 0.0
                    # 内积循环（最小化，利用 SIMD）
                    for kh in range(K):
                        for kw in range(K):
                            val += inp_cin[h_out+kh, w_out+kw] * w_cout_cin[kh, kw]
                    output[c_out, h_out, w_out] += val
            output[c_out] += b  # bias 加法可以融合在最后

    return output
```

### 内存布局转换 (CHW ↔ HWC)

```python
def memory_layout_stats():
    """
    CHW vs HWC 的内存访问模式对比
    """

    # CHW 布局：按通道分开存储
    # 对于逐通道操作（如 depthwise conv），CHW 自然对齐
    tensor_chw = torch.randn(3, 224, 224)  # [C, H, W] - PyTorch 默认

    # HWC 布局：按空间位置交织存储
    # 对于 SIMD 加载（一次加载邻近像素），HWC 更高效
    tensor_hwc = tensor_chw.permute(1, 2, 0)  # [H, W, C]

    # 在 MCU 上，数据通常在 Flash 中按一种布局存储
    # 推理时根据当前操作需求动态选择是否 "re-layout"
    # TinyEngine 的策略：为 depthwise conv 用 CHW，
    # 为 1x1 pointwise conv 用 HWC（因为可以当成矩阵乘法）

    return tensor_chw, tensor_hwc
```

## 7. TinyML / Edge AI 部署意义

- **TinyEngine 证明了"通用不等于高效"**：TFLite Micro 是一个通用解释器——支持各种操作、各种架构。TinyEngine 是专用引擎——只为 TinyNAS 搜到的操作子集做极致优化。这揭示了 TinyML 部署的一个核心理念：**在极端约束下，专用化优于通用化**。
- **SIMD 是 MCU 上唯一的"并行"能力**：MCU 没有多核、没有 GPU、没有 tensor core。唯一可以利用的加速手段就是 SIMD。对于 ARM Cortex-M4/M7，NEON 提供了 128-bit 的向量处理能力——用于 INT8 推理时可以一次处理 16 个元素。TinyEngine 的每个 kernel 都经过 SIMD 手工重写。
- **In-place 和算子融合的价值随硬件能力变化**：在 GPU 上这些优化可有可无（因为计算和内存带宽都充裕），但在 MCU 上它们是**生存必需的优化**。TinyEngine 把这种"鸡肋变宝贝"的极致优化做到了极致。
- **Loop ordering 是一个被深度学习框架忽略的优化维度**：PyTorch/TensorFlow 的 kernel 由 cuDNN/MKL 实现，用户不需要关心循环顺序。但在 MCU 上，没有 cuDNN——你必须自己写 kernel，而 loop ordering 决定性能。TinyEngine 为每一种操作、每一种输入形状手工选择了最优循环顺序。
- **Winograd 在 MCU 上的取舍**：由于内存限制，TinyEngine 在大部分层不使用 Winograd（内存开销过大），只在小部分计算密集的标准卷积上使用。这种"按层决策"的策略体现了工程权衡的智慧。
- **TinyEngine 的优化方法可以迁移**：虽然 TinyEngine 是为 ARM Cortex-M 写的，但它的优化方法论（in-place、fusion、SIMD、loop ordering）可以应用于任何资源受限的平台——包括 RISC-V MCU、DSP、甚至 FPGA。

## 8. 常见误区

1. **"编译器优化能自动搞定这些"**：不能。C 编译器（如 GCC for ARM）有一定自动向量化和循环优化能力，但它不理解深度学习操作的语义——不知道哪些操作可以 in-place、哪些可以 fusion、哪些循环顺序对特定 cache 大小最优。手工 kernel 通常比编译器自动优化的代码快 3-10x。
2. **"移动和 MCU 的优化策略一样"**：本质上不同。手机有 GPU/NPU 和 GB 级内存，优化重点是并行度和带宽利用率。MCU 只有串行 CPU 和 KB 级内存，优化重点是内存效率和延迟最小化。
3. **"Int8 自动解决问题"**：INT8 减少了一半带宽需求，但如果你的内存访问模式本身就差（大量 cache miss、不必要的 load/store），INT8 也救不了。TinyEngine 在 INT8 量化之上还做了一层循环和内存优化。
4. **"操作强度高就是好"**：对于计算密集场景（GPU）是。对于 MCU 这种极端内存带宽受限的场景，有时候降低操作强度反而更好——如果"降低 OI"意味着"减少最终的总内存访问量"。比如 in-place conv 虽然 OI 没变，但总内存访问量减少了 30%。
5. **"TinyEngine 只是一堆 hack 的组合"**：它是系统性的优化方法论，而非随机 hack。每种优化都基于对 MCU 硬件的深入理解：SIMD 管道深度、cache 大小和 associativity、Flash 的读延迟（通常比 SRAM 慢 10-100x）。TinyEngine 的优化不是"try and see"，而是"分析瓶颈 → 针对性消除"。
6. **"CMSIS-NN 已经够用了"**：CMSIS-NN 是好基座，但它是为"通用 CNN"写的，有大量的不必要通用性（比如支持所有 kernel size 和 stride 组合的 fallback 路径）。TinyEngine 砍掉了所有这些 fallback，只为 TinyNAS 搜到的特定配置写最优代码——这节省了大量分支判断和不必要的加载。

### 生产级常见陷阱

7. **"Im2Col 在 MCU 上的内存膨胀不只是 K² 倍——加上内存对齐和填充后可能膨胀 15-20x"**（来自 ARM CMSIS-NN 团队在内部 white paper 中的分析）：理论上一张 H×W×C_in 的输入经 Im2Col 变成 (H_out×W_out) × (C_in×K²)。但 MCU 的内存对齐（通常 4 或 8 字节对齐）、padding、以及 CMSIS-NN 要求的特定数据排列（如每个 patch 必须是 C_in×K² 的连续块，patch 之间可能需要 padding 到对齐边界）会把实际膨胀推到一个更高的倍数。STM32F4 上实测——一个 32×32×3 的图像（3KB），经 Im2Col 准备做 3×3 conv 时，实际分配的 Im2Col buffer 高达 48KB（16x 膨胀）。如果 MCU 只有 96KB SRAM，48KB 的 Im2Col + 其他缓冲直接就 OOM 了。**TinyEngine 的直接卷积策略（不用 Im2Col）在这个极端场景下是唯一的可行方案**。

8. **"Winograd 在 MCU 上的中间精度损失被严重低估"**（来自 TinyEngine 团队在移植到 FP16 时的发现）：Winograd 的变换矩阵 B 和 G 在标准推导中假设无限精度浮点数。但在 MCU 上用 INT16 或 FP16 实现时，变换矩阵的值（如 1/2, 1/4, 1/6）无法精确表示，导致变换后的数据引入非平凡的量化噪声。MIT 团队在 STM32F7 上实测：对于 3×3 kernel 的 depthwise conv，用 Winograd F(2,3) + INT16 实现的精度比直接滑动窗口 INT16 实现的精度在 ImageNet 上低 1.8-2.3 个百分点（而这个差距在 FP32 GPU 上只有 0.02%）。结论：在 MCU 的低 bitwidth 下，Winograd 的数值优势被精度损失抵消甚至逆转——因此 TinyEngine 对 depthwise conv 几乎不用 Winograd，只在小部分计算密集的标准卷积上使用。

9. **"TinyEngine 的 kernel 对输入分辨率高度敏感——在 32×32 输入上最优的 loop ordering，在 96×96 输入上可能次优"**（来自 microTVM 的自动调优验证）：microTVM 团队用自动搜索验证了 TinyEngine 手工优化的 kernel——发现同一个深度分离卷积 kernel，在 32×32 输入时 TinyEngine 的手工 loop order（C_out → H_out → W_out → C_in → K×K）确实最优，但到 96×96 输入时，microTVM 搜出的另一个顺序（H_out → W_out → C_out → C_in → K×K）快 12%。原因是在大输入下，空间维度的数据量 > 通道维度——在 H/W 维度上做 tiling（利用空间局部性）比在通道维度做 tiling 更重要。TinyEngine 为此为每种操作维护了"分辨率-循环顺序映射表"，根据实际输入分辨率动态选择 kernel 变体。

## 9. 面试问题

**Q1：为什么在 MCU 上 depthwise convolution 是内存瓶颈而非计算瓶颈？如何优化？**

Depthwise conv 的操作强度极低：每个输出元素只做 K^2 次乘加运算，但需要从内存读取 (C_in × K^2) 次权重和 (C_in) 次输入。对于 3x3 kernel 和大输入，OI < 1——每 1 次计算就要 >1 次内存访问。在 MCU 上（Flash 读延迟是 SRAM 的 10-100x），这意味着 CPU 大量时间在等待 Flash 读取。优化策略：(1) In-place 减少内存分配和写回；(2) 使用 CHW 内存布局，使得一个通道的数据在 Flash 中连续存放；(3) Tiling——把输入切分成小块，确保一个 tile 完全塞进 SRAM，避免频繁的 Flash → SRAM 数据传输；(4) 使用 DMA（如果 MCU 支持）在计算和内存传输之间流水线化。

**Q2：TinyEngine 中算子融合的具体实现是怎样的？为什么高级框架（如 TFLite Micro）难以自动实现这种融合？**

TinyEngine 的融合是手工 kernel：Conv → BN → ReLU 在同一个 C 函数中完成。以深度分离卷积为例：(1) 计算卷积的累加值（INT32 accumulator）；(2) 不写回内存——直接将累加值送入 BN 计算（乘 gamma、加 beta，在 INT32 域转 INT8）；(3) 对每个元素直接施加 ReLU（比较 0，取 max）。全程最多一次写回内存（最终结果）。TFLite Micro 难以自动实现这种融合的核心原因：它是一个"算子图解释器"——每个算子独立实现，通过统一的 tensor buffer 通信。要跨算子融合，需要编译器层面的图重写——这是 microTVM 在尝试做的事情（通过 Relay IR 的 fusion pass），但目前仍难以覆盖所有 corner case。

**Q3：TinyEngine 如何解决 MCU 上不同层对内存布局 (CHW vs HWC) 的不同偏好？**

TinyEngine 的策略是"按需转换 (lazy re-layout)"：不是全局强制一种布局，而是在每层推理前的最后一刻，根据需要做转换。具体来说：(1) 深度分离卷积（逐通道操作）偏好 CHW——每个通道的数据连续，加载效率高；(2) 1×1 点卷积（本质是矩阵乘法）偏好 HWC——可以当成 matmul 来优化。TinyEngine 的 re-layout kernel 本身也是 SIMD 优化的——用 NEON 的 `vld4q` / `vst4q`（交错加载/存储指令）在寄存器内完成重排，避免在内存中创建一个完整的转换副本。这种极致的"内存零浪费"设计是通用框架无法实现的。

**Q4（高难度）：假设你在 ARM Cortex-M7（双 issue 流水线）上实现一个 3×3 depthwise convolution kernel。写两个版本：一个用 SIMD NEON（一次处理 16 个 INT8），一个用纯标量。在什么输入条件下，SIMD 版本反而比标量版本慢？**

SIMD 比标量慢的条件发生在**小输出 + 大开销场景**：(1) 当 H_out × W_out × C_in 极小时（如 4×4×3），SIMD 的"打包/拆包"开销（将数据加载到 NEON 寄存器、重新排列以满足 SIMD 宽度、写回时的对齐处理）可能占整个 kernel 执行时间的 50-70%。纯标量不需要这些开销。(2) Cortex-M7 的双 issue 流水线在标量代码时可以同时发射一条 ALU 指令和一条 load/store 指令——在某些简单循环（如逐元素加法）上标量代码的 IPC 可以达到 ~1.8，而 SIMD 代码由于指令依赖（等待 NEON 结果）IPC 可能只有 ~1.2。两者的实际吞吐差距大大缩小。

但在 TinyEngine 的实际场景中，这种"SIMD 更慢"的情况很少见——因为 MCUNet 搜出的模型通常有足够大的激活图（至少 8×8）使得 SIMD 的"打包"开销被均摊。TinyEngine 中的实际策略是：为极小的层（如 stride=2 后的第一层只有 4×4）使用"fallback 标量 kernel"——节省了 SIMD 调度的开销，同时也减少了代码大小（标量 kernel 通常只有 SIMD kernel 的 1/5）。

**Q5（高难度）：TinyEngine 的内存分配策略是静态 arena（在编译时确定所有 buffer 大小和复用关系），这与 PyTorch 的动态分配完全不同。如果模型含有动态 shape（如不同输入分辨率导致不同的中间张量大小），TinyEngine 如何处理？**

不处理——这就是 TinyEngine 的根本局限。TinyEngine 的静态 arena 设计假设整个计算图的所有张量大小在编译时完全确定。如果输入分辨率变化，所有中间张量的大小都变，预计算的 arena 布局就失效了。

实际项目中的处理方案：(1) **多配置编译**：为每种预知的输入分辨率分别编译一套 TinyEngine 代码（每个分辨率对应一个静态 arena 配置），运行时根据实际输入选择对应的二进制。对于产品定义（如 KWS 固定 16kHz×1s 输入，或图像固定 224×224），这完全可行。STM32Cube.AI 和 Edge Impulse 都采用这个策略。(2) **预留最大尺寸**：如果变化范围可预测（如输入在 128×128 到 224×224 之间），预编译时为最大尺寸（224×224）分配 arena，小输入时 arena 利用率低但 OOM 风险为零。这对 1MB SRAM 的 MCU 不是大问题，但对 128KB SRAM 的场景——为 224×224 分配的 arena 可能是 256KB，根本放不下。(3) **运行时重规划**（microTVM 的研究方向）：在 MCU 上运行一小段代码，根据实际输入大小做快速内存规划，生成新的 arena 布局。这代表了编译型和动态型的折中，但尚未在 TinyEngine 中实现。

**Q6（高难度）：如果在 Cortex-M7 上运行的 TinyEngine 模型突然精度下降（在相同输入上、相同模型权重、相同代码），但在 Cortex-M4 上正常，最可能的原因是什么？如何排查？**

这是 MCU 部署中最令人头疼的 heisenbug 场景。最可能的根因（按概率排序）：

**(1) Cortex-M7 的 L1 cache 不一致（概率 40%）**：M7 有数据和指令 cache（通常各 4-16KB），而 M4 没有 cache。如果某段代码使用 DMA 从 Flash 加载权重到 SRAM，而 CPU 的 data cache 中还缓存着旧数据——CPU 读到的是 cache 中的脏数据，而非 DMA 刚写入的新数据。M4 没有 cache 所以不会有这个问题。**排查**：在 DMA 传输完成后插入 `SCB_CleanInvalidateDCache()`（ARM CMSIS 函数）强制刷新 data cache。

**(2) Cortex-M7 的 double-precision FPU 引入了浮点累加顺序差异（概率 30%）**：M7 有硬件 FP64，M4 用 FP32。如果代码中某处 float 被隐式提升到 double（如 `float result = int_val * 2.0`，C 语言中 `2.0` 是 `double`），M7 会用 FP64 计算——精度高于 FP32 但结果值与 M4 不同。多次累积后精度差异可能放大到 0.01-0.1 量级。**排查**：在编译选项中加 `-Wdouble-promotion` 警告所有隐式类型提升，将可疑的常量后缀改为 `f`（如 `2.0f`）。

**(3) NEON SIMD 的"tail handling"溢出 bug（概率 20%）**：当通道数不是 16 的倍数时（INT8 NEON 一次处理 16 个），剩余的几个元素需要用标量代码处理（tail loop）。如果 tail loop 的边界条件写错了（如迭代次数 off by 1），M4（全标量，无 tail handling）不会有问题，M7（SIMD + tail loop）就会出现数据损坏。**排查**：将通道数强制改为 16 的倍数（padding zeros），看精度是否恢复正常。

**(4) 电源/时钟不稳（概率 10%）**：M7 在高频率（480MHz）下如果供电电压 margin 不足，可能出现偶尔的位翻转（bit flip）——极其罕见但在工业环境中确有案例（STM32 官方 errata 中有记载）。如果是这个问题，降低频率到 240MHz 应该恢复正常。

## 10. 本讲总结

TinyEngine 是"全栈优化"的教科书级别案例——从 SIMD 汇编到内存布局到算子融合，每一层都被精心调校：

- **Naive 实现在 MCU 上是灾难**：MCU 的内存层级（SRAM 极快极小，Flash 大但极慢）使得未经优化的算法性能断崖式下降。
- **SIMD 是 MCU 上唯一的"并行"**：NEON 128-bit 指令一次处理 16 个 INT8 元素，是 TinyEngine 加速的核心手段。
- **In-place + Fusion** 将"无用的内存分配和搬运"砍到最低：对于只有 256KB SRAM 的 MCU，省一个 buffer = 省 3-12% 的总内存。
- **Loop ordering 决定 cache 性能**：正确的顺序可以将 cache miss rate 从 30% 降到 5%。
- **Winograd vs Im2Col vs 直接卷积** 的选择需要基于每层的特征和硬件能力——TinyEngine 为每一层单独决策。
- **TinyEngine vs TFLite Micro 的根本差异**：专用 vs 通用。TinyEngine 只为 TinyNAS 搜到的操作子集写最优代码，不做任何不必要的通用性妥协。

一句话总结：TinyEngine 教给我们的不只是"如何为 MCU 写代码"，而是**在资源受限的系统中，"软件和硬件深度耦合"才是最优解**——你必须理解硬件的每个细节（cache 行大小、SIMD 宽度、Flash 延迟），然后为它们写专门定制的代码。通用框架在极端约束下注定失败，专用化是唯一的出路。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 在 MCU 上禁用 Im2Col，必须使用直接滑动窗口卷积 | ARM CMSIS-NN 实测：32×32×3 输入经 Im2Col 为 3×3 conv 准备时，实际 buffer 膨胀 16×（48KB），超过 MCU SRAM 预算 | STM32F4（96KB SRAM）上 Im2Col 导致 OOM，推理直接崩溃 |
| Winograd 在 MCU 低 bitwidth（INT16/FP16）下应避免使用于 depthwise conv | MIT 团队实测 STM32F7：Winograd F(2,3) + INT16 的 depthwise conv 精度比直接滑动窗口 INT16 低 1.8-2.3%（ImageNet） | 使用 Winograd"优化"反而导致线上精度恶化，优化的方向完全错误 |
| TinyEngine kernel 对输入分辨率敏感，必须为不同分辨率维护"分辨率-循环顺序映射表" | microTVM 验证：32×32 和 96×96 输入的最优 loop order 不同，同一 kernel 在大输入上慢 12% | 模型输入分辨率变化时推理延迟不可预测地波动，产品体验不稳定 |
| Cortex-M7 部署时必须处理 L1 cache 一致性问题 | M7 有数据和指令 cache（M4 无），DMA 加载权重后 CPU cache 可能读旧数据；须插入 SCB_CleanInvalidateDCache() | 权重数据被 cache 脏数据污染，输出随机错误——heisenbug 排查需数天 |
| NEON SIMD 的 tail loop 边界条件必须彻底测试 | INT8 NEON 一次处理 16 个元素，通道数非 16 倍数时 tail loop off-by-1 → M4（无 SIMD）无问题，M7 静默出错 | 特定通道配置下模型精度间歇性异常，问题极难复现和定位 |
| TFLite Micro 到 TinyEngine 迁移时须重新做 INT8 校准 | TFLite Micro 和 TinyEngine 的 rounding mode 和 saturation 行为可能不同，量化误差差异在深层网络累积放大 | 看似等价替换后精度额外下降 1-3%，需回滚重校准 |
| TinyEngine 的静态 arena 不支持动态 shape，须为每种输入尺寸单独编译 | STM32Cube.AI 和 Edge Impulse 的做法：预编译多份二进制；不支持时需预留最大尺寸的 arena（小输入浪费严重） | 128KB SRAM 上为 224×224 预留 256KB arena——完全无法部署 |

## 12. 学习闭环补充：TinyEngine、Kernel 与 Memory Planner

### 12.1 工业核心

TinyEngine 的关键是静态化：静态内存规划、静态 kernel 选择、静态代码生成。相比通用 runtime，TinyEngine 牺牲灵活性换取更小内存和更低开销。

### 12.2 Kernel 选择

| 算子 | 常见实现 | 工业判断 |
|---|---|---|
| Conv 1x1 | GEMM/direct | compute-bound，SIMD 友好 |
| Conv 3x3 | im2col/GEMM/direct/Winograd | 需权衡 SRAM 和计算 |
| Depthwise | direct specialized kernel | FLOPs 低但 memory-bound |
| FC/Linear | int8 GEMM | requantization 影响性能 |

### 12.3 Memory Planning

TinyML runtime 通常用 arena allocator：提前计算每个 tensor 生命周期，让不重叠的 tensor 复用内存。目标是 peak SRAM 最小化，而不是 malloc/free。

### 12.4 对应代码实验

```bash
python src/lecture-11/main.py
```

观察代码生成、tiling、buffer reuse 的概念输出。

### 12.5 本讲验收问题

1. TinyEngine 和 TFLite Micro 的设计取舍是什么？
2. 为什么 depthwise conv 在 MCU 上不一定快？
3. arena allocator 如何降低 peak memory？
4. Winograd 为什么减少乘法但可能增加内存/数值问题？
5. int8 convolution 的 accumulator 和 requantization 是什么？

## 13. Python 代码补充：Tensor 生命周期复用示意

TinyEngine/TFLite Micro 会用 arena planner 复用 buffer。下面代码演示 first-fit 生命周期分配思想。

```python
def overlaps(a, b):
    return not (a["death"] < b["birth"] or b["death"] < a["birth"])

def first_fit_plan(tensors):
    tensors = sorted(tensors, key=lambda x: -x["bytes"])
    placed = []
    for t in tensors:
        offset = 0
        while True:
            conflict = False
            for p in placed:
                if overlaps(t, p):
                    a0, a1 = offset, offset + t["bytes"]
                    b0, b1 = p["offset"], p["offset"] + p["bytes"]
                    if not (a1 <= b0 or b1 <= a0):
                        offset = b1
                        conflict = True
                        break
            if not conflict:
                break
        t["offset"] = offset
        placed.append(t)
    peak = max(t["offset"] + t["bytes"] for t in placed)
    return placed, peak

tensors = [
    {"name": "a", "bytes": 1024, "birth": 0, "death": 2},
    {"name": "b", "bytes": 2048, "birth": 1, "death": 3},
    {"name": "c", "bytes": 1024, "birth": 3, "death": 4},
]
plan, peak = first_fit_plan(tensors)
print(plan)
print("peak", peak)
```

工业解读：memory planner 的收益来自复用不重叠生命周期，而不是压缩权重本身。

