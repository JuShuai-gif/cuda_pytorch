# EIE: Efficient Inference Engine on Compressed Deep Neural Network

> Song Han et al., ISCA 2016

## 1. 论文解决什么问题

深度学习模型推理在数据中心和移动设备上消耗大量计算资源和能耗。虽然 Deep Compression（剪枝+量化+霍夫曼编码）能将模型压缩数十倍，但压缩后的稀疏矩阵在通用 CPU/GPU 上并不能直接获得等比例加速——不规则的内存访问模式导致硬件利用率极低。本文解决了**如何设计专用硬件加速器高效执行压缩后的稀疏神经网络推理**这一核心问题。

具体挑战包括：(1) 非零权重分布极不规则，难以向量化；(2) 激活值本身也是稀疏的（ReLU 后约 50-70% 为零），但传统硬件无法跳过零激活值的计算；(3) 压缩后的权重需要间接寻址（CSR/CSC 格式），引入了额外的索引开销。

## 2. 核心方法

EIE 是一种专用 ASIC 加速器，针对 Deep Compression 输出的稀疏权重矩阵设计了高效的推理流水线。核心设计包括四大组件：

1. **CSC（Compressed Sparse Column）格式存储**：权重矩阵按列存储，每列包含非零权重值及其对应的行索引。输入激活向量遍历时，每个非零激活值触发对应列的非零权重乘法。这样既跳过了零权重，也跳过了零激活值。

2. **分布式处理单元（PE）阵列**：每个 PE 处理矩阵的若干列。PE 内部包含：
   - **Sparse Matrix Read Unit**：从 SRAM 中读取 CSC 格式的权重
   - **Arithmetic Unit**：执行乘加操作
   - **Activation Queue**：缓存非零激活值及其索引

3. **Leading Non-zero Detection**：从压缩的激活向量中快速定位下一个非零元素，避免逐元素扫描。

4. **权重共享的查表计算**：量化后的权重通过码书索引存储，PE 内部维护中心值查找表（centroid table），计算时将码书索引转换为实际浮点值再相乘，大幅减少权重的存储和带宽需求。

整体数据流为：非零输入激活 → 索引匹配 → 读取对应列权重 → 乘加累积 → 激活函数 → 压缩输出激活 → 传递到下一层。

## 3. 关键公式（LaTeX）

**CSC 格式下的稀疏矩阵-向量乘法**：

$$
y_j = \sum_{i \in \text{nz\_cols}[j]} W[i, j] \cdot a_i
$$

其中 $\text{nz\_cols}[j]$ 是第 $j$ 列中非零权重对应的行索引集合。只有当 $a_i \neq 0$ 且 $W[i,j] \neq 0$ 时，乘法才会执行。

**能量效率定义**：

$$
\text{Energy Efficiency} = \frac{\text{Operations}}{\text{Energy}} \quad (\text{GOPS/W})
$$

**权重共享的查表计算**：

$$
W_{ij} = \text{codebook}[\text{idx}[i, j]], \quad y_j \mathrel{+}= \text{codebook}[\text{idx}[i, j]] \cdot a_i
$$

索引 $\text{idx}[i,j]$ 通常为 4-bit，对应码书中的 16 个聚类中心值。

**稀疏度加速比**（理想情况）：

$$
\text{Speedup} \approx \frac{1}{(1 - S_W) \cdot (1 - S_A)}
$$

其中 $S_W$ 为权重稀疏度，$S_A$ 为激活稀疏度。当 $S_W=0.9$、$S_A=0.6$ 时，理论加速约 $\frac{1}{0.1 \times 0.4} = 25\times$。

**总功耗模型**：

$$
P_{\text{total}} = P_{\text{SRAM}} + P_{\text{PE}} + P_{\text{ActQueue}} + P_{\text{LNZD}}
$$

## 4. 实验结论

- **能效**：EIE 达到 120 GOPS/W，而同期 GPU（Tegra K1）约为 2 GOPS/W，能效提升约 60 倍；比同期 CPU 提升约 2700 倍
- **吞吐量**：在 9 层全连接网络上（AlexNet 的 FC 层），EIE 处理单张图片仅需 0.6ms，吞吐量达 102 GOPS
- **稀疏加速**：当权重稀疏度为 90%、激活稀疏度为 60% 时，实际加速比约为 13-16×（相比于 dense 基线），接近理论上限
- **面积效率**：在 45nm CMOS 工艺下，EIE 面积约 40mm²，功耗约 600mW，适合嵌入式和移动端部署
- **精度损失**：由于 EIE 直接执行 Deep Compression 后的模型，精度损失由压缩算法决定（通常 <1%），硬件本身不引入额外精度损失

## 5. 工业价值

EIE 开创了**稀疏神经网络专用加速器**的研究方向，对后续 AI 芯片设计产生了深远影响。其核心思想——"跳过零权重和零激活值"——已成为现代 AI 加速器的基本设计原则：

- **NVIDIA A100/H100** 的稀疏张量核心（2:4 结构化稀疏，2× 加速）直接继承了 EIE 的稀疏加速理念
- **Google TPU**、**Apple Neural Engine** 等均在不同程度上借鉴了压缩模型加速的策略
- EIE 证明了**算法-硬件协同设计**的威力：将 Deep Compression 的软件压缩成果通过专用硬件"兑现"为真正的速度/能效提升
- 将 AI 推理从"云端大 GPU"扩展到 IoT/边缘设备，奠定了 TinyML 的硬件基础

## 6. 与课程 lecture 的关系

- **Lecture 04（Pruning II）**：EIE 是剪枝后的"落地"环节——剪枝产生稀疏权重矩阵，EIE 负责在硬件上高效执行这些稀疏矩阵乘法。剪枝的最终价值需要在稀疏硬件上才能体现，否则不规则稀疏在 GPU 上的加速效果有限。
- **Lecture 11（TinyEngine）**：TinyEngine 是剪枝+量化在 MCU 上的软件优化编译器，EIE 是其 ASIC 硬件对应物。两者都解决"如何让压缩模型在实际硬件上跑得快"的问题，只是层次不同——TinyEngine 在软件层做算子调度和循环优化，EIE 在硬件层做数据流和微架构优化。

## 7. 我应该如何复现

由于 EIE 是 ASIC 设计，完全复现需要硬件设计工具和流片。可以从以下路径接近：

1. **RTL 仿真复现**：
   - 搭建 Verilog/VHDL 仿真环境（ModelSim 或 Verilator）
   - 实现 CSC 格式的稀疏矩阵乘法单元
   - 实现 Leading Non-zero Detection 模块
   - 使用 PyTorch 导出的量化+剪枝模型权重作为测试输入

2. **FPGA 原型验证**：
   - 在 Xilinx/Intel FPGA 上部署简化版 EIE 架构
   - 使用 HLS（高层次综合）快速原型：用 C++ 描述 PE 阵列逻辑，Vivado HLS 综合到 FPGA

3. **软件模拟器**：
   - 编写 cycle-accurate 的 EIE 模拟器（Python/C++）
   - 输入 AlexNet/VGG 的压缩权重，统计每层的周期数、访存次数、能耗
   - 与 GPU/CPU 推理时间做对比

4. **参考开源项目**：
   - NVDLA（NVIDIA 开源深度学习加速器）的架构与 EIE 有相似之处
   - Gemmini（UC Berkeley 的 systolic array 生成器）提供了学术界的加速器设计框架
   - 关注 MIT HAN Lab 后续的 Eyeriss 系列芯片，它们在 EIE 基础上持续演进
