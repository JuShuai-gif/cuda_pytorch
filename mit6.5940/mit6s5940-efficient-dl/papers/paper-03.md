# Paper 03: Once-for-All (Cai et al., ICLR 2020)

> 论文全称：**Once-for-All: Train One Network and Specialize it for Efficient Deployment**
> 发表会议：ICLR 2020
> 作者：Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han（MIT HAN Lab）

---

## 1. 论文解决什么问题

传统 NAS 的一个根本问题是：**针对每个硬件平台和设备约束都需要从头重新搜索和训练**。目标硬件从 GPU → CPU → MCU，每个平台有不同的延迟/功耗/内存约束，如果每个部署场景都跑一次完整的 NAS，成本极高（hundreds of GPU hours × N 种配置）。

**Once-for-All (OFA)** 提出：**只训练一个"超网"（supernet），然后从中提取不同大小、不同延迟的子网**，适应各种硬件约束，无需重新训练。这大幅降低了针对不同硬件的模型定制成本。

---

## 2. 核心方法

### 超网设计
- 基于 MobileNetV3 结构构建超网，支持搜索的维度：
  - **Elastic depth**：每 stage 的 block 数量可变
  - **Elastic width**：每个卷积层的输出通道数可变（channel-wise 可选比例：1.0, 0.8, 0.6, 0.4）
  - **Elastic kernel size**：每个 depthwise conv 的 kernel size 可选（3×3, 5×5, 7×7）
  - **Elastic resolution**：输入图像分辨率可变（128~224）

### Progressive Shrinking 训练策略
不直接训练整个超网（会导致精度崩溃），而是分阶段逐步缩小：

**Stage 1: 最大网络训练（full network）**
- 训练 kernel=7, width=1.0, depth=full, resolution=224 的完整网络
- 得到性能上界（精度最高的网络）

**Stage 2: Elastic Kernel Size 训练**
- 固定其他维度，只变化 kernel size（7, 5, 3, 7, 5, 3...）
- 在每个 training step 随机抽一种 kernel size 做 forward

**Stage 3: Elastic Depth 训练**
- 固定 kernel size 和 width，只变化 depth（使用前 K 个 blocks）
- 关键技巧：用 skip connection 连接，使得浅层子网也能学到有用的特征

**Stage 4: Elastic Width 训练**
- 固定其他维度，变化 channel width
- 使用 channel sorting（按 L1 norm 排序）确保重要的通道被优先保留

### 子网搜索
- 训练完成后，在超网上用**进化算法**搜索满足给定硬件约束的子网
- 无需 fine-tuning（直接提取子网权重，精度已有保证）
- 硬件约束通过 latency predictor 建模（每个 deploy scenario 建一个简单回归模型）

---

## 3. 关键公式

### Progressive Shrinking 目标
设 $\theta$ 为超网参数，$\mathcal{A}$ 为所有可能的子网架构：

$$\min_{\theta} \mathbb{E}_{a \sim \mathcal{A}} \left[ \mathcal{L}(\theta_a; \mathcal{D}) \right]$$

其中 $\theta_a$ 是架构 $a$ 对应的子网权重（超网的子集）。

### Channel Sorting
对卷积核按 L1 norm 排序：

$$\text{importance}(i) = \|\theta_{:,i,:,:}\|_1$$

保留 L1 norm 最大的前 k% 通道。

### 硬件约束搜索
给定 target latency $T$，求最优子网：

$$a^* = \arg\max_{a \in \mathcal{A}} \text{Acc}(a) \quad \text{s.t.} \quad \text{Latency}(a) \leq T$$

Latency 通过硬件平台上实测数据训练的 predictor 估算。

---

## 4. 实验结论

### ImageNet 上不同延迟约束的结果（Samsung S20 手机）

| 子网延迟 (ms) | Top-1 Acc | 相比独立训练节省 GPU 时间 |
|---------------|-----------|---------------------------|
| 22 ms | 73.0% | 40× ↓ |
| 36 ms | 76.0% | 40× ↓ |
| 48 ms | 76.9% | 40× ↓ |
| 60 ms | 77.9% | 40× ↓ |
| 80 ms | 79.1% | 40× ↓ |

- **关键结论**：所有子网来自同一个超网，每个子网的精度都与独立训练相当
- 在多种硬件上（Samsung S20, NVIDIA 2080Ti, Intel Xeon CPU, Raspberry Pi）都验证有效
- Progressive Shrinking 比直接 joint training 精度高 **1.5-2.5%**

---

## 5. 工业价值

- **部署效率革命**：一次训练，N 次部署，极大降低模型定制的边际成本
- **硬件供应商广泛采用**：Samsung、Qualcomm、ARM 都使用或借鉴了 OFA 思路
- **开源即插即用**：GitHub 仓库 (>2k stars) 提供了多种硬件平台的预训练子网，可直接使用
- **产品形态灵活**：同一模型包内含多个子网，运行时根据设备性能和电池状态动态切换

---

## 6. 与课程 Lecture 的关系

- **Lecture 5 (Neural Architecture Search)**：OFA 是 weight-sharing NAS 的 SOTA 方案，是 Lecture 5 的重点讨论论文
- **Lecture 6 (Automated Pruning)**：Progressive Shrinking 受 automated channel pruning 思想的启发
- **Lecture 7 (System Co-design)**：OFA 的 multi-hardware 部署思路体现了 algorithm-system co-design
- **Lecture 8 (Deployment)**：子网搜索阶段用硬件 latency lookup table 进行加速，是部署环节的实践

---

## 7. 我应该如何复现

1. **构建超网**：用 PyTorch 实现 MobileNetV3 超网，支持 elastic depth/width/kernel/resolution
2. **Progressive Shrinking 训练**：
   - Stage 1: 最大网络，100 epoch
   - Stage 2: Elastic kernel，在每个 batch 随机选 kernel size（3/5/7），50 epoch
   - Stage 3: Elastic depth，随机选 depth，25 epoch
   - Stage 4: Elastic width，用 channel sorting 选 top-k%，25 epoch
3. **Latency Predictor**：
   - 在目标 CPU 上随机跑 1000 个子网，测实际延迟
   - 用 3 层 MLP 拟合（架构编码 → 延迟）
4. **进化搜索**：用 NSGA-II 进化算法，population=100，世代=500，目标=最大化精度，约束=延迟 < T
5. **验证**：提取搜索到的最优子网，在 ImageNet 验证集上测精度，在目标设备上测实际延迟
6. **简化方案**：用 torch.profiler 或 fvcore 估算 FLOPs 代替实际延迟测量

