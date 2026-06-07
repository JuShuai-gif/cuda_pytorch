# Paper 01: Deep Compression (Han et al., ICLR 2016)

> 论文全称：**Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding**
> 发表会议：ICLR 2016（Oral）
> 作者：Song Han, Huizi Mao, William J. Dally（Stanford / NVIDIA）

---

## 1. 论文解决什么问题

深度学习模型的参数量和计算量巨大，难以部署到移动端、嵌入式等资源受限的设备上。例如 AlexNet 需要 240MB 存储，VGG-16 需要 552MB，无法放入手机 App 中。本文提出的 **Deep Compression** 三阶段流水线将剪枝（Pruning）、量化（Quantization）和霍夫曼编码（Huffman Coding）组合成一条完整的模型压缩流程，在不损失精度的前提下将模型体积压缩 35-49 倍。

---

## 2. 核心方法

Deep Compression 分三个步骤，按序执行：

### 阶段一：剪枝（Pruning）
- 先正常训练一个网络，然后移除 `weight` 值接近零的连接（低于阈值的连接被置零并永久移除）
- 通过迭代剪枝：每次剪一小部分，然后重新训练（fine-tune）恢复精度
- 由于剪枝后改变了网络结构（变为稀疏连接），需要使用 **Compressed Sparse Row (CSR)** 或 **Compressed Sparse Column (CSC)** 格式存储稀疏权重矩阵

### 阶段二：权重量化（Weight Quantization）
- 使用 **Trained Quantization**（即训练感知量化）：在量化后继续训练，让网络适应量化误差
- 采用 K-means 聚类来确定量化中心点（centroids），权重被替换为最近中心点的索引
- 共享权重的 centroid 表进一步用霍夫曼编码压缩

### 阶段三：霍夫曼编码（Huffman Coding）
- 对量化后的权重索引（centroid indices）进行霍夫曼编码
- 高频出现的 centroid index 用短码表示，低频的用长码表示
- 进一步减少模型存储体积 20-30%

---

## 3. 关键公式

### 剪枝阈值设置
设 $W$ 为权重矩阵，剪枝阈值为 $\theta$：

$$W_{ij}' = \begin{cases} W_{ij} & \text{if } |W_{ij}| > \theta \\ 0 & \text{otherwise} \end{cases}$$

### K-means 量化目标函数
最小化量化前后的权重均方误差：

$$\min_{C} \sum_{i=1}^{n} \|W_i - C_{\text{idx}(i)}\|^2$$

其中 $C$ 是 centroid 表，$\text{idx}(i)$ 将权重 $W_i$ 映射到最近的 centroid。

### 反向传播（梯度更新）
尽管量化后权重离散，前向传播使用量化后的权重，反向传播时梯度直接传给原始全精度权重（Straight-Through Estimator, STE）：

$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial W_i^{\text{quantized}}}$$

### 压缩比计算
总压缩比 = 剪枝压缩比 × 量化压缩比 × 编码压缩比

$$R_{\text{total}} = R_{\text{prune}} \times R_{\text{quant}} \times R_{\text{huffman}}$$

---

## 4. 实验结论

| 网络 | 原始大小 | 压缩后 | 压缩比 | Top-1 精度变化 |
|------|----------|--------|--------|----------------|
| AlexNet | 240 MB | 6.9 MB | 35× | 无损失 |
| VGG-16 | 552 MB | 11.3 MB | 49× | 无损失 |
| LeNet-300-100 | 1070 KB | 27 KB | 40× | 无损失 |
| LeNet-5 | 1720 KB | 44 KB | 39× | 无损失 |

- 卷积层对剪枝的敏感度高于全连接层（需要更低的剪枝率）
- Trained Quantization 比直接 PTQ 精度更高
- 霍夫曼编码在 conv 层贡献 20%-30% 额外压缩

---

## 5. 工业价值

- **奠基性工作**：这是将剪枝+量化+编码三个技术组合成完整 pipeline 的开创性工作
- **实际部署**：后续被 NVIDIA TensorRT、Qualcomm SNPE 等推理引擎借鉴
- **概念验证**：证明了深度学习模型可以在边缘设备上高效运行
- **启发性**：OpenAI 在 2016 年报告中也引用了该工作的压缩思路

---

## 6. 与课程 Lecture 的关系

- **Lecture 3 (Pruning)**：本文是 magnitude-based pruning 的经典代表，结合了 fine-tuning 恢复精度的策略
- **Lecture 4 (Quantization)**：K-means 量化和 STE 梯度估计是量化技术的重要基础
- **Lecture 7 (System + Algorithm Co-design)**：三阶段流水线体现了算法-系统协同设计的理念
- **Lecture 1 (Introduction)**：论文的 motivation（模型太大无法部署到边缘设备）正是课程开篇讨论的核心问题

---

## 7. 我应该如何复现

1. **选模型**：用 PyTorch 加载预训练 ResNet-18 或 MobileNetV2
2. **剪枝**：对每个 conv 层按 magnitude 排序，去掉绝对值最小的 30%-50% 权重，生成稀疏 mask，fine-tune 5-10 epoch
3. **量化**：将剩余权重收集起来做 K-means（`sklearn.cluster.KMeans`），n_clusters=256（对应 8-bit），用 centroids 替换权重，继续 fine-tune 5 epoch
4. **编码**：统计每个 centroid index 的频率，用 `dahuffman` 或手写霍夫曼树编码
5. **验证**：在 CIFAR-10/ImageNet 验证集测试精度，计算模型文件大小变化
6. **关键注意事项**：剪枝和量化都要分步迭代，一次性大比例剪枝会导致精度崩溃

