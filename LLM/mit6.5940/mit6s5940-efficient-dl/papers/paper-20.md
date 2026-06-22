# Paper 20: BEVFusion — Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation (Liu et al., ICRA 2023)

> 论文全称：**BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation**
> 发表会议：ICRA 2023（Oral）
> 作者：Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela Rus, Song Han（MIT HAN Lab / MIT CSAIL）

---

## 1. 论文解决什么问题

自动驾驶系统依赖**多传感器融合**（摄像头 + LiDAR）来实现鲁棒的三维感知。然而，摄像头生成的是透视视图（2D），LiDAR 产生的是稀疏点云（3D），两者的数据模态和几何空间天然不同。现有融合方法要么计算量大（基于点的融合需要为每个 LiDAR 点查找图像特征），要么信息损失严重（基于提案的融合只保留局部区域）。

BEVFusion 提出一个**统一的鸟瞰图（Bird's-Eye View, BEV）表示**——将摄像头和 LiDAR 的信息都投影到共享的 BEV 空间中进行融合，同时完成 3D 检测和 BEV 语义分割，推理速度满足实时要求（在 Jetson Orin 上 >25 FPS）。

---

## 2. 核心方法

### 统一 BEV 表示

BEV 是俯瞰角度下的二维网格表示（如 200×200 的栅格，覆盖 100m×100m 范围）。在 BEV 空间中：
- 自动驾驶场景的物体高度差异不大（车高约 1.5m，行人高度约 1.7m）
- 遮挡问题显著减少
- 传感器融合变得自然——所有传感器投影到同一个坐标系

### 摄像头 → BEV 投影（Lift-Splat-Shoot）

使用 **LSS（Lift-Splat-Shoot）** 方法：
1. **Lift**：对每个图像像素 $(u, v)$ 预测一条沿射线方向的深度分布 $D \in \mathbb{R}^{N_D}$（$N_D$ 为离散深度 bins），将像素特征沿深度方向 "抬起" 成视锥（frustum）特征
2. **Splat**：通过相机内参和外参矩阵，将 3D 视锥特征"拍平"投影到 BEV 栅格上，对落入同一栅格的特征做 pooling
3. **Shoot**：在 BEV 空间用轻量级 CNN backbone 进一步提取特征

### LiDAR → BEV 投影

对 LiDAR 更简单——LiDAR 点云本身就在 3D 空间中：
1. 将每个 LiDAR 点 $(x, y, z, f)$ 的 3D 坐标映射到 BEV 栅格
2. 对同一栅格内的点特征做 max pooling 或 mean pooling
3. 得到与摄像头 BEV 特征同尺寸的 LiDAR BEV 特征图

### 融合策略

在 BEV 空间中将两个特征图拼接（concatenate）后送入后续的 detection/segmentation head：

$$F_{\text{BEV}} = \text{Concat}(F_{\text{camera\_BEV}}, F_{\text{LiDAR\_BEV}})$$

### 多任务 Head

BEVFusion 共享 BEV 特征编码器，输出分叉到两个任务：
- **3D 检测 Head**：基于 CenterPoint（anchor-free 检测器），预测物体中心、尺寸、朝向、速度
- **BEV 语义分割 Head**：对每个 BEV 栅格做 semantic classification（道路、车道线、人行道、车辆等）

---

## 3. 关键公式

### Lift-Splat-Shoot：从图像像素到 BEV 栅格

对于图像特征 $I \in \mathbb{R}^{H \times W \times C}$ 和深度分布 $D \in \mathbb{R}^{H \times W \times N_D}$：

视锥特征（frustum features）：
$$F_{\text{frustum}}[u, v, d] = I[u, v] \cdot D[u, v, d]$$

投影到 3D 世界坐标（通过相机内参 $K$ 和外参 $[R|t]$）：
$$\begin{bmatrix} x \\ y \\ z \end{bmatrix} = R^{-1} \left(K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \cdot d - t\right)$$

拍平到 BEV 栅格 $(b_x, b_y)$：
$$F_{\text{camera\_BEV}}[b_x, b_y] = \sum_{(u,v,d): \text{project to }(b_x, b_y)} F_{\text{frustum}}[u, v, d]$$

### LiDAR → BEV 投影

对于 LiDAR 点 $(x_i, y_i, z_i)$，特征 $f_i$，BEV 分辨率 $\Delta$（如 0.5m/pixel）：

$$F_{\text{LiDAR\_BEV}}\left[\left\lfloor \frac{x_i}{\Delta} \right\rfloor, \left\lfloor \frac{y_i}{\Delta} \right\rfloor\right] = \max\left(F_{\text{LiDAR\_BEV}}[\dots], f_i\right)$$

### 融合特征

$$F_{\text{fused}} = \text{Conv}_{3\times3}\left([F_{\text{camera\_BEV}}; F_{\text{LiDAR\_BEV}}]\right)$$

---

## 4. 实验结论

### nuScenes 3D 检测（验证集）

| 方法 | Modality | NDS ↑ | mAP ↑ | FPS (A100) |
|------|----------|-------|-------|------------|
| PointPillars | L | 61.3 | 46.4 | 62 |
| CenterPoint | L | 67.3 | 60.3 | 28 |
| FCOS3D | C | 43.7 | 35.8 | 2 |
| BEVDet | C | 48.8 | 42.4 | 15 |
| TransFusion | L+C | 71.7 | 68.9 | 8 |
| **BEVFusion** | L+C | **72.9** | **70.2** | **25.2** |

### nuScenes BEV 语义分割

| 方法 | mIoU ↑ | FPS |
|------|--------|-----|
| BEVFusion (LiDAR-only) | 55.0 | 98 |
| BEVFusion (Camera-only) | 47.3 | 42 |
| **BEVFusion (Fusion)** | **62.7** | **25** |

### 边缘设备推理

| 平台 | Model | FPS |
|------|-------|-----|
| Jetson Orin (30W) | BEVFusion-Tiny | **25.2** |
| Jetson AGX Orin (60W) | BEVFusion-Base | **18.5** |
| RTX 4090 | BEVFusion-Base | **103** |

- 融合方案在 nds 和 mAP 上均超过纯 LiDAR 和纯视觉基线（+3% NDS vs TransFusion）
- BEV 语义分割 mIoU 达 62.7（LiDAR only: 55.0），说明摄像头能显著丰富语义信息
- 在 Jetson Orin 上实时运行（>25 FPS），证明了实际部署可行性
- 摄像头在夜间/雨雪天气下失效时，LiDAR 分支仍能独立工作（多模态冗余）

---

## 5. 工业价值

- **已被自动驾驶公司采用**：BEVFusion 在 Waymo Open Dataset 和 nuScenes 挑战赛中多次登顶，NVIDIA DRIVE 平台已集成
- **统一的感知架构**：检测 + 分割双任务共享 BEV 特征，减少了车端芯片上的模型数量（原来需要两个独立模型）
- **边缘部署验证**：在 Jetson Orin（车载级别芯片）上达到实时，是量产自动驾驶系统的重要参考实现
- **范式影响**：BEVFusion 的 BEV 统一表示思想影响了后续 BEVFormer、PETR、OccNet 等大量工作

---

## 6. 与课程 Lecture 的关系

- **Lecture 17（Efficient Point Cloud Understanding & Efficient Video Understanding）**：BEVFusion 是多视角摄像头 + LiDAR 融合的效率标杆，也是课程中点云理解模块的核心论文
- **Lecture 1-2（Efficiency Metrics）**：论文在精度（NDS/mAP/mIoU）+ 速度（FPS）+ 硬件（A100/Jetson）三维权衡中寻找 Pareto 最优
- **Lecture 7（NAS / Co-design）**：BEVFusion 的架构设计（LSS 投影 + BEV 融合 + 多任务 Head）体现了手工精心设计的高效架构，其思想与 NAS 的搜索目标一致

---

## 7. 我应该如何复现

1. **环境准备**：
   - PyTorch 1.13+，CUDA 11.6+，MMDetection3D 1.0+
   - 数据集：nuScenes（完整版 ~350GB，mini 版 ~10GB 用于快速验证）
   - 硬件：至少 1× RTX 3090 (24GB) 用于训练，Jetson Orin 用于边缘推理验证
2. **数据处理**：
   - 下载 nuScenes 数据（包含 6 摄像头 + 1 LiDAR + 标定文件）
   - 使用 nuscenes-devkit 解析标定参数（内参 $K$ + 外参 $[R|t]$）
3. **实现 LSS 投影**：
   - 对每个摄像头，根据预定义的深度 bins（如 $d \in [2, 4, 6, \dots, 60]$ 米）
   - 构建 frustum grid（$H \times W \times N_D \times 3$），通过相机矩阵投影到 3D
   - 使用 `torch.scatter` 或 `torch.cumsum` 实现高效的 splatting
4. **训练**：
   - 使用 MMDetection3D 训练 pipeline (配置见 `projects/BEVFusion`)
   - 端到端训练（图像 backbone + LSS 投影 + BEV encoder + detection head），20 epochs
   - 使用 CBGS（Class-Balanced Grouping and Sampling）策略处理类别不平衡
5. **验证**：
   - 在 nuScenes val set 上评估 NDS、mAP、mATE 等指标
   - 度量推理 FPS（在 A100 和 Jetson Orin 上分别测试）
6. **关键注意事项**：
   - LSS 中的深度 bins 数量 $N_D$ 是精度-速度权衡的关键超参，建议从 $N_D=64$ 开始
   - 多摄像头融合时要注意 BEV 栅格的对齐和重叠区域的处理
   - Jetson Orin 部署需要用 TensorRT 对模型进行 FP16/INT8 优化
