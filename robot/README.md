# CUDA 并行 Planning 算法集

基于 CUDA 实现的常用机器人/自动驾驶 Planning 算法，每个算法均包含 GPU 并行版本与 CPU 版本的性能对比。

## 算法总览

| 算法 | 加速比 | 核心并行策略 | 本质 |
|------|--------|-------------|------|
| [粒子群优化 (PSO)](#1-粒子群优化-pso) | 105x | 每线程一个粒子 | 批量数据并行 |
| [遗传算法 (GA)](#2-遗传算法-ga) | 3.9x | 每线程一个个体 | 批量数据并行 |
| [蚁群优化 (ACO)](#3-蚁群优化-aco) | 147x | 每线程块一只蚂蚁 | 批量数据并行 |
| [Dijkstra 最短路径](#4-dijkstra-最短路径) | 132x | 每线程一个节点 | 批量数据并行 |
| [横纵时空联合轨迹优化](#5-横纵时空联合轨迹优化) | 562x | 每线程一条轨迹 | 批量数据并行 |
| [卡尔曼滤波 KF/EKF/UKF](#6-卡尔曼滤波-batch) | 63x/967x/10.5x | 每线程一个滤波器 | 批量数据并行 |
| [GPS+IMU 松耦合定位](#7-gpsimu-松耦合定位) | 16.6x | 每线程一个目标 | 批量数据并行 |

> **加速的本质不是矩阵乘法，是批量数据并行 (Batch Data Parallelism)**：
> 同样的运算逻辑作用在不同数据上，GPU 上万核心同时执行。
> 单个运算很轻量（几十次浮点），但 10,000 个同时算，就比 CPU 串行快几百倍。

---

## 1. 粒子群优化 (PSO)

### 原理

模拟鸟群觅食。每个粒子跟踪个体最优 + 全局最优来更新速度与位置：

```
v = w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
```

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| 无人机编队路径规划 | 大疆 FlightAutonomy：数百架无人机同时优化无碰撞轨迹 |
| 5G Massive MIMO 波束赋形 | 华为/中兴基站用 PSO 优化 64+ 天线相位权重 |
| 电动汽车电池 SOC 估计 | 特斯拉 BMS 用 PSO 在线拟合电池等效电路模型参数 |
| AutoML 超参搜索 | Google Vizier / Microsoft NNI，替代 grid search |
| 工业 PID 自整定 | ABB/KUKA 机器人关节电机参数自动优化 |

```bash
python robot/particle_swarm/pso.py
```

---

## 2. 遗传算法 (GA)

### 原理

模拟自然选择：种群通过锦标赛选择 → 算术交叉 → 高斯变异逐代进化，精英个体保留防止退化。

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| 外卖骑手调度 | 美团/饿了么：2000 单 × 500 骑手，GA 秒级出调度方案 |
| 芯片布局布线 | Google TPU 用 GA 优化片上网络 (NoC) 拓扑 |
| 卫星星座相位调整 | 遥感卫星多目标 GA 同时优化燃料消耗 + 覆盖范围 |
| 游戏 AI 进化 | Gran Turismo Sophy 类方案：赛车 AI 自动进化最优策略 |
| 蛋白质结构预测 | Rosetta@home 早期用 GA 搜索折叠构象空间 |

```bash
python robot/genetic/ga.py
```

---

## 3. 蚁群优化 (ACO)

### 原理

模拟蚂蚁觅食：信息素正反馈使种群收敛到最短路径，信息素蒸发避免过早收敛。

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| PCB 钻孔最短路径 | 华为/富士康 SMT 产线：数百钻孔的最优走刀路径 |
| 快递末端配送 | 顺丰 CVRP：多车多站点 + 容量约束 + 时间窗 |
| 数据中心流量调度 | Google B4 SDN 用类蚁群算法做实时流量工程 |
| 灾害救援任务分配 | 地震后多救援队路径规划，动态避开坍塌建筑 |
| 药物合成路径 | CADD 中搜索最优化学反应路径 |

```bash
python robot/ant_colony/aco.py
```

---

## 4. Dijkstra 最短路径

### 原理

带权图单源最短路径。标准算法 O(V²) 天然串行（每轮找全局最小距离节点）。
GPU 版本用并行 Bellman-Ford 松弛 + atomicMin，稀疏大图上可达 132x 加速。

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| 地图导航引擎 | 百度/高德/Google Maps 底层就是 Dijkstra + A* |
| 互联网路由 (OSPF) | 你访问 GitHub 的每一跳路径由 Dijkstra 决定 |
| 芯片 EDA 自动布线 | Cadence/Synopsys 全局布线引擎 |
| 扫地机器人 | iRobot/科沃斯区域划分后区域间转移路径 |
| 运营商骨干网 | MPLS-TE CSPF (Constrained SPF) 基于 Dijkstra 扩展 |

```bash
python robot/dijkstra/dijkstra.py
```

---

## 5. 横纵时空联合轨迹优化

### 原理

自动驾驶规划核心。Frenet 坐标系下，横向五次多项式 + 纵向四次多项式采样生成 10,000+ 条候选轨迹，通过 9 个代价函数加权选优。

核心论文：Werling et al., "Optimal Trajectory Generation in a Frenet Frame", ICRA 2010。

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| L4 自动驾驶规划 | 百度 Apollo EM Planner —— 就是这套方案 |
| Robotaxi 行为决策 | Waymo：数十万条候选轨迹并行评估，GPU 加速 |
| 端到端规划进化 | 特斯拉 FSD：类似框架，代价函数由神经网络学习 |
| 自动泊车 (APA) | 小鹏/蔚来：低速 2D 时空联合搜索最优泊车路径 |
| 高速 NOA 变道 | 理想/问界：横纵解耦多项式实时生成变道轨迹 |

```bash
python robot/spacetime_traj/spacetime_traj.py
```

---

## 6. 卡尔曼滤波 (Batch)

### 原理

状态估计基石。利用系统动力学模型预测 + 传感器观测更新，递归估计状态及不确定性。支持三种变体：

| 变体 | 适用场景 | 加速比 |
|------|---------|--------|
| KF (线性) | GPS/视觉位置观测 | 63x |
| EKF (非线性) | 雷达距离+方位角 | 967x |
| UKF (Sigma 点) | 强非线性系统 | 10.5x |

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| 自动驾驶多目标跟踪 | Apollo：每个感知目标跑一个 KF，数百个滤波器并行 |
| 手机 GPS 定位 | iPhone/Android：GPS + 加速度计 EKF 融合 |
| 火箭垂直回收 | SpaceX Falcon 9：UKF 融合 GPS+IMU+雷达高度计 |
| 激光雷达目标跟踪 | Waymo/Cruise：点云聚类后每个目标 EKF 估计位速朝向 |
| 开源无人机飞控 | ArduPilot/PX4：EKF 融合 IMU+GPS+气压计+磁力计 |

```bash
python robot/kalman_filter/kalman_filter.py
```

---

## 7. GPS+IMU 松耦合定位

### 原理

IMU 提供 100Hz 加速度 → 高频位置预测 → GPS 1Hz 绝对位置校正 → 输出 100Hz 无漂移位姿。

### 实际落地案例

| 场景 | 具体产品/系统 |
|------|-------------|
| 自动驾驶组合导航 | Apollo/NovAtel：GNSS+INS 紧耦合输出 100Hz 位姿 |
| 手机 ARKit/ARCore | Apple/Google：视觉惯性里程计 = IMU 预测 + 视觉校正 |
| 无人机自动返航 | 大疆 Failsafe RTH：GPS 丢失后纯 IMU 推算，恢复后校正 |
| 洲际导弹制导 | 民兵-3/东风：INS+GPS 校正，30 分钟飞行 CEP < 100m |
| 水下机器人导航 | 无 GPS 时 DVL+IMU 死推算，定时浮出水面 GPS 校正 |
| 高精地图采集车 | Mobileye/Here：紧耦合 GPS+IMU，输出 cm 级采集轨迹 |

```bash
python robot/gps_imu_fusion/gps_imu_fusion.py
```

---

## 文件结构

```
robot/
├── README.md                        ← 本文件
├── particle_swarm/
│   ├── README.md
│   ├── pso_kernel.cu
│   └── pso.py
├── genetic/
│   ├── README.md
│   ├── ga_kernel.cu
│   └── ga.py
├── ant_colony/
│   ├── README.md
│   ├── aco_kernel.cu
│   └── aco.py
├── dijkstra/
│   ├── README.md
│   ├── dijkstra_kernel.cu
│   └── dijkstra.py
├── spacetime_traj/
│   ├── README.md
│   ├── spacetime_kernel.cu
│   └── spacetime_traj.py
├── kalman_filter/
│   ├── README.md
│   ├── kalman_kernel.cu
│   └── kalman_filter.py
└── gps_imu_fusion/
    ├── README.md
    ├── gps_imu_kernel.cu
    └── gps_imu_fusion.py
```
