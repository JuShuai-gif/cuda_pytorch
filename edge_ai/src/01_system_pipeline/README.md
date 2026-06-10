# 机器人感知 → 规划 → 控制流水线

处理真实机器人传感器数据，经过 3 阶段感知 → 规划 → 控制流水线。使用真实的图像/LiDAR 处理、卡尔曼滤波、三次样条轨迹生成和 PID 控制。

## 文件结构

```
01_system_pipeline/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文件
├── pipeline_config.h           # 数据结构：CameraImage、PointCloud、DetectionBox、
│                               #   Trajectory、ControlCommand、KalmanTrack、流水线流类型
├── pipeline_stage.h            # 阶段函数声明（perception、planning、control）
├── pipeline_stage.cpp          # 阶段实现，包含真实数据处理
├── pipeline_executor.h         # 顺序与流水线执行器声明
├── pipeline_executor.cpp       # 顺序与流水线执行器实现
├── latency_stats.h             # 延迟统计收集器（线程安全）
├── latency_stats.cpp           # 统计 + JSON 指标导出
└── main.cpp                    # CLI 入口点
```

## 数据流水线

### 数据结构

| 结构体 | 字段 | 描述 |
|--------|------|------|
| `CameraImage` | width, height, channels, timestamp_ns, data（uint8 向量）, encoding | 原始相机帧（默认 1920x1080x3 rgb8） |
| `Point3D` | x, y, z, intensity, ring | 单个 LiDAR 点 |
| `PointCloud` | timestamp_ns, points 向量 | LiDAR 扫描（10 万点，64 环） |
| `DetectionBox` | class_id, confidence, x/y/z, w/h/d, vx/vy | 带速度的 3D 边界框 |
| `Detections` | timestamp_ns, frame_id, boxes 向量 | 每帧检测结果 |
| `Trajectory` | timestamp_ns, waypoints 向量 | 三次样条轨迹 |
| `Waypoint` | x, y, t（参数化）, v（目标速度） | 单个轨迹路点 |
| `ControlCommand` | throttle（0-1）, brake（0-1）, steering（-1..1）, timestamp_ns | 车辆控制输出 |
| `KalmanTrack` | track_id, class_id, x/y/vx/vy, P（4x4 协方差）, age, missed | 被跟踪物体状态 |

### 处理阶段

**感知阶段：**
1. 生成合成相机图像（1920x1080x3 随机像素）
2. 相机预处理：RGB → 灰度（ITU-R BT.601）、双线性缩放到 640x480、归一化到 [0,1]
3. 生成合成 LiDAR 点云（10 万点，64 环，真实环角度 ±25° 到 +15°）
4. LiDAR 预处理：距离过滤（0.5m–100m）、体素网格降采样（10cm）
5. 基于处理后数据密度生成检测结果

**规划阶段：**
1. 从检测结果初始化卡尔曼滤波器（恒速模型，4 状态）
2. 卡尔曼预测步骤（状态转移 F，过程噪声 Q）
3. 卡尔曼更新步骤（观测 z=[x,y]，2x2 矩阵求逆计算新息协方差）
4. 每个被跟踪物体的碰撞时间（TTC）计算
5. 通过控制点生成三次样条轨迹（自然样条，Thomas 算法）

**控制阶段：**
1. 横向误差计算（到轨迹段的符号距离）
2. 横向 PID 控制器 → 转向指令
3. 纵向 PID 控制器 → 油门/刹车指令

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

## Run

```bash
# 流水线模式（默认，3 帧飞行中）
./pipeline_sim

# 顺序模式
./pipeline_sim --mode sequential

# 自定义配置
./pipeline_sim --depth 5 --frames 200 --verbose

# 帮助
./pipeline_sim --help
```

## Options

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--mode` | pipelined | `sequential` 或 `pipelined` |
| `--depth` | 3 | 流水线深度（飞行中的帧数） |
| `--frames` | 100 | 总处理帧数 |
| `--verbose` | false | 输出每帧耗时到 stderr |

## Output

输出 `pipeline_metrics.json`，包含：
- 聚合统计：均值、标准差、P50、P99、每阶段最大延迟（纳秒）
- 分解：感知（图像预处理、LiDAR 预处理、检测）、规划、控制
- 端到端延迟与吞吐量（FPS）
