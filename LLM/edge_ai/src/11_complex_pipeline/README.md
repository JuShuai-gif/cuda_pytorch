# Autonomous Driving Pipeline Simulator

Realistic 7-stage autonomous driving pipeline with actual data processing.
Every stage performs real computation (CNN sliding window, Kalman filter,
A* planner, Stanley controller, etc.) - no simulate_work() burn loops.

## Pipeline Stages

```
SensorFrame -> Preprocess -> Detection -> Tracking -> Prediction -> Planning -> Control
```

## File Structure

```
11_complex_pipeline/
  pipeline_config.h      - PipelineConfig struct (depth, frames, seed)
  stage_types.h          - All 7 data structures with timestamps
  latency_stats.h/cpp    - LatencyStats: recording, stats, JSON output
  stage_queue.h          - Thread-safe inter-stage queue template
  stage_ops.h/cpp        - Real stage implementations
  pipeline_executor.h/cpp - SequentialExecutor + PipelinedExecutor
  main.cpp               - CLI entry point
  CMakeLists.txt
  README.md
```

## Stage Details

| Stage | Real Work |
|-------|-----------|
| Sensor | Generate 1920x1080x3 uint8 camera, 100K LiDAR points, IMU, GPS |
| Preprocess | RGB->YUV + resize 640x480, LiDAR RANSAC ground removal + 0.1m voxel grid, IMU bias correction |
| Detection | CNN 3x3 conv sliding window (224x224, stride 112), Euclidean LiDAR clustering |
| Tracking | 9-state Kalman filter (x,y,z,vx,vy,vz,ax,ay,az) with predict + update |
| Prediction | Constant acceleration model extrapolation for 5s horizon (100 timesteps) |
| Planning | A* on 100x100 grid with 15% random obstacles, cubic spline path smoothing |
| Control | Stanley controller (lateral) + PID (longitudinal) throttle/brake/steering |

## 推荐阅读顺序

1. **`pipeline_config.h` + `stage_types.h`** — 基础类型：PipelineConfig 控制执行，stage_types 定义流水线中流动的所有数据结构
2. **`latency_stats.h` + `latency_stats.cpp`** — 自包含的度量收集器，被 executor 使用
3. **`stage_queue.h`** — 阶段间有界队列模板，被 PipelinedExecutor 使用
4. **`stage_ops.h` + `stage_ops.cpp`** — 核心所在：7 个阶段的真实计算实现（Sensor → Control）
5. **`pipeline_executor.h` + `pipeline_executor.cpp`** — 编排层：SequentialExecutor 与 PipelinedExecutor
6. **`main.cpp`** — 最后阅读，CLI 参数解析、选择执行模式、输出最终报告

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Sequential mode
./pipeline_sim --mode sequential --frames 50

# Pipelined mode (default)
./pipeline_sim --mode pipelined --depth 3 --frames 100

# Custom seed
./pipeline_sim --seed 123 --frames 200

# Verbose per-frame output
./pipeline_sim --verbose
```

## Output

`pipeline_telemetry.json` with per-stage per-frame latency, E2E latency,
throughput, and bottleneck identification.
