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
