# Robot Perception → Planning → Control Pipeline

Processes realistic robot sensor data through a 3-stage perception → planning → control pipeline. Uses actual image/LiDAR processing, Kalman filters, cubic spline trajectory generation, and PID control.

## File Structure

```
01_system_pipeline/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── pipeline_config.h           # Data structures: CameraImage, PointCloud, DetectionBox,
│                               #   Trajectory, ControlCommand, KalmanTrack, pipeline flow types
├── pipeline_stage.h            # Stage function declarations (perception, planning, control)
├── pipeline_stage.cpp          # Stage implementations with real data processing
├── pipeline_executor.h         # Sequential + Pipelined executor declarations
├── pipeline_executor.cpp       # Sequential + pipelined executor implementations
├── latency_stats.h             # Latency statistics collector (thread-safe)
├── latency_stats.cpp           # Statistics + JSON metrics export
└── main.cpp                    # CLI entry point
```

## Data Pipeline

### Data Structures

| Struct | Fields | Description |
|--------|--------|-------------|
| `CameraImage` | width, height, channels, timestamp_ns, data (uint8 vector), encoding | Raw camera frame (default 1920x1080x3 rgb8) |
| `Point3D` | x, y, z, intensity, ring | Single LiDAR point |
| `PointCloud` | timestamp_ns, points vector | LiDAR scan (100K points, 64 rings) |
| `DetectionBox` | class_id, confidence, x/y/z, w/h/d, vx/vy | 3D bounding box with velocity |
| `Detections` | timestamp_ns, frame_id, boxes vector | Per-frame detection results |
| `Trajectory` | timestamp_ns, waypoints vector | Cubic spline trajectory |
| `Waypoint` | x, y, t (parametric), v (target speed) | Single trajectory waypoint |
| `ControlCommand` | throttle (0-1), brake (0-1), steering (-1..1), timestamp_ns | Vehicle control output |
| `KalmanTrack` | track_id, class_id, x/y/vx/vy, P(4x4 cov), age, missed | Tracked object state |

### Processing Stages

**Perception Stage:**
1. Generate synthetic camera image (1920x1080x3 random pixels)
2. Camera preprocessing: RGB → grayscale (ITU-R BT.601), bilinear resize to 640x480, normalize to [0,1]
3. Generate synthetic LiDAR point cloud (100K points, 64 rings, realistic ring angles ±25° to +15°)
4. LiDAR preprocessing: range filter (0.5m–100m), voxel grid downsampling (10cm)
5. Detection generation from processed data density

**Planning Stage:**
1. Kalman filter initialization from detections (constant velocity model, 4-state)
2. Kalman predict step (state transition F, process noise Q)
3. Kalman update step (measurement z=[x,y], 2x2 matrix inversion for innovation covariance)
4. Time-to-collision (TTC) computation for each tracked object
5. Cubic spline trajectory generation through control points (natural spline, Thomas algorithm)

**Control Stage:**
1. Cross-track error computation (signed distance to trajectory segment)
2. Lateral PID controller → steering command
3. Longitudinal PID controller → throttle/brake commands

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

## Run

```bash
# Pipelined mode (default, 3 frames in flight)
./pipeline_sim

# Sequential mode
./pipeline_sim --mode sequential

# Custom configuration
./pipeline_sim --depth 5 --frames 200 --verbose

# Help
./pipeline_sim --help
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | pipelined | `sequential` or `pipelined` |
| `--depth` | 3 | Pipeline depth (frames in flight) |
| `--frames` | 100 | Total frames to process |
| `--verbose` | false | Print per-frame timing to stderr |

## Output

Writes `pipeline_metrics.json` with:
- Aggregate stats: mean, stddev, P50, P99, max latency per stage (nanoseconds)
- Breakdown: perception (image preprocess, lidar preprocess, detection), planning, control
- End-to-end latency and throughput (FPS)
