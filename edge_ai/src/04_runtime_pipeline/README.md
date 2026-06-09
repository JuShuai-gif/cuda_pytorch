# Robot Task Graph Runtime

Executes a 7-node robot processing pipeline on a DAG task scheduler with real data processing: sensor capture, image/LiDAR preprocessing, detection, Kalman tracking, A* path planning, and PID control.

## File Structure

```
04_runtime_pipeline/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── task_node.h              # TaskNode struct, PipelineContext, data structures
├── task_executor.h          # Timer, TaskGraphExecutor, node function declarations
├── task_executor.cpp        # DAG executor + 7 robot node implementations + JSON export
├── double_buffer.h          # DoubleBuffer< T > template (ping-pong buffer)
├── pipeline.h               # BoundedQueue (thread-safe bounded queue)
├── pipeline.cpp             # BoundedQueue implementation
└── main.cpp                 # Robot pipeline + double buffering + pipeline parallelism demos
```

## Robot Task Graph

### 7-Node DAG Pipeline

```
sensor_capture ─┬→ image_preprocess ─┬→ detection → tracking → planning → control
                └→ lidar_preprocess ─┘
```

| Node | Depends On | Processing |
|------|-----------|------------|
| `sensor_capture` | (none) | Generate synthetic 1920x1080x3 camera image + 100K-point 64-ring LiDAR scan |
| `image_preprocess` | sensor_capture | RGB→grayscale (BT.601), bilinear resize 1920x1080→640x480, normalize [0,1] |
| `lidar_preprocess` | sensor_capture | Range filter (0.5m–100m), voxel grid downsampling (10cm) |
| `detection` | image_preprocess, lidar_preprocess | 3x3 Sobel edge detection (sliding window) + NMS + Euclidean point cloud clustering |
| `tracking` | detection | Kalman filter (constant velocity, 4-state) predict + update for each object |
| `planning` | tracking | A* search on 100x100 random occupancy grid (30% obstacles) |
| `control` | planning | PID lateral controller (cross-track error → steering) + longitudinal controller (speed → throttle/brake) |

### Data Structures

| Struct | Purpose |
|--------|---------|
| `TaskCameraImage` | Raw camera frame (uint8 RGB) |
| `TaskPointCloud` / `TaskPoint3D` | LiDAR scan with ring ID |
| `TaskPreprocessedImage` | 640x480 normalized float grayscale |
| `TaskDetections` / `TaskDetectionBox` | 3D bounding boxes with class/confidence |
| `TaskTrackingResult` / `TaskKalmanTrack` | Kalman track states (x,y,vx,vy + 4x4 cov) |
| `TaskTrajectory` / `TaskWaypoint` | A* path waypoints |
| `TaskControlCommand` | throttle/brake/steering output |
| `PipelineContext` | All intermediate pipeline data (shared across nodes) |

### DoubleBuffer< T >

Ping-pong buffer template used to decouple sensor capture (producer) from image preprocessing (consumer):
- Producer writes to buffer A while consumer reads from buffer B
- Swap after both complete
- Producer never overwrites data consumer is reading

### Output

Writes `task_graph_profile.json` with:
- Per-node execution time (us, ms), dependencies
- Critical path analysis (longest dependency chain)
- Total sequential vs. parallel wall time
- Speedup and parallelism efficiency

## 推荐阅读顺序

1. **`task_node.h`** — 所有数据结构的定义（TaskCameraImage、PipelineContext 等），是 task_executor 中所有节点操作的基础
2. **`pipeline.h` + `pipeline.cpp`** — 自包含的线程安全 BoundedQueue 工具类，被 main 中的流水线并行演示使用
3. **`double_buffer.h`** — 自包含的 Ping-pong 缓冲模板，被 main 中的双缓冲演示使用
4. **`task_executor.h` + `task_executor.cpp`** — DAG 执行器逻辑 + 7 个机器人节点函数实现，核心演示所在
5. **`main.cpp`** — 最后阅读，调用 run_robot_pipeline()、run_double_buffer_demo() 和 run_pipeline_demo()

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

## Run

```bash
# Default: 4 threads
./task_graph_demo

# Custom thread count
./task_graph_demo --threads 8

# Help
./task_graph_demo --help
```
