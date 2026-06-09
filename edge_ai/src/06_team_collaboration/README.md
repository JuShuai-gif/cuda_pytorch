# Cross-Team Performance Contract Validation

Realistic robot system pipeline with actual computation workloads and contract-based SLA validation.

## Pipeline

```
PerceptionModule → PlanningModule → ControlModule
     ↓                  ↓                ↓
  Image resize     A* pathfinding    PID controller
  Sobel edge det   200x200 grid      200-step sim
```

## File Structure

```
06_team_collaboration/
├── CMakeLists.txt
├── README.md
├── interface.h        # PerformanceContract, MeasurementBatch, Violation, module interfaces
├── modules.h          # Module declarations include
├── modules.cpp        # Real implementations:
│                      #   PerceptionModule - image gen → bilinear resize → Sobel edge detection
│                      #   PlanningModule  - A* on 200x200 grid with obstacle generation
│                      #   ControlModule   - PID controller with vehicle dynamics simulation
├── validator.h        # ContractValidator declaration
├── validator.cpp      # Percentile computation, violation checking, stats
└── main.cpp           # Pipeline execution, validation, JSON report generation
```

## 推荐阅读顺序

1. **`interface.h`** — 定义 PerformanceContract、MeasurementBatch、Violation 以及三个模块的抽象接口，为所有其他文件的基础
2. **`modules.h` + `modules.cpp`** — 三个模块的实际实现（PerceptionModule / PlanningModule / ControlModule）
3. **`validator.h` + `validator.cpp`** — ContractValidator：接收合约与测量值，通过百分位数计算判定违规
4. **`main.cpp`** — 最后阅读，实例化模块、运行 100 帧流水线、调用验证器、生成 JSON 报告

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Run

```bash
./contract_demo
```

Outputs `contract_validation_report.json` with per-module contract, measured metrics, and violations.

## Contract Metrics

| Module | P50 Target | Key Metric |
|--------|-----------|------------|
| Perception | ≤50ms | Sobel edge detection throughput |
| Planning | ≤20ms | A* expansions < 20K |
| Control | ≤5ms | PID loop 200 steps @ 10ms |
| E2E | System-wide | Summed pipeline latency |

## Module Implementations

- **PerceptionModule**: Generates a synthetic 640x480x3 image, bilinear resizes to 320x240, runs 3x3 Sobel edge detection (actual convolution loops), and counts edge pixels above threshold as "detections."

- **PlanningModule**: Generates a 200x200 occupancy grid with clustered obstacles, runs A* search from (0,0) to (199,199) with 8-connected movement. Path reconstruction yields trajectory waypoints. Includes expansion limit for timeout detection.

- **ControlModule**: Simulates 200 timesteps (2 seconds) of PID-controlled vehicle dynamics. Applies speed and steering control with actual physics (acceleration, drag, heading integration). Outputs final throttle and steering commands.
