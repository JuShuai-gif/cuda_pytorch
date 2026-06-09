# Robot Perception Pipeline Profiler

Realistic profiling framework for a 5-stage robot perception pipeline
with actual numpy-based data processing.

## Pipeline Stages

```
image_capture -> image_preprocess -> lidar_preprocess -> detection -> postprocess
```

## File Structure

```
05_profiling_monitor/
  timer.py         - TimerContext class (scoped timing) and @profile_func decorator
  tracker.py       - LatencyTracker class (recording, stats, JSON report)
  pipeline_sim.py  - 5-stage perception pipeline with real numpy computation
  main.py          - Entry point, runs pipeline, generates latency_report.json
  README.md
```

## Data Flow

| Stage | Input | Output | Real Work |
|-------|-------|--------|-----------|
| image_capture | - | 1920x1080x3 uint8 image, 100K LiDAR points | Random sensor generation |
| image_preprocess | Raw image | 640x480x3 float32 [0,1] tensor | RGB->YUV, resize, normalize |
| lidar_preprocess | Raw LiDAR | Downsampled voxels, ground plane | Range filter, RANSAC ground removal, voxel grid |
| detection | Image tensor + LiDAR voxels | Image boxes + LiDAR clusters | CNN sliding window, Euclidean clustering |
| postprocess | Detections | Filtered boxes + projections | NMS, 3D bbox, pinhole projection |

## Run

```bash
python main.py
python main.py --frames 100 --seed 123
```

## Output

`latency_report.json` with per-stage statistics:
- mean_us, p50_us, p99_us per stage
- histogram distributions
- bottleneck identification with percentage
- end-to-end latency summary
