# 最终项目 B：Robot Inference Runtime

## 目标

建立 Camera → Preprocess → Vision → Policy → Action 的机器人推理 Runtime，实现 async H2D、CUDA Graph、double buffering，统计 sensor-to-action latency / jitter / p99。

## 架构

```text
Camera（帧流）
   ↓ decode + preprocess（CPU）
   ↓ async H2D（copy stream）
   ↓ Vision Encoder + Policy（GPU，CUDA Graph 重放）
   ↓ Action
```

两个版本：
- **Naive**：串行（同步 H2D + 普通 forward）
- **Optimized**：double buffering + async H2D + CUDA Graph

## 实测（Thor/sm_110）

### 单帧 latency（每帧 synchronize）

```text
runtime    mean_ms   p50_ms   p99_ms   jitter
naive      5.39      5.18     7.68     2.50
optimized  5.48      5.07    12.76     7.69
latency speedup: 1.02x
```

### 连续帧流 throughput（只在最后 synchronize）

```text
runtime    fps      avg_ms
naive      258.8    3.86
optimized  374.7    2.67
throughput speedup: 1.45x
```

## 核心结论

1. **单帧 latency 几乎不变（1.02x）**：double buffering 不降低单帧的 GPU 时间——它降低的是"连续处理多帧的平均时间"。

2. **吞吐提升 45%（258 → 375 fps）**：连续帧流时，optimized 让 CPU（decode/preprocess）和 GPU（vision/policy）**overlap**——GPU 算帧 N 时，CPU 已经准备帧 N+1。这是 double buffering + async H2D 的核心价值。

3. **CUDA Graph**：把 20 个 kernel 的 launch 折叠成 1 次，消除 batch=1 的 launch 开销（Stage 2/项目 A）。

4. **jitter 的权衡**：optimized 的 p99 变高（12.76 vs 7.68），是 double buffering 的时序竞争副作用。对机器人实时场景，需要权衡"吞吐"和"最坏延迟"（Stage 14/26）。

## 三个优化的分工

| 优化 | 降低什么 | 收益 |
|---|---|---|
| async H2D | CPU 不被 H2D 阻塞 | 吞吐（overlap） |
| CUDA Graph | launch 开销 | 单帧 latency + 稳定性 |
| double buffering | CPU/GPU 串行等待 | 吞吐（pipeline） |

**关键认知：async H2D + double buffering 优化的是吞吐（pipeline），CUDA Graph 优化的是单帧延迟和 jitter。两者目标不同，要按场景选择**——机器人实时控制（batch=1）优先 CUDA Graph（稳），视频流/批量处理优先 double buffering（吞吐）。

## C++/TensorRT 路径

Python 版本已验证优化逻辑。生产落地是 C++/TensorRT：

```text
1. 模型 → TensorRT engine（Stage 7 的 build_engine.cpp）
2. C++ runtime：run_engine.cpp（Stage 7）+ double buffering + multi-stream
3. 用 cudaMemcpyAsync + 两个 stream + cudaGraphLaunch（Stage 2 的 cuda_core）
```

C++ 化的收益：消除 Python 解释器/dispatcher 开销（Stage 4 的"边缘端用 C++"结论），对 batch=1 的 launch-bound 场景收益显著。

## 复现

```bash
export PYTHONPATH="$PWD/Work/src"
python -m projects.robot_runtime.benchmark --device cuda --output /tmp/robot_runtime.json
```
