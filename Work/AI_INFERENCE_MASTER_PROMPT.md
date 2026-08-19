# AI Infra · GPU Inference / Embodied AI Infra 学习工程（Master Prompt）

本项目是面向两类岗位的工业实战学习工程，与同目录下的 `AI_Train_Infra`（Training Systems）区分：

- **岗位 A：GPU Inference / AI Infra 算法工程师** —— 在 GPU 上最大化推理效率（latency↓ throughput↑ utilization↑ memory↓），面向 LLM/VLM/VLA/Robot Policy/Multimodal。
- **岗位 B：Embodied AI Infra / 机器人 AI Infra 研发工程师** —— 构建 Cloud→Edge→Robot 的完整 AI Infrastructure（serving、云边协同、OTA、数据闭环、可观测、规模化）。

## 学习原则

不是"知识点百科"，每个技术必须走完整闭环：

```text
问题 → 原理 → 性能模型 → Baseline → Benchmark → Profiler
     → 找到瓶颈 → 优化方案 → 工程实现 → 重测 → 生产问题 → 故障排查
```

- 所有优化必须保存 Before/After，解释"为什么变快"，不能只写"提速 30%"。
- 禁止伪造实验数据；拿不到 GPU/Jetson/K8s 就正确实现代码 + 提供运行方法 + 标记 `Not Validated`。
- 统一性能指标：latency（p50/p90/p95/p99）、throughput（QPS/samples/s/tokens/s）、LLM（TTFT/TPOT/ITL）、GPU（utilization/occupancy/Tensor Core/DRAM/L2）、memory、system、机器人（sensor→action latency、jitter、deadline miss）。

## 目录结构

```text
note/   知识讲解（inference / kernel / compression / serving / edge /
        cloud_edge / robotics / observability / system_design / profiling）
src/    代码（与 note 同构，含 baseline/optimized/benchmark/profile/tests/scripts）
```

## 阶段路线图

| Stage | 主题 |
|---|---|
| 1 | 推理性能基础、CUDA 执行模型、Nsight Systems/Compute（当前） |
| 2-3 | CUDA 执行模型深入、nsys/ncu SOP |
| 4-5 | Triton kernel、CUDA custom kernel |
| 6 | Operator fusion |
| 7 | TensorRT（含 dynamic shape、plugin） |
| 8-16 | 量化、剪枝、蒸馏、LLM 推理、vLLM/SGLang、VLM、VLA、batch=1、Edge AI |
| 17-30 | Inference server、生产服务、K8s、autoscaling、灰度、A/B、分布式、云边、OTA、robot runtime、实时性、数据闭环、可观测、可靠性、watchdog、生产工程 |
| System Design | 1000 台机器人模型发布 / GPU 推理集群 / 数据闭环 |
| 最终项目 | A GPU 推理优化报告 / B 机器人推理 Runtime / C 云边 Infra |

## 本机环境锚点

- 硬件：NVIDIA Thor（Jetson 平台，sm_110，20 SM，L2 32MB，**统一内存** ~128GB，14 核 ARM CPU）。
- 软件：CUDA 13.0、nvcc、nsys 2025.3、ncu 2025.3（ncu 需授权，见 `note/profiling/02`）、TensorRT 10.13（C++，Python 未装）。
- Python：`/home/guhaoran/miniconda3/envs/flashrt/bin/python`（torch 2.11.0+cu130、triton 3.6.0）。
- 运行约定：`export PYTHONPATH="$PWD/Work/src"`。

## 能力目标

看到"推理慢"能逐层判断 CPU/GPU/kernel/memory/H2D/shape/batch/quantization/synchronization/queue；看到"GPU 利用率低"能判断 CPU starvation/launch overhead/tiny kernel/sync/H2D/并行度不足；看到"机器人偶尔卡顿"能看 p99/jitter/thermal/GC；看到"1000 台机器人升级"能想到版本/灰度/回滚/健康检查/监控。最终能够从 GPU Kernel 一路分析到机器人端到端 latency，从单模型 benchmark 一路设计到数千台机器人的 AI Infrastructure，并用真实 profiling/benchmark 数据证明方案。
