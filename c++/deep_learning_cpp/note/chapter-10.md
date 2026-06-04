# 第 10 章：模型部署与推理优化

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 10 章，第 406–448 页。

---

## 目录

1. [章节概述](#章节概述)
2. [文件索引](#文件索引)
3. [编译与运行](#编译与运行)
4. [技术速查](#技术速查)
5. [PDF 完整内容对照](#pdf-完整内容对照)
6. [注意事项](#注意事项)

---

## 章节概述

第 10 章聚焦于将训练好的模型投入生产环境的完整流程：从模型导出（TorchScript/ONNX）、数值一致性验证、微批次调度器设计，到运行时优化（FP16/INT8 量化、剪枝、蒸馏、CUDA Graphs）。后半部分涵盖实时模型服务（HTTP/gRPC）、可观测性（Metrics/Logs/Traces）、漂移检测和安全发布策略（Shadow/Canary/Blue-Green）。

### 六大核心主题

| 主题 | 说明 |
|------|------|
| 模型导出 | TorchScript trace/save/load + ONNX 导出与消费 |
| Parity Check | 验证导出的模型与原生模型在数值上一致（max_abs_diff） |
| 微批次调度 | 有界队列 + 窗口触发 + Promise/Future 异步返回 |
| 推理优化 | FP16、INT8、channels-last、线程调优、CUDA Graphs |
| 实时服务 | HTTP/gRPC 契约、截止时间传播、背压与快速失败 |
| 运维与演进 | 可观测性、漂移检测、安全发布、有目的再训练 |

### 五大挑战

| 挑战 | 说明 |
|------|------|
| 导出精度损失 | trace 过程中的拓扑改变、图融合可能导致数值偏差 |
| 冷启动延迟 | 首次推理触发 JIT 编译、CUDA kernel 编译、分配器初始化 |
| 尾延迟控制 | 微批次窗口放大 p99 延迟，需精细调参 |
| 模型-环境漂移 | 特征分布、预测分布、概念随时间变化，模型退化 |
| 安全发布 | 新模型上线需 shadow→canary→champion 渐进策略，防止回退困难 |

---

## 文件索引

### 一、模型导出 — PDF 第 406–416 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `01_torchscript_export.cpp` | 406–410 | TinyNet 模型定义（C++前端）、`torch::save`/`torch::load` 状态字典序列化、`torch::jit::load` 加载 TorchScript（Python 导出） | LibTorch |
| `02_parity_check.cpp` | 410–416 | 原生模型 vs 保存/加载回环数值一致性验证（max_abs_diff）、确定性检查、输入敏感性分析、预热消除冷启动 | LibTorch |

### 二、微批次调度 — PDF 第 416–424 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `03_micro_batcher.cpp` | 416–424 | 有界队列、时间/大小双触发窗口、Promise/Future 异步、并发控制、背压与快速拒绝 | LibTorch |

### 三、推理优化 — PDF 第 424–432 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `04_optimization_benchmark.cpp` | 424–432 | FP16 推理 (GPU)、channels-last 布局、线程配置 (`set_num_threads`)、cuDNN benchmark、CUDA Graphs 捕获回放 | LibTorch + CUDA |
| `05_onnx_inference.cpp` | 428–432 | ONNX Runtime 消费模型、INT8 Q/DQ 量化图、Graph Optimization Level、线程配置 | ONNX Runtime |

### 四、模型服务 — PDF 第 432–442 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `06_model_serving.cpp` | 432–442 | HTTP POST `/predict` 端点、截止时间传播、输入验证、JSON 序列化、快速失败 | LibTorch |

### 五、运维与演进 — PDF 第 442–448 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `07_drift_detection.cpp` | 442–446 | 数据漂移（PSI/Population Stability Index）、KS 检验、预测分布漂移、漂移告警阈值 | STL |
| `08_safe_release.cpp` | 446–448 | Shadow 部署（流量复制）、Canary（1-5% 切换 + 自动中止）、Blue-Green（双栈切换）、回滚策略 | STL |

---

## 编译与运行

### 环境要求

```bash
# 必需
C++17 编译器（GCC 11+ / Clang 14+）
CMake 3.22+
LibTorch → $HOME/Downloads/libtorch

# 可选
ONNX Runtime → 预编译包于 $HOME/Downloads/onnxruntime-linux-x64-gpu-1.25.1
CUDA Toolkit 12.x（GPU 优化示例需要）
cpp-httplib（header-only，仅 06 需要，已内联提供模拟实现）
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行示例

```bash
# 模型导出
./build/chapter10/torchscript_export
./build/chapter10/parity_check

# 微批次调度
./build/chapter10/micro_batcher

# 推理优化
./build/chapter10/optimization_benchmark
./build/chapter10/onnx_inference       # 需 ONNX Runtime

# 模型服务
./build/chapter10/model_serving

# 运维与演进
./build/chapter10/drift_detection
./build/chapter10/safe_release
```

---

## 技术速查

### 模型导出

| 格式 | 生产方 | 消费方 | 特点 |
|------|--------|--------|------|
| TorchScript (.ts) | `torch::jit::trace_module` | `torch::jit::load` | 保留完整 PyTorch 语义 |
| ONNX (.onnx) | `torch.onnx.export` (Python) | `Ort::Session` | 跨框架交换，支持 TensorRT/OpenVINO |

| 操作 | TorchScript | ONNX Runtime |
|------|------------|--------------|
| 加载 | `torch::jit::load("m.ts")` | `Ort::Session(env, "m.onnx", opt)` |
| 推理 | `m.forward({x}).toTensor()` | `sess.Run(...)` |
| GPU | `m.to(torch::kCUDA)` | `Ort::MemoryInfo::Create("Cuda", ...)` |
| 优化 | 图融合自动 | `SetGraphOptimizationLevel(ORT_ENABLE_ALL)` |

### Parity Check（数值一致性验证）

| 检查项 | 方法 | 阈值 |
|--------|------|------|
| FP32 精度 | `max_abs_diff(native, ts)` | < 1e-4 |
| FP16 精度 | 同上 + 相对误差 | < 1e-2（宽松） |
| INT8 精度 | 交叉熵 / KL 散度 vs FP32 | 校准集 CELoss 涨幅 < 5% |

### 微批次调度器

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `max_batch` | 8-32 (CV), 16-64 (NLP) | 单次推理最大样本数 |
| `max_delay_ms` | 2-8ms | 批次窗口，越大吞吐越高但尾延迟越差 |
| `queue_capacity` | 128-1024 | 满载时快速拒绝，防止无限排队 |

### 推理优化技术

| 技术 | 加速比 | 精度影响 | 适用场景 |
|------|--------|----------|----------|
| FP16 (GPU) | 1.5-2× | 极小 | GPU 推理（Volta+） |
| INT8 量化 | 2-4× | 低（需 QAT） | CPU oneDNN / GPU TensorRT |
| Channels-Last | 1.05-1.3× | 无 | 卷积网络 + CUDA |
| CUDA Graphs | 0.5-2ms 节省 | 无 | 固定形状批次 |
| 结构化剪枝 | 1.5-3× | 中（需微调） | 移除整个通道/头 |
| 知识蒸馏 | 视学生大小 | 低-中 | 小模型替代大模型 |

### 可观测性三大信号

| 信号 | 追踪内容 | 工具 |
|------|----------|------|
| **Metrics** | 延迟 (p50/p95/p99)、吞吐 (QPS)、错误率、队列深度、GPU 利用率 | Prometheus / Grafana |
| **Logs** | 结构化日志、关联 ID、模型版本、输入 schema 版本 | ELK / Loki |
| **Traces** | 端到端链路：解析→预处理→排队→推理→后处理 | Jaeger / Zipkin |

### 漂移检测

| 类型 | 现象 | 检测方法 |
|------|------|----------|
| 数据漂移 (Covariate) | 输入特征分布变化 | PSI > 0.25、KS 检验 |
| 预测漂移 | 输出分布偏移 | 类别直方图变化、校准曲线偏移 |
| 概念漂移 | 标签含义变化 | 业务 KPI 下降 + 离线精度正常 |

### 安全发布策略

| 策略 | 流量比例 | 触发回滚条件 |
|------|----------|-------------|
| Shadow | 100%（仅记录，不生效） | 与生产一致 |
| Canary | 1-5% | p99 延迟 + 20%、错误率 + 5%、业务 KPI 下跌 |
| Blue-Green | 100%（切换后） | 新版本健康检查失败即回切 |

---

## PDF 完整内容对照

以下是 PDF 第 406–448 页的完整纲要，标注了各节对应的实现文件：

| PDF 页 | 内容 | 实现文件 |
|--------|------|---------|
| 406–408 | 章节概述、技术要求、TinyNet 模型定义（Conv→Relu→Conv→GAP→Linear） | `01_torchscript_export.cpp` |
| 408–410 | LibTorch 2.x 模型序列化（`torch::save`/`torch::load`）、TorchScript `.ts` 文件导出（Python `torch.jit.trace`）| `01_torchscript_export.cpp` |
| 410–413 | 数值一致性验证（max_abs_diff）、确定性检查、ONNX Runtime 消费模型 | `02_parity_check.cpp` |
| 413–416 | I/O 契约锁定、布局一致性、预热消除冷启动、线程配置 | `02_parity_check.cpp` |
| 416–419 | 微批次调度器设计（有界队列、窗口触发、Promise/Future） | `03_micro_batcher.cpp` |
| 419–422 | 微批次工作线程实现、批次聚合 (`torch::cat`)、结果分发 | `03_micro_batcher.cpp` |
| 422–424 | 并发控制（背压、快速拒绝、队列深度限制）、云/本地/边缘部署差异 | `03_micro_batcher.cpp` |
| 424–427 | FP16 混合精度推理（TorchScript + CUDA）、channels-last 布局 | `04_optimization_benchmark.cpp` |
| 427–429 | INT8 量化推理（ONNX Runtime Q/DQ）、结构化剪枝部署、知识蒸馏部署 | `05_onnx_inference.cpp` |
| 429–432 | 运行时调优（cuDNN benchmark、线程数、CUDA Graphs 捕获回放）、基准测试工具 | `04_optimization_benchmark.cpp` |
| 432–436 | 实时模型服务三层架构、服务契约（shape/dtype/layout/归一化/截止时间） | `06_model_serving.cpp` |
| 436–439 | HTTP POST `/predict` 端点、gRPC 服务定义、批次策略 | `06_model_serving.cpp` |
| 439–442 | 尾延迟规范（限制一切、预热、分离 I/O 与计算、固定形状）、端到端最小形态 | `06_model_serving.cpp` |
| 442–444 | 可观测性三大信号（Metrics/Logs/Traces）、黄金信号度量 | `07_drift_detection.cpp`（Metrics 部分） |
| 444–446 | 漂移检测（数据漂移 PSI/KS、预测漂移、概念漂移）、漂移笔记本 | `07_drift_detection.cpp` |
| 446–447 | 安全发布策略（Shadow/Canary/Blue-Green）、回滚规则 | `08_safe_release.cpp` |
| 447–448 | 有目的再训练（触发器/流水线/门控/冠军挑战者/人机协作） | `08_safe_release.cpp` |
| 448 | 可靠性/成本/治理、章节问题、下一章预告 | — |

---

## 注意事项

### 外部库依赖

| 文件 | 需要的外部库 | 未安装时的行为 |
|------|-------------|---------------|
| `01_torchscript_export.cpp` | LibTorch | 必需 |
| `02_parity_check.cpp` | LibTorch + ONNX Runtime（可选） | `#ifdef HAS_ONNX` 保护 ONNX 部分 |
| `03_micro_batcher.cpp` | LibTorch | 必需 |
| `04_optimization_benchmark.cpp` | LibTorch + CUDA | GPU 部分 `#ifdef __CUDACC__` 保护 |
| `05_onnx_inference.cpp` | ONNX Runtime | `#ifdef HAS_ONNX` 保护 |
| `06_model_serving.cpp` | LibTorch | 必需 |
| `07_drift_detection.cpp` | 纯 STL | 始终可编译 |
| `08_safe_release.cpp` | 纯 STL | 始终可编译 |

### PDF 中提及但未独立实现的用法

| 知识点 | PDF 页 | 说明 |
|--------|--------|------|
| CUDA Graphs 实际捕获 | 431 | 需要 CUDA 12+ 和固定批次大小，在 `04_optimization_benchmark.cpp` 中提供注释伪代码 |
| gRPC `.proto` 服务定义 | 436–438 | 需要 protobuf + gRPC 库，在 `06_model_serving.cpp` 中以注释形式给出契约定义 |
| cpp-httplib HTTP 服务器 | 436 | HTTP 库，`06_model_serving.cpp` 提供纯 STL 模拟实现 |
| TorchScript 模型服务器 | 438–442 | 需要完整部署环境，在 `06_model_serving.cpp` 中提供架构说明 |
| 漂移笔记本 | 445 | 需要 Prometheus + Jupyter 环境，在 `07_drift_detection.cpp` 中提供指标计算函数 |
| Grafana/Prometheus 集成 | 442–444 | 需要完整可观测性栈，在代码注释中说明集成方式 |

### 其他注意事项

- LibTorch 2.x 移除了 `torch::jit::trace_module`，本书 C++ 端导出 TorchScript 的功能已改用 Python 端 `torch.jit.trace` + C++ 端 `torch::jit::load` 组合。
- `01_torchscript_export.cpp` 演示了 `torch::save`/`torch::load` 状态字典回环，TorchScript `.ts` 文件需从 Python 导出后用 `torch::jit::load` 加载。
- 若模型包含控制流（if/for 等动态分支），应使用 `torch::jit::script` 而非 `trace`（Python 端）。
- ONNX 导出通常在 Python 端进行（`torch.onnx.export`），C++ 端通过 `Ort::Session` 消费。
- 微批次调度器的 `max_delay_ms` 需根据 GPI 尾延迟 SLO 设定，典型值为 2-8ms。
- FP16 推理仅对 GPU 有效，CPU 上 FP16 通常无加速甚至更慢。
- Channels-last 布局 (`NHWC`) 对卷积网络在 NVIDIA GPU 上有约 1.1-1.3× 加速，但在 CPU 上可能无益。
- `cuDNN benchmark` 启用后每个新形状首次调用会有 1-5 秒的搜索开销，适合固定形状场景。
