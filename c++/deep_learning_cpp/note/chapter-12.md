# 第 12 章：部署模型的监控

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 12 章，第 457–513 页。

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

模型进入生产环境后，目标从"离线准确"转为"在线稳定"。关键问题不再是模型在留存测试集上表现好不好，而是它在实时流量下是否保持快速、可靠、低成本且可观测。**监控**让这些期望变得可度量。

### 监控五大核心概念

| 概念 | 说明 |
|------|------|
| SLI（Service Level Indicator） | 可度量的服务质量信号（p95延迟、错误率、ECE、VRAM余量） |
| SLO（Service Level Objective） | 对 SLI 承诺的目标范围（p95 ≤ 120ms、可用性 ≥ 99.9%） |
| SLA（Service Level Agreement） | 基于 SLO 的商业承诺，通常含违约惩罚 |
| 错误预算（Error Budget） | SLO 与完美之间的可接受偏离（99.9% = ~43分钟/月停机） |
| 漂移预算（Drift Budget） | 将错误预算概念扩展到分布变化和校准退化 |

### 三大监控维度

| 维度 | 核心指标 | 典型告警 |
|------|---------|---------|
| 服务响应性 | p50/p95/p99延迟、TTFB/TTFT、QPS、tokens/s | p95 > 120ms 持续10分钟 |
| 在线质量 | ECE、Brier分数、冠军-挑战者分歧率、前导指标 | ECE > 3% 在某个队列持续30分钟 |
| 资源效率 | GPU/CPU利用率、VRAM余量、批次大小分布、H2D/D2H | VRAM < 5%、GPU利用率 < 40% |

### 延迟标签处理策略

| 阶段 | 信号 | 目的 |
|------|------|------|
| 标签到达前 | 类别直方图、置信度十分位数、冠军-挑战者分歧 | 实时检测输出偏移和过度自信 |
| 标签到达后 | 滚动混淆矩阵、校准汇总（ECE/Brier） | 用延迟真值重建精确质量指标 |

---

## 文件索引

### 一、基础百分位计算 — PDF 第 459–460 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `01_percentile_calculation.cpp` | 459–460 | p95 via nth_element（O(n)）、p50/p75/p90/p99、均值 vs 尾部分布的对比 | STL |

### 二、延迟直方图 — PDF 第 461, 475 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `02_latency_histogram.cpp` | 461, 475 | 固定桶直方图（原子计数）、百分位近似（累积桶计数）、批次大小直方图 | STL |

### 三、在线校准与 ECE — PDF 第 462–463, 468–471 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `03_online_calibration_ece.cpp` | 462–463, 468–471 | CalibBin（原子计数器）、ECE 聚合器、Brier 分数、可靠度图表、正确校准 vs 过度自信对比 | STL |

### 四、资源利用率监控 — PDF 第 476–478 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `04_gpu_utilization.cpp` | 476–478 | RSS 读取（/proc/self/status）、CPU 利用率、GPU NVML（mock/真实）、H2D/D2H 跟踪、利用率诊断 | STL（NVML 可选） |

### 五、Prometheus 风格指标 — PDF 第 480–488 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `05_prometheus_metrics.cpp` | 480–488 | Counter/Gauge/Histogram 原语、Labels、Registry（带标签查找）、Prometheus text exposition 渲染 | STL |

### 六、结构化日志 — PDF 第 488–489 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `06_structured_logs.cpp` | 488–489, 465 | ISO8601 时间戳、JSON 结构化日志（含 req_id/trace_id/模型/版本/设备/批次/形状/计时）、按队列诊断日志 | STL |

### 七、追踪与 Span — PDF 第 462, 474, 490–491 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `07_traces_spans.cpp` | 462, 474, 490–491 | RAII Span（含日志+注册表）、完整请求处理（parse→preprocess→queue→infer→postprocess→serialize）、Span 统计聚合、瀑布图诊断 | STL |

### 八、延迟标签处理 — PDF 第 492–495 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `08_delayed_labels.cpp` | 492–495 | CohortKey（紧凑队列标识）、PredRecord（轻量预测记录）、RollingQuality（ECE/Brier 累积）、DelayedLabelJoiner（存储/连接/查询） | STL |

### 九、前导指标 — PDF 第 495–499 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `09_leading_indicators.cpp` | 495–499 | Softmax 熵、Top-2 边际、冠军-挑战者分歧计数器、CohortMonitor（EWMA 前导指标）、按队列诊断 | STL |

### 十、告警与行动 — PDF 第 502–509 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `10_alerts_and_action.cpp` | 502–509 | DriftBudget（累积 ECE + PSI 违规）、AlertPolicy（滞后告警）、Tuning + ActionRouter（运行时调优）、Shadow→Canary→Promote 发布阶梯、策略测试 | STL |

---

## 编译与运行

### 环境要求

```bash
C++17 编译器（GCC 11+ / Clang 14+）
CMake 3.22+
# 本章所有示例均为纯 STL 实现，无外部库依赖
# (NVML 为可选，默认 mock)
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行示例

```bash
# 基础监控
./build/chapter12/percentile_calculation
./build/chapter12/latency_histogram
./build/chapter12/online_calibration_ece
./build/chapter12/gpu_utilization

# C++ 插桩
./build/chapter12/prometheus_metrics
./build/chapter12/structured_logs
./build/chapter12/traces_spans

# 在线质量
./build/chapter12/delayed_labels
./build/chapter12/leading_indicators

# 告警与行动
./build/chapter12/alerts_and_action
```

---

## 技术速查

### 延迟百分位分析

| 指标 | 说明 |
|------|------|
| p50 | 中位数，一半请求更快 |
| p95 | **用户感受的延迟** — 20个请求中有1个在此之上 |
| p99 | 尾延迟 — 突发的队列/GC暂停在此显现 |
| TTFB/TTFT | 流式系统的首字节/首token时间，锚定用户感知 |

### 校准与可靠性指标

| 指标 | 公式 | 解读 |
|------|------|------|
| ECE | Σ (n_b/N) * abs(acc_b - conf_b) | < 0.03 良好; > 0.06 需行动 |
| Brier | (1/N) * Σ (p_i - y_i)² | < 0.25 优于随机; 越小越好 |

### 前导指标速查

| 信号 | 含义 | 行动方向 |
|------|------|---------|
| 熵上升 | 模型置信度降低 | 检查校准，按队列影子评估 |
| Top-2边际下降 | 预测模糊不清 | 阈值可能需调整 |
| 冠军-挑战者分歧上升 | 模型间分歧扩大（漂移信号） | 调查特征分布，等待标签确认 |
| 弃权率激增 | 数据流水线或上游合约问题 | 检查 schema、归一化、未知token |

### 告警设计原则

| 原则 | 说明 |
|------|------|
| 持续窗口 | 告警应在连续 N 个窗口后触发（如 p95 > 120ms 连续 10 分钟） |
| 滞后清除 | 信号返回正常范围后才清除告警（如连续 3 个窗口正常） |
| 按队列切片 | 所有告警应按 region/device/app_version 分解 |
| 可执行映射 | 每个 SLO 违规应有对应的 runbook 操作 |

### 运行时调优参数

| 参数 | 说明 | 典型调整场景 |
|------|------|------------|
| microbatch_delay_ms | 微批次等待超时 | 减少 → 降低 p95 延迟 |
| max_concurrency | 每 GPU 最大并发请求数 | 增加 → 提升 GPU 利用率 |
| score_threshold | 决策阈值 | 降低/升高 → 应对先验漂移 |

---

## PDF 完整内容对照

以下是 PDF 第 457–513 页的完整纲要，标注了各节对应的实现文件：

| PDF 页（书） | PDF 页（文件） | 内容 | 实现文件 |
|-------------|--------------|------|---------|
| 457–459 | 490 | 章节概述、SLI/SLO 定义、TTFB/TTFT、可用性、成本可视性 | `note.md` |
| 459–460 | 491 | p95 百分位计算（nth_element） | `01_percentile_calculation.cpp` |
| 460–461 | 491–492 | SLI/SLO 合约设计：计数器/仪表/直方图/汇总四种指标类型 | `05_prometheus_metrics.cpp` |
| 461 | 492 | 固定桶延迟直方图（Histo 结构） | `02_latency_histogram.cpp` |
| 461–462 | 492 | 性能/质量/资源信号解读 — Span 作用域计时器 | `07_traces_spans.cpp` |
| 462–463 | 492–493 | Little's Law、微批次分析；CalibBin + ECE 原子聚合器 | `03_online_calibration_ece.cpp` |
| 463–464 | 493 | 资源可见性：GPU 利用率、VRAM、H2D/D2H | `04_gpu_utilization.cpp` |
| 464–465 | 493–494 | 监控实践化：结构化请求日志、将 SLI 映射到 playbook | `06_structured_logs.cpp` |
| 465–466 | 494 | SLO 仪表盘总览（质量/延迟/成本）、为什么尾延迟重要（图12.2） | `01_percentile_calculation.cpp`（尾部对比） |
| 466–467 | 494–495 | 吞吐量-延迟-成本前沿（图12.3） | `note.md` |
| 467–469 | 495–496 | 在线准确率与校准：ECE、Brier 分数（图12.4, 12.5） | `03_online_calibration_ece.cpp` |
| 469–471 | 496–497 | CalibBin + ECE + Brier C++ 实现（无原子/有序版本） | `03_online_calibration_ece.cpp` |
| 471–472 | 497–498 | 解读校准数字：队列特定诊断场景 | `03_online_calibration_ece.cpp` |
| 472–474 | 498–499 | 延迟与吞吐量：Span 分解表、单请求剖析（图12.6） | `07_traces_spans.cpp` |
| 474–475 | 499–500 | 延迟插桩：Span 作用域计时器、固定桶直方图、流式TTFT | `07_traces_spans.cpp`, `02_latency_histogram.cpp` |
| 475–476 | 500–501 | 读取数字：延迟队列 vs 计算诊断 | `07_traces_spans.cpp` |
| 476– | 501 | CPU/GPU/内存利用率：RSS、NVML、H2D/D2H | `04_gpu_utilization.cpp` |
| 477–478 | 501–502 | 利用率插桩：RSS 读取、NVML 采样、H2D/D2H 跟踪 | `04_gpu_utilization.cpp` |
| 478 | 502 | 指标驱动行动：SLO 违规→行动映射表 | `10_alerts_and_action.cpp` |
| 479–480 | 502–503 | C++ 插桩概述：指标/日志/追踪关系（图12.7） | `note.md` |
| 480–481 | 503–504 | 工具清单（Prometheus/OpenTelemetry/NVML/spdlog 等） | `note.md` |
| 481–482 | 504 | 设计原则：线程安全、基数控制、固定桶、单一导出线程 | `05_prometheus_metrics.cpp` 注释 |
| 482–484 | 504–505 | Metric 原语（metrics.hpp）：Counter/Gauge/Histogram | `05_prometheus_metrics.cpp` |
| 484– | 506 | Labels 结构：键值对、text 渲染 | `05_prometheus_metrics.cpp` |
| 484–487 | 506–507 | Registry：标签查找、render_metrics_text、get 模板辅助 | `05_prometheus_metrics.cpp` |
| 487–488 | 507–508 | /metrics HTTP 端点（serve_metrics）、指标注册示例 | `05_prometheus_metrics.cpp` |
| 488–489 | 508–509 | 结构化日志：log.hpp（ISO8601辅助、log_json） | `06_structured_logs.cpp` |
| 489–491 | 509–510 | 追踪与Span：trace.hpp（Span含注册表集成、handle_request 示例） | `07_traces_spans.cpp` |
| 491 | 510 | 三步汇总：指标→抓取→存储/仪表盘 | `note.md` |
| 492–493 | 510–511 | 在线质量：延迟标签 → 队列键 + 预测记录 | `08_delayed_labels.cpp` |
| 493–494 | 511 | RollingQuality（ECE + Brier 累积）、DelayedLabelJoiner | `08_delayed_labels.cpp` |
| 495 | 511–512 | 前导指标：Softmax 熵 + Top-2 边际 | `09_leading_indicators.cpp` |
| 496 | 512 | 冠军-挑战者分歧计数器 | `09_leading_indicators.cpp` |
| 497–498 | 512 | 按队列前导指标（图12.10）、CohortLeading + CohortMonitor | `09_leading_indicators.cpp` |
| 500 | 513 | 指标接入：# quality_ece, # lead_entropy_ewma 等 | `09_leading_indicators.cpp` |
| 500–503 | 513 | 可靠度图表按队列（图12.11） | `03_online_calibration_ece.cpp` |
| 503–505 | 514–515 | 告警策略：P95LatencySLO、AvailabilityBudgetBurn、CalibrationDrift、FeatureDriftPSI | `10_alerts_and_action.cpp` |
| 505–506 | 515 | 漂移预算：DriftBudget（ece_cum, psi_breaches） | `10_alerts_and_action.cpp` |
| 506–508 | 516–517 | Tuning + ActionRouter：运行时动态调整 | `10_alerts_and_action.cpp` |
| 508 | 517 | 示例映射：校准违规→温度缩放、特征PSI→数据流水线 | `10_alerts_and_action.cpp`（注释） |
| 508–509 | 517–518 | 触发慢循环（shadow→canary→promote）、策略测试 | `10_alerts_and_action.cpp` |
| 510 | 518 | 章节总结 | `note.md` |
| 510–511 | 518–519 | 章节问题 | — |
| 511–513 | 519–520 | 进一步阅读、参考答案 | — |

---

## 注意事项

### 外部库依赖

本章所有实现均为**纯 STL（C++17）**，无外部库强依赖。

| 文件 | 需要的外部库 | 说明 |
|------|-------------|------|
| 所有文件 | 无（纯 STL） | C++17 标准库即可编译运行 |
| `04_gpu_utilization.cpp` | NVML（可选） | 默认使用 mock 模拟 GPU 数据；安装 `nvidia-ml-dev` 后编译 `-DUSE_NVML=1` 可读取真实 GPU 指标 |

### PDF 中提及但未独立实现的工具

以下知识点在 PDF 中有详细示例，但在本章代码中以引用说明的形式出现：

| 知识点 | PDF 页 | 说明 |
|--------|--------|------|
| OpenTelemetry 集成 | 参考阅读 | `07_traces_spans.cpp` 提供了 Span RAII 实现，可对接 OpenTelemetry |
| Prometheus 抓取 + Grafana 仪表盘 | 480–488 | `05_prometheus_metrics.cpp` 渲染 Prometheus text exposition 格式；实际抓取需外部 Prometheus 服务 |
| NVML GPU 监控 | 476 | `04_gpu_utilization.cpp` 提供了 mock 实现和 NVML 真实代码（`#ifdef USE_NVML`） |
| spdlog 集成 | 参考 | `06_structured_logs.cpp` 提供了最小结构化日志实现，生产环境可替换为 spdlog |

### 其他注意事项

- `04_gpu_utilization.cpp` 中的 `/proc/self/status` 和 `/proc/stat` 读取仅在 Linux 上可用。macOS/Windows 需用对应 API。
- `05_prometheus_metrics.cpp` 中的 Registry 使用 `inline static std::mutex`（C++17），静态初始化顺序安全。
- 所有示例使用 C++17 标准。直方图的原子操作使用 `std::memory_order_relaxed` 以最小化开销。
- 延迟标签连接器（`08_delayed_labels.cpp`）的建议架构：预测时只记录轻量记录，在后台 worker 或 sidecar 中进行标签连接。
- 告警策略中的阈值（ECE 0.03、PSI 0.2、p95 120ms）为演示参考值；生产阈值应根据实际业务需求校准。
