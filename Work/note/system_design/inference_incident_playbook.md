# Inference Incident Playbook

推理与机器人系统的故障处置手册。每种故障按六步闭环记录：**Symptom（现象）→ First Evidence（第一证据）→ Diagnosis（诊断）→ Root Cause（根因）→ Recovery（临时恢复）→ Fix（长期修复）**。

原则：**Recovery 治标（先止血），Fix 治本（防再发），两者都要做。** 只做 recovery 不做 fix，故障会反复发生。

---

## 1. GPU OOM

| 步骤 | 内容 |
|---|---|
| Symptom | 推理报 `CUDA out of memory`，服务重启循环 |
| First Evidence | `torch.cuda.OutOfMemoryError` / `nvidia-smi` 显存接近满 |
| Diagnosis | 定位哪张 tensor 吃满显存（KV cache？激活？权重？） |
| Root Cause | KV cache 无界增长 / batch 过大 / 模型过大 / 显存泄漏 |
| Recovery | 清缓存、减小 batch、重启进程 |
| Fix | 显存预算 + 分页 KV cache（PagedAttention）+ 泄漏修复 |

## 2. Inference Latency Spike

| 步骤 | 内容 |
|---|---|
| Symptom | 推理延迟突增，p99 飙升 |
| First Evidence | 监控 p99 曲线跳变 |
| Diagnosis | 用 trace 定位是哪个 stage（Stage 28）变慢 |
| Root Cause | CPU 抢占（GC/调度）/ thermal throttling / 新模型变慢 / 队列堆积 |
| Recovery | 回滚模型、限流、扩容 |
| Fix | 消除 jitter（CUDA Graph）、实时监控 p99、灰度发布 |

## 3. GPU Utilization Low

| 步骤 | 内容 |
|---|---|
| Symptom | GPU util 低，吞吐上不去 |
| First Evidence | `nvidia-smi` GPU-Util 持续低 |
| Diagnosis | 判断 CPU-bound / launch-bound / memory-bound / 数据管道慢（Stage 1） |
| Root Cause | CPU 喂不饱 / 大量 tiny kernel / H2D 阻塞 / 数据加载慢 |
| Recovery | 加大 batch（如果是 batch 太小） |
| Fix | fusion / CUDA Graph / async H2D / 数据 prefetch |

## 4. GPU 100% but Throughput Low

| 步骤 | 内容 |
|---|---|
| Symptom | GPU util 100%，但 tokens/s 远低于峰值 |
| First Evidence | `nvidia-smi` 100% + 吞吐不达标 |
| Diagnosis | 用 ncu 看 Tensor Core util / DRAM throughput / occupancy（Stage 3） |
| Root Cause | kernel 低效（memory-bound、occupancy 低、非 Tensor Core） |
| Recovery | 换更优 kernel / 换精度 |
| Fix | kernel 优化（Triton/CUTLASS/TensorRT） |

## 5. TensorRT Engine Load Failure

| 步骤 | 内容 |
|---|---|
| Symptom | 服务起不来，engine 加载失败 |
| First Evidence | 启动日志报 deserialize 失败 |
| Diagnosis | engine 文件损坏？版本不兼容？plugin 缺失？ |
| Root Cause | OTA 下载损坏 / TRT 版本升级 / plugin 未注册 |
| Recovery | 回滚到上一个可用 engine |
| Fix | checksum 校验 + 健康检查 + plugin 版本管理（Stage 7/24） |

## 6. CUDA Illegal Memory Access

| 步骤 | 内容 |
|---|---|
| Symptom | `CUDA error: an illegal memory access was encountered`，后续 kernel 全失败 |
| First Evidence | `cudaGetErrorString` 报错码 700 |
| Diagnosis | 定位是哪个 kernel 越界 |
| Root Cause | 索引越界 / 越界读写 / 竞争条件 |
| Recovery | 重置 CUDA context（治标） |
| Fix | compute-sanitizer 定位 + 修复边界检查 |

## 7. NCCL Failure

| 步骤 | 内容 |
|---|---|
| Symptom | 多卡训练/推理通信失败，任务挂起 |
| First Evidence | NCCL 错误日志 / 通信超时 |
| Diagnosis | 是拓扑、网卡、还是配置问题 |
| Root Cause | 网络故障 / NCCL 配置错误 / 拓扑不一致 |
| Recovery | 重试 / 降级单卡 |
| Fix | 网络冗余 + NCCL 配置校验 |

## 8. Model Accuracy Regression

| 步骤 | 内容 |
|---|---|
| Symptom | 新模型上线后业务指标下降 |
| First Evidence | A/B 对比业务指标下降（Stage 21） |
| Diagnosis | 是模型本身退化还是部署问题（量化/精度） |
| Root Cause | 训练退化 / 量化过度 / 输入预处理不一致 |
| Recovery | 回滚到旧模型 |
| Fix | 灰度 + 多指标监控（不只 accuracy） |

## 9. INT8 Accuracy Collapse

| 步骤 | 内容 |
|---|---|
| Symptom | INT8 量化后精度暴跌 |
| First Evidence | 量化前后输出误差大 |
| Diagnosis | 是 outlier 还是 calibration 不足 |
| Root Cause | activation outlier（per-tensor 被拉大）/ calibration 数据不具代表性 |
| Recovery | 回退 fp16 / 换 per-channel |
| Fix | SmoothQuant 迁移 outlier / 更好 calibration（Stage 8） |

## 10. Robot Inference Timeout

| 步骤 | 内容 |
|---|---|
| Symptom | 机器人动作延迟、控制环掉帧 |
| First Evidence | watchdog 超时 / deadline miss 率上升 |
| Diagnosis | 是模型慢还是 CPU/GPU 抢占 |
| Root Cause | batch 过大 / jitter / thermal throttling / 模型超预算 |
| Recovery | fallback 安全动作（Stage 29 watchdog） |
| Fix | 压 p99 到 deadline 内（Stage 14/26 优化） |

## 11. Cloud-Edge Disconnect

| 步骤 | 内容 |
|---|---|
| Symptom | 机器人无法连云端，任务/数据断流 |
| First Evidence | 心跳丢失 / MQTT 断连 |
| Diagnosis | 是网络、证书、还是云端故障 |
| Root Cause | 网络故障 / 证书过期 / 云端宕机 |
| Recovery | 降级到 edge 本地自治 |
| Fix | edge 离线缓存 + 本地自治（Stage 23） |

## 12. OTA Failure

| 步骤 | 内容 |
|---|---|
| Symptom | 模型升级失败，部分机器人版本不一致 |
| First Evidence | OTA 任务失败率 / 版本分布异常 |
| Diagnosis | 是下载、校验、还是加载失败 |
| Root Cause | 网络中断 / 磁盘不足 / 模型损坏 / 不兼容 |
| Recovery | 回滚旧版本 |
| Fix | 断点续传 + checksum + 健康检查 + 灰度（Stage 24/20） |

## 13. Thermal Throttling

| 步骤 | 内容 |
|---|---|
| Symptom | 延迟突然变慢且抖动 |
| First Evidence | 温度超阈值 + 频率下降（Stage 15 tegrastats） |
| Diagnosis | 散热、环境温度、还是持续满载 |
| Root Cause | 散热不足 / 环境高温 / 持续高负载 |
| Recovery | 降负载 / 降频 |
| Fix | 散热改进 + 功耗预算 |

## 14. Memory Leak

| 步骤 | 内容 |
|---|---|
| Symptom | 内存持续增长，最终 OOM |
| First Evidence | 内存曲线单调上升 |
| Diagnosis | 哪个对象没释放 |
| Root Cause | 缓存无界 / 引用未释放 |
| Recovery | 定时重启（治标） |
| Fix | 泄漏检测工具定位 + 修复 |

## 15. Disk Full

| 步骤 | 内容 |
|---|---|
| Symptom | 写日志/数据失败 |
| First Evidence | 磁盘使用率 100% |
| Diagnosis | 什么占满磁盘 |
| Root Cause | 日志无轮转 / 数据无清理 |
| Recovery | 清理日志 + 停写入 |
| Fix | 日志轮转 + 数据生命周期管理 |

## 16. Service Overload

| 步骤 | 内容 |
|---|---|
| Symptom | 请求大量超时，延迟爆炸 |
| First Evidence | p99 飙升 + 队列堆积 |
| Diagnosis | 是流量突发还是下游变慢 |
| Root Cause | 流量超过容量 / 下游故障拖累 |
| Recovery | rate limit + load shedding（Stage 17） |
| Fix | autoscaling + 容量规划 + 过载保护 |

---

## 使用方式

1. 故障发生时，先按 **First Evidence** 快速定位故障类型。
2. 按 **Diagnosis** 确认根因（不要跳过诊断直接重启）。
3. 先执行 **Recovery** 止血，再安排 **Fix** 根治。
4. 每次故障后回填：如果 playbook 没有覆盖，就新增一条；如果诊断步骤不清晰，就细化。
