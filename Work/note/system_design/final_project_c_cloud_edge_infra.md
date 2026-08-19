# 最终项目 C：Robot Cloud-Edge Infra

## 目标

构建完整的云边 Infra：Cloud（Model Registry / Task Scheduler / Data Service / Monitoring / OTA）→ Edge Runtime → Robot，实现模型部署、版本、远程更新、回滚、健康检查、metrics/logs/trace、故障恢复。

## 架构

```text
Cloud
├── Model Registry（版本管理）
├── Task Scheduler（任务调度）
├── Data Service（数据平台）
├── Monitoring（metrics/logs/traces）
└── OTA Service（远程更新）
        │
        ▼
Edge Runtime（转发 / 聚合 / 缓存）
        │
        ▼
Robot（执行任务 / 上报数据 / 应用更新）
```

本模块 `src/projects/cloud_edge_infra/` **整合**了前面各 Stage 的组件，而不是重新实现。

## 整合的组件（来自哪些 Stage）

| 组件 | 来源 |
|---|---|
| Cloud/Edge/Robot 三层架构 | Stage 23（cloud_edge/architecture.py） |
| OTA（下载/校验/升级） | Stage 24（cloud_edge/ota/ota.py） |
| Metrics/Logs/Trace | Stage 28（observability/observability.py） |
| 故障注入 + 恢复 | Stage 29（reliability/） |

## 端到端生命周期演示

```text
1. publish_and_ota("v2")：发布模型 v2 → OTA 到 3 台机器人
   结果：robot_0/1/2 全部升级到 v2（ota_success_rate 100%）

2. run_task(t1)：调度任务 → edge 转发 → robot 执行
   结果：task_success_rate 记录 + trace span（cloud.schedule / edge.forward）

3. inject_offline("robot_1")：故障注入（机器人掉线）
   结果：任务失败 → 故障记录 ['robot_1:offline'] → 可触发重调度
```

## 核心设计决策

1. **分层整合而非重写**：云边 Infra 是前面各 Stage 组件的编排，不是新造轮子。

2. **每个操作都带 trace + metrics + log**：模型发布、任务调度、故障恢复都走三体系（Stage 28），任何问题可追溯。

3. **故障恢复是内置能力**：机器人掉线 → 检测 → 记录 → 重调度（Stage 23 的 fault recovery + Stage 29 的故障闭环）。

4. **版本是贯穿维度**：模型版本（registry）、机器人版本（monitor）、OTA 结果（metrics）都是带 version 标签的。

## 三个最终项目的关系

```text
项目 A（GPU 优化）   → 单模型在 GPU 上跑到最快
项目 B（Robot Runtime）→ 模型在机器人端低延迟执行
项目 C（云边 Infra）  → 模型和任务在云边规模化管理
```

三者合起来，就是 master prompt 最终目标的两条能力线：

```text
纵向：Model → Graph → Operator → CUDA Kernel → GPU 架构（项目 A/B）
横向：Cloud → Inference Service → Edge Runtime → Model → Robot（项目 C）
```

## 复现

```bash
export PYTHONPATH="$PWD/Work/src"
python -m projects.cloud_edge_infra.benchmark --output /tmp/cloud_edge_infra.json
```
