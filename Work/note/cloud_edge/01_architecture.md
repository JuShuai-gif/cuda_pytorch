# 01｜云边架构：Cloud → Edge → Robot 的分层设计

## 本模块解决的问题

从单机推理走向**数千台机器人规模化部署**，第一步是设计分层架构。本章回答：

```text
Cloud / Edge Gateway / Robot Runtime 各层做什么？
任务调度、模型分发、数据上传、故障恢复怎么跨层协同？
为什么需要 Edge Gateway 这一层？
```

配套代码：`src/cloud_edge/`（三层组件 + 四个协同流程模拟）。

---

## 1. 分层架构

```text
                  Cloud（云端）
                    │
         ┌──────────┼──────────┐
         │          │          │
     Model       Task       Data
     Registry    Scheduler  Platform
         │
         ▼
     Edge Gateway（边缘网关，聚合一个区域的机器人）
         │
         ▼
     Robot Runtime（机器人端）
         │
         ▼
      AI Model（VLA / Policy）
```

| 层 | 职责 | 本模块组件 |
|---|---|---|
| Cloud | 模型注册、任务调度、数据平台、监控 | `Cloud`（model_registry / schedule / store_data / record_failure） |
| Edge Gateway | 向下分发、向上聚合、本地缓存 | `EdgeGateway`（register / forward / collect） |
| Robot | 执行任务、上报状态、应用模型更新 | `Robot`（execute / apply_model） |

### 为什么需要 Edge Gateway

如果云端直接管数千台机器人：

```text
问题：连接数爆炸（数千长连接）、带宽浪费（每台都连云端）、故障影响大
Edge Gateway 的价值：
  1. 聚合：一个区域的机器人连到本地 gateway，gateway 只维护一条到云端的连接
  2. 缓存：模型、配置缓存在 gateway，机器人就近下载（省云端带宽）
  3. 本地自治：云端断连时，gateway 能本地调度（降级运行）
```

Edge Gateway 是**规模化**的关键——它是"云边协同"里"边"的载体。

---

## 2. 八个关键能力

master prompt 要求的八个能力，各自落在哪一层：

| 能力 | 落点 | 说明 |
|---|---|---|
| Task Scheduling | Cloud → Edge → Robot | 云端调度，edge 转发 |
| Model Distribution | Cloud → Edge | 模型下发到 edge，edge 缓存 |
| Model Update | Edge → Robot | robot 从 edge 拉取新版本 |
| Data Upload | Robot → Edge → Cloud | 数据逐层上传聚合 |
| Remote Diagnosis | Cloud（读 robot 状态） | 云端诊断 robot 故障 |
| Configuration | Cloud → Robot | 配置下发 |
| Permission Control | Cloud | 权限管理 |
| Fault Recovery | Edge 检测 → Cloud 重调度 | 故障发现 + 任务转移 |

---

## 3. 实测：四个协同流程

本模块模拟了四个核心流程（`simulate.py`）：

```text
1. task dispatch    cloud 下发 t1 -> edge 转发 -> robot_0 执行 ok
2. model update     cloud 发布 v2 -> 3 台 robot 全部升级到 v2
3. data upload      robot_0 上报温度 -> edge 聚合 -> cloud 存储
4. fault recovery   robot_1 掉线 -> 任务失败 -> cloud 重调度到 robot_2 -> ok
```

**故障恢复是核心价值**：

```text
robot_1 掉线
   ↓ edge 转发任务失败（offline）
   ↓ cloud 记录故障
   ↓ cloud 重调度任务到健康的 robot_2
   ↓ robot_2 执行成功
```

这一条链展示了"规模化"的本质：**单台机器人的故障不影响整体任务**，靠 edge 检测 + cloud 重调度完成自愈。这是 Stage 29（可靠性）和最终项目 C（云边 Infra）的地基。

---

## 4. 与后续 Stage 的关系

```text
Stage 23（本模块）  分层架构 + 协同流程骨架
Stage 24 OTA        模型更新的深化（下载/校验/回滚）
Stage 27 数据闭环    数据上传的深化（异常挖掘 -> 训练 -> 部署）
Stage 28 可观测性    Remote Diagnosis 的深化（metrics/logs/traces）
Stage 29 可靠性      Fault Recovery 的深化（fault injection + 恢复）
```

---

## 5. 本模块闭环小结

```text
问题：数千台机器人怎么分层管理
      ↓
架构：Cloud（注册/调度/数据/监控）+ Edge（聚合/缓存/自治）+ Robot（执行/上报）
      ↓
流程：任务下发、模型更新、数据上传、故障恢复（重调度）
      ↓
结论：Edge Gateway 是规模化的关键；故障自愈靠 edge 检测 + cloud 重调度
      ↓
下一步：Stage 24 OTA / 模型更新（下载/校验/安装/健康检查/回滚）
```

要继续就说「继续」。
