# System Design 1：1000 台机器人模型发布系统

## 需求

把新模型安全地发布到 1000 台异构机器人（不同硬件、不同网络、部分离线），要求：model registry、versioning、OTA、rollback、monitoring、permission、gray release。

---

## 1. 架构总览

```text
                         Cloud
  ┌─────────────────────────────────────────────────────┐
  │ Model Registry（版本管理）      Permission（权限）      │
  │ Gray Release Controller（灰度）  Monitoring（发布监控）  │
  │ OTA Service（下发/校验/回滚）                          │
  └──────────────────────┬──────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │    Edge Gateway     │  ← 区域聚合 + 模型缓存
              └──────────┬──────────┘
                         │
        ┌────────┬───────┼────────┬────────┐
        │        │       │        │        │
    Robot 1  Robot 2  ...  Robot N（1000 台，部分离线）
```

---

## 2. 核心组件

### Model Registry

```text
职责：模型 artifact 的存储 + 版本管理

artifact 内容：
  model 文件（.engine / .pt / .safetensors）
  checksum（SHA256，防损坏/篡改）
  metadata（框架、dtype、量化方式、兼容的 runtime 版本）
  config（输入 shape、预处理参数）

版本策略：语义化版本 v{major}.{minor}.{patch}
  v1.0.0 → v1.1.0（兼容升级，可灰度）
  v2.0.0（不兼容，需全量切换）
```

### OTA Service（见 Stage 24）

```text
下发流程：
  registry → artifact 下载 → checksum 校验 → 磁盘检查
  → staging 安装 → 健康检查 → 原子切换 → 上报结果
失败回滚：任一环节失败，保留旧版本（staging + 原子切换）
```

### Gray Release Controller（见 Stage 20）

```text
灰度：1% → 10% → 50% → 100%
判据：error rate / latency / task success rate 与基线对比
异常：自动回滚到上一个稳定版本
```

### Permission（权限控制）

```text
模型发布权限分级：
  algorithm（算法工程师）→ 上传模型到 staging
  reviewer（审核）→ 审批灰度放量
  infra（运维）→ 执行发布 / 回滚
  viewer（只读）→ 查看版本和监控

实现：RBAC，发布操作需要审批流（approval chain）
```

### Monitoring（发布监控）

```text
按版本维度监控：
  每版本的 task success rate、error rate、latency
  版本分布（多少台机器在 v1、多少台在 v2）
  灰度进度（当前放量百分比）
  回滚事件（何时回滚、原因）
```

---

## 3. 关键流程

### 发布流程（端到端）

```text
1. 算法上传 v1.1.0 到 registry（staging）
2. 自动检查：checksum、格式、兼容性
3. reviewer 审批
4. 灰度：1% 机器人（10 台）先升级
5. 观察 24h：监控 task success rate / error rate
6. 无回归 → 10% → 50% → 100%
7. 有回归 → 自动回滚到 v1.0.0
```

### 离线机器人的处理

```text
问题：1000 台里有部分离线（关机/网络断）

策略：
  1. Edge Gateway 缓存新模型（就近下载）
  2. 机器人上线后主动拉取最新版本（pull 模型）
  3. 版本落后检测：机器人上报版本号，云端对比，落后则下发升级任务
  4. 超期未升级：告警 + 人工介入
```

### 回滚流程

```text
触发：灰度监控发现回归 / 人工回滚指令
执行：
  1. 标记目标版本为 unstable
  2. 下发回滚命令（机器人降级到上一个稳定版本）
  3. 确认所有机器人回滚完成
  4. 记录回滚事件（供事故分析）
```

---

## 4. 规模化的关键权衡

| 问题 | 权衡 | 决策 |
|---|---|---|
| 1000 台同时下载 | 云端带宽 vs 发布速度 | Edge 缓存 + 分批（灰度天然分批） |
| 异构硬件 | 一个模型 vs 多模型 | 按硬件分组，每组独立 artifact |
| 离线机器人 | 强制 vs 容忍 | 上线后拉取 + 落后告警 |
| 回滚速度 | 快 vs 稳 | 回滚比升级优先（回滚走快速通道） |

---

## 5. 用到的 Stage 知识

| 能力 | 来源 |
|---|---|
| OTA 下载/校验/回滚 | Stage 24 |
| 灰度发布 + 监控 | Stage 20 |
| 幂等（命令唯一 ID） | Stage 22 |
| 云边分层 + Edge 缓存 | Stage 23 |
| 可观测性（版本维度监控） | Stage 28 |
| 故障恢复（回滚） | Stage 29 |

---

## 6. 设计要点总结

```text
1. 版本是发布系统的一等公民（registry + 语义化版本 + 维度监控）
2. 灰度是默认路径，不是可选（1%→10%→50%→100% + 自动回滚）
3. 离线是常态不是例外（1000 台必有离线，pull 模型 + 落后告警）
4. 回滚比升级优先（回滚走快速通道）
5. 权限分级 + 审批流（发布是高风险操作，不能人人可做）
```
