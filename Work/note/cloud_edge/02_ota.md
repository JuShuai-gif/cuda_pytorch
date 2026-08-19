# 02｜OTA / 模型更新：从下载到切换的完整链路与故障处理

## 本模块解决的问题

把新模型安全地更新到成百上千台机器人，不能"直接覆盖文件"。本章回答：

```text
OTA 的完整流程是什么？
下载中断 / 模型损坏 / 磁盘不足 / 加载失败 / 断电各怎么办？
为什么"先安装后切换"是核心设计？
```

配套代码：`src/cloud_edge/ota/`（完整 OTA 流程 + 故障注入）。

---

## 1. OTA 完整流程

```text
Cloud Model Registry
   ↓ 1. 下载（分块，可中断重试）
   ↓ 2. 完整性校验（checksum）
   ↓ 3. 磁盘空间检查
   ↓ 4. 安装到 staging 目录
   ↓ 5. 健康检查（测试推理）
   ↓ 6. 切换版本（原子切换）
Robot 运行新模型
   ↓ 任一环节失败
回滚到旧版本（不切换）
```

核心原则：**只有所有检查都通过，才切换版本**。任何失败都保留旧版本，机器人永远有一个能用的模型。

---

## 2. 实测：五个场景

```text
scenario             result                 final_version
healthy              ok                     v2
download_interrupt   ok                     v2（重试后成功）
corrupted_artifact   corrupted              v1
disk_full            disk_full              v1
load_failure         health_check_failed    v1
```

### 读法

1. **healthy**：正常升级到 v2。
2. **download_interrupt**：下载中断 → 重试 → 成功。**下载中断不是失败，是常态**——真实网络下载几 GB 模型必然有中断，必须支持重试/断点续传。
3. **corrupted_artifact**：checksum 不匹配 → 拒绝。**完整性校验防止"模型下载错了/被篡改/损坏"**，这是模型分发安全的第一道防线。
4. **disk_full**：磁盘不足 → 中止，保留旧版本。**安装前检查磁盘**，避免"装到一半没空间"的中间态。
5. **load_failure**：健康检查失败 → 回滚。**"下载成功 ≠ 能加载"**——模型可能格式错、不兼容当前 runtime，必须用真实推理测试验证。

---

## 3. 断电怎么办（staging + 原子切换）

本模块的代码用 `staging 目录 + 最后切换` 的设计，这是**断电安全**的关键：

```text
安装到 staging 目录（不动正在运行的模型）
   ↓ 全部检查通过
原子切换（current_version 指针从 v1 指向 v2）
```

如果断电发生在**安装中间**（staging 写到一半）：

```text
staging 目录损坏，但 current_version 还是 v1 → 重启后加载 v1（正常）
```

如果断电发生在**切换之后**（已经指向 v2）：

```text
重启后加载 v2（v2 已经完整安装 + 健康检查通过，正常）
```

**所以"先安装后切换"让断电永远不落在"半安装"的中间态**。这是 OTA 和任何"热更新"系统的通用设计（数据库迁移、服务发布、配置更新同理）。

---

## 4. 完整故障清单（master prompt 五问）

| 故障 | 处理 | 本模块 |
|---|---|---|
| 下载过程中断 | 重试 / 断点续传 | `download_fails` → retry |
| 模型损坏 | checksum 校验拒绝 | `corrupt` → checksum mismatch |
| 磁盘不足 | 安装前检查，中止 | `disk_too_small` → abort |
| 升级后不能加载 | 健康检查，回滚 | `load_fails` → rollback |
| 设备断电 | staging + 原子切换 | 见上（理论） |

---

## 5. 与灰度发布（Stage 20）的配合

OTA 是"分发机制"，灰度是"发布策略"，两者配合：

```text
OTA：把模型送到机器人（下载/校验/安装/切换）
灰度：多少机器人先升级（1% → 10% → 100%，异常回滚）
```

实际发布一个模型 = 先灰度（小范围验证）+ 用 OTA 分发（安全更新）。最终项目 C（云边 Infra）会把两者串起来。

---

## 6. 本模块闭环小结

```text
问题：新模型怎么安全更新到成千上万台机器人
      ↓
流程：下载 → 校验 → 磁盘检查 → 安装 → 健康检查 → 切换
      ↓
原则：先安装后切换（staging + 原子切换），失败回滚
      ↓
实测：5 个场景，只有健康升级才切版本
      ↓
下一步：Stage 25 Robot Runtime（Sensor/Data Sync/Model/Action/Controller + ROS 与 non-ROS）
```

要继续就说「继续」。
