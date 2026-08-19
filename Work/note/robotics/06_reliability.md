# 06｜可靠性：故障注入、Watchdog 与 Incident Playbook

## 本模块解决的问题

规模化后，故障是常态而非例外。本章回答：

```text
九种常见故障各自怎么诊断、恢复、根治？
故障注入（混沌工程）为什么是必要的？
Watchdog 怎么在进程崩溃/推理超时时保住控制环？
```

配套代码：`src/robotics/reliability/`（故障 profile + watchdog）。

---

## 1. 故障的六步闭环

每种故障都按同一个 checklist 处理，把诊断从"靠经验猜"变成"走流程"：

```text
Symptom         现象（用户/监控看到什么）
First Evidence  第一证据（最先在哪个信号上暴露）
Diagnosis       诊断（怎么确认是这种故障）
Root Cause      根因（为什么会发生）
Recovery        临时恢复（先止血，让服务恢复）
Fix             长期修复（根治，防止再发）
```

**关键：Recovery 和 Fix 是两回事**。`recovery` 是"先重启让服务恢复"（治标），`fix` 是"修根因防止再发"（治本）。生产事故最常见的错误是**只做 recovery 不做 fix**，同一个故障反复发生。

---

## 2. 九种故障 profile

`faults.py` 编码了 master prompt 要求的九种故障（完整六要素见代码/playbook）：

| 故障 | 第一证据 | 临时恢复 | 长期修复 |
|---|---|---|---|
| Process Crash | 心跳超时/退出码 | watchdog 重启 | 修段错误/越界 |
| GPU OOM | OOM 异常 | 清缓存/减 batch | 显存预算+分页 KV |
| CUDA Error | 错误码 | 重置 context | compute-sanitizer 定位 |
| Model Load Failure | 启动日志 | 回滚旧版本 | checksum+健康检查 |
| Network Failure | 连接错误 | 重试+熔断 | 冗余链路 |
| Cloud Disconnect | 心跳丢失 | 本地自治降级 | edge 离线缓存 |
| Disk Full | 磁盘 100% | 清理日志 | 日志轮转+生命周期 |
| Memory Leak | 内存曲线单调升 | 定时重启 | 泄漏检测+修复 |
| Thermal Throttling | 温度阈值+降频 | 降负载 | 散热+功耗预算 |

---

## 3. 故障注入（混沌工程）

**为什么要主动注入故障？** 因为：

```text
故障在生产环境迟早会发生。
如果在生产第一次遇到，就是"手忙脚乱地救火"。
如果提前注入、演练过，就是"按 playbook 走流程"。
```

混沌工程（Chaos Engineering）= 主动在生产/预发注入故障（杀进程、断网、占满磁盘、升温），验证系统的恢复能力。本模块的 benchmark 就是雏形：注入 crash 和 hang，观察 watchdog 的响应。

---

## 4. 实测：Watchdog 的恢复行为

**崩溃故障（第 3 次调用崩溃）**：

```text
outcomes: [action, action, safe_stop, action, action, safe_stop]
restarts=2
```

读法：进程在第 3 次调用崩溃 → watchdog 重启 + 返回 `safe_stop`（fallback 动作）→ 新进程恢复 → 第 3 次又崩（同一个 bug）→ 再重启。**watchdog 保证了控制环永不因崩溃而阻塞**——即使 bug 未修，机器人也能"崩溃-恢复"循环，而不是卡死。

**挂起故障（inference 无限等待）**：

```text
outcome: safe_stop  fallbacks=1
```

读法：inference 挂起 → watchdog 超时（0.1s）→ 返回 `safe_stop`。**没有 watchdog，控制环会永远卡在一个挂起的推理上**（Stage 14 的 deadline miss 极端情况）。

### Watchdog 的两个职责

```text
1. 进程崩溃 → 检测（心跳/退出）→ 重启
2. 推理超时 → 检测（timeout）→ fallback（安全动作）
```

fallback 是安全关键：机器人推理失败时，宁可执行一个保守的安全动作（safe_stop），也不要执行一个过时的、可能危险的动作。

---

## 5. Incident Playbook

master prompt 要求最终形成一份 Incident Playbook（见 `note/system_design/inference_incident_playbook.md`）。它是所有故障六要素的汇总表，覆盖 16 种推理/机器人故障。本模块的 `faults.py` 是它的机器可读版本。

---

## 6. 本模块闭环小结

```text
问题：故障是常态，怎么保证服务不因故障崩溃
      ↓
方法论：六步闭环（Symptom→Evidence→Diagnosis→RootCause→Recovery→Fix）
      ↓
手段：故障注入（提前演练）+ Watchdog（自动恢复）+ Playbook（流程化诊断）
      ↓
关键：Recovery 治标、Fix 治本，两者都要做
      ↓
下一步：Stage 30 生产工程（Linux/Docker/CI-CD/配置/权限/Secret/健康检查）
```

要继续就说「继续」。
