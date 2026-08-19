# 07｜生产工程：从 Research Script 到 Production Service

## 本模块解决的问题

前面所有模块的代码大多是"能跑的脚本"。生产环境的代码要"能被部署、监控、恢复"。本章回答：

```text
research script 和 production service 差在哪？
配置 / 日志 / 健康检查 / 优雅关闭 / Secret 各怎么工程化？
Python 和 C++ 在推理系统里怎么分工？
```

配套代码：`src/serving/production/`（config + service + Dockerfile + 脚本）。

---

## 1. Research Script vs Production Service

| 维度 | Research Script | Production Service |
|---|---|---|
| 配置 | 硬编码 `batch=8` | 环境变量/配置文件（12-factor） |
| 日志 | `print()` | 结构化日志（JSON，可采集） |
| 健康 | 无 | health check（探活/就绪） |
| 关闭 | 直接 kill | 优雅关闭（排空在途请求） |
| 密钥 | 写死在代码 | 环境变量注入，日志脱敏 |
| 部署 | `python xx.py` | Docker 镜像 + 编排 |

**演进路径**：本模块就是示范——把 Stage 16 的 inference server 包成 production service。

---

## 2. 生产工程的五个必须

### 配置管理（12-factor）

```python
# 硬编码（research）
max_batch = 8

# 12-factor（production）：同一镜像，不同环境不同配置
max_batch = int(os.environ.get("MAX_BATCH", "8"))
```

好处：**同一镜像在 dev/staging/prod 跑不同配置**，不用改代码重新构建。

### 结构化日志

```python
# research
print("服务启动了")

# production：JSON lines，日志采集器（ELK/Loki）能解析
print(json.dumps({"ts": ..., "level": "INFO", "msg": "service_started", "config": {...}}))
```

好处：日志可被检索、聚合、告警（"过去 5 分钟有多少 error"）。

### 健康检查

```python
def health(self):
    return {"status": "ok", "version": ...}
```

好处：编排系统（K8s liveness/readiness probe）能判断"要不要路由流量/重启"。

### 优雅关闭

```python
# 收到 SIGTERM → 停止接新请求 → 排空在途请求 → 退出
```

好处：发布/扩缩容时不丢请求、不产生半成品状态。

### Secret 管理

```python
# 错误：密钥写死，还打进日志
api_key = "abc123"
print(api_key)

# 正确：环境变量注入，日志脱敏
api_key = os.environ.get("API_KEY", "")
print("***")
```

好处：密钥不进代码仓库、不进日志、不进镜像层。

---

## 3. Docker / CI-CD（理论 + 本模块模板）

本模块提供了 `Dockerfile`（模型运行时挂载，不打进镜像）和 `scripts/`（启动 + 健康检查脚本）。

```text
Docker：把代码 + 依赖打成可复现的镜像，任何环境跑一样的结果
CI-CD：提交 → 测试 → 构建镜像 → 部署（本工程尚未接入真实 CI，标记 Not Validated）
K8s：编排（Stage 18 已跳过）
```

**关键实践：模型不打进镜像**。镜像 = 代码 + 依赖（不变），模型 = 运行时挂载（多变）。这样升级模型不需要重新构建镜像（呼应 Stage 24 OTA 和模型版本管理）。

---

## 4. Python vs C++ 的分工（Stage 44 总结）

master prompt 要求：Python 做 model/benchmark/service/automation，C++ 做 runtime/low-latency/CUDA/TensorRT。本工程的实践：

| 层 | 语言 | 本工程模块 |
|---|---|---|
| 模型原型 / benchmark / 服务编排 | Python | inference/、serving/、compression/ |
| CUDA kernel / 低延迟 runtime / TensorRT | C++ | kernel/cuda_core/、inference/tensorrt/ |
| 机器人集成（ROS、传感器） | C++ | （最终项目 B） |

**"Python prototype → C++ runtime"** 是本工程的最终形态之一（最终项目 B）。Python 快速验证，C++ 落地低延迟。

---

## 5. 本模块闭环小结

```text
问题：脚本怎么变成能部署的服务
      ↓
五个必须：配置化 + 结构化日志 + 健康检查 + 优雅关闭 + Secret 管理
      ↓
部署：Docker 镜像（模型挂载，不进镜像）+ 编排
      ↓
分工：Python（原型/服务）+ C++（runtime/kernel/TensorRT）
      ↓
下一步：System Design 专项（3 个系统设计）+ 最终项目 A/B/C
```

要继续就说「继续」。
