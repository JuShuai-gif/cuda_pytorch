# System Design 2：GPU Inference Cluster

## 需求

一个 GPU 推理集群，支持 LLM（文本生成）、VLM（视觉语言）、VLA（机器人策略）三类负载，要求：scheduler、routing、batching、autoscaling、monitoring。

---

## 1. 架构总览

```text
                    客户端（API / 机器人）
                         │
              ┌──────────┴──────────┐
              │   API Gateway       │  rate limit + auth（Stage 17）
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │    Scheduler        │  负载类型路由 + 调度（Stage 12/16）
              └──────┬───────┬──────┘
                     │       │
        ┌────────────┘       └────────────┐
        ▼                                 ▼
  LLM 节点池（text）              VLM/VLA 节点池（vision+policy）
  ┌─────────────────┐          ┌─────────────────────┐
  │ GPU x N         │          │ GPU x M             │
  │ vLLM/SGLang     │          │ vision enc + policy │
  │ continuous batch│          │ batch=1 + CUDA Graph│
  └─────────────────┘          └─────────────────────┘
                         │
              ┌──────────┴──────────┐
              │    Monitoring        │  metrics/logs/traces（Stage 28）
              └─────────────────────┘
```

---

## 2. 三类负载的差异（决定架构）

| 负载 | 目标 | 关键特征 | 调度策略 |
|---|---|---|---|
| LLM | max tokens/s | decode memory-bound（Stage 11） | continuous batching |
| VLM | 平衡延迟/吞吐 | vision encoder 重 + LLM | 分阶段 pipeline |
| VLA | low latency + low jitter | batch=1 实时（Stage 14） | 专用节点 + CUDA Graph |

**关键决策：不要用一套调度策略服务三类负载。** LLM 要 continuous batching（吞吐），VLA 要 batch=1 + 低抖动（实时），混在一起会互相拖累。

---

## 3. 核心组件

### Scheduler（路由 + 调度）

```text
路由：按负载类型分到不同节点池
  text 请求 → LLM 池
  image+text → VLM 池
  robot policy → VLA 池

调度（LLM 池内）：continuous batching（Stage 12）
  请求随到随 prefill，decode 动态增删
  KV cache 分页管理（PagedAttention）
```

### Batching

```text
LLM 池：continuous batching（吞吐优先）
VLM 池：vision 阶段 batch + LLM 阶段 dynamic batch
VLA 池：batch=1（实时优先），用 CUDA Graph 消除 launch 开销
```

### Autoscaling（Stage 19）

```text
指标：queue length / pending 请求数（不是 CPU util！）
策略：
  LLM 池：queue 深度 > 阈值 → 扩容
  VLA 池：deadline miss 率 > 阈值 → 扩容（或降级）
冷却：扩容后观察，避免抖动（scale-in 要慢）
```

### Monitoring（Stage 28）

```text
按节点池监控：QPS、latency p50/p99、tokens/s、GPU util、显存
按模型版本监控：灰度/回滚追踪
链路：request_id 贯穿 gateway → scheduler → worker
```

---

## 4. 关键流程

### LLM 请求的生命周期

```text
1. 客户端请求 → gateway（rate limit + auth）
2. scheduler 路由到 LLM 池，分配到一个 worker
3. worker：prefill（compute-bound）→ 加入 decode batch
4. continuous batching：decode 循环，完成则移除，新请求加入
5. 流式返回 token（streaming）
6. 监控记录 TTFT / TPOT / tokens/s
```

### VLA 请求的生命周期

```text
1. 机器人 sensor 数据 → gateway（低延迟通道）
2. 路由到 VLA 池（专用节点，batch=1）
3. preprocess → vision encoder → policy → action（Stage 13/14）
4. CUDA Graph 重放（消除 launch 开销，稳定延迟）
5. 监控 deadline miss 率
6. 超时 → fallback 安全动作（Stage 29）
```

---

## 5. 规模化的关键权衡

| 问题 | 权衡 | 决策 |
|---|---|---|
| 三类负载混跑 | 资源利用率 vs 互相干扰 | 分节点池，按负载隔离 |
| LLM 吞吐 vs 延迟 | batch 大吞吐高但延迟高 | continuous batching 动态平衡 |
| VLA 实时性 | 吞吐 vs 抖动 | 专用节点 + batch=1 + CUDA Graph |
| 扩容指标 | CPU util（错）vs queue（对） | queue length / pending 数 |
| GPU 打满 | 无限接 vs 拒绝 | rate limit + load shedding（Stage 17） |

---

## 6. 用到的 Stage 知识

| 能力 | 来源 |
|---|---|
| Prefill/Decode + KV cache | Stage 11 |
| Continuous batching + PagedAttention | Stage 12 |
| VLM pipeline 分解 | Stage 13 |
| VLA 实时 + batch=1 | Stage 14 |
| Inference server batching | Stage 16 |
| 过载保护 | Stage 17 |
| Autoscaling 指标 | Stage 19 |
| 可观测性 | Stage 28 |

---

## 7. 设计要点总结

```text
1. 按负载类型隔离节点池（LLM/VLM/VLA 的调度策略完全不同）
2. LLM 用 continuous batching，VLA 用 batch=1 + CUDA Graph
3. 扩容指标用 queue length / pending 数，不用 CPU util
4. 过载保护（rate limit + load shedding）和扩容（autoscaling）互补
5. request_id 贯穿全链路，定位延迟到具体 worker
```
