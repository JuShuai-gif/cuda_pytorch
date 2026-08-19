# 01｜LLM Serving：vLLM 的核心机制（Continuous Batching + PagedAttention）

## 本模块解决的问题

master prompt 的要求是：**不能只会 `vllm serve`，要理解 request scheduler、continuous batching、KV cache manager、paged attention**。本章回答：

```text
为什么静态 batch 的 LLM 服务吞吐低、TTFT 高？
Continuous Batching 怎么让请求"随到随算、随完随走"？
PagedAttention 怎么解决 KV cache 的显存碎片？
```

配套代码：`src/serving/llm_scheduler/`（离散事件模拟器 + KV cache 分配器）。

---

## 1. 本机的诚实约束（为什么用模拟器）

vLLM 0.27.1 有 aarch64 wheel，但安装它会**强制升级 torch 2.11 → 2.13**，破坏本仓库依赖的整个环境（triton、flashrt）。而且本机没有 LLM 权重（需下载几个 GB）。所以：

```text
vLLM/SGLang 实测 → Not Validated（环境约束）
核心机制 → 用自写的离散事件模拟器演示（本模块）
```

这是"正确实现代码 + 标记 Not Validated"原则的应用：**机制的逻辑是平台无关的，用模拟器完全能讲清楚**。

---

## 2. 静态 Batch 的问题

最简单的 LLM 服务：请求攒满一个 batch，一起 prefill + 一起 decode，全部完成后处理下一批。

```text
请求 A（输出 500 token）  ┐
请求 B（输出 10 token）   ├─ batch（一起处理）
请求 C（输出 300 token）  ┘
```

两个致命问题：

1. **TTFT 高**：请求必须等 batch 攒满才被处理（排队延迟）。
2. **GPU 利用率低**：请求 B 输出 10 token 就完了，但 batch 还没结束，B 的位置空着，GPU 空转——直到整个 batch 的请求都完成。

### 本模块实测（模拟器，200 请求）

```text
static（batch=8）      ：TTFT p50 = 3.09s  p95 = 5.74s  吞吐 378 tok/s
continuous（max=32）   ：TTFT p50 = 0.09s  p95 = 0.23s  吞吐 384 tok/s
```

**TTFT 下降 34 倍**——这是 continuous batching 最直接的收益：请求"随到随 prefill"，不用等 batch 满。

---

## 3. Continuous Batching：随到随算、随完随走

核心思想：**把 batch 当成一个动态集合，而不是固定分组**。

```text
循环：
  1. 有新请求到达 → 立即 prefill，加入 decode 集合
  2. 对当前所有 running 请求，一起做一步 decode
  3. 有请求生成 <EOS> → 从集合移除
```

请求的 prefill 和 decode 可以交错进行（新请求 prefill 时，老请求继续 decode）。这样 batch 始终保持"尽量满"，GPU 不空转。

### 为什么 decode 阶段 batch 要满

回到 Stage 11 的结论：**decode 是 memory-bound，吞吐 ∝ batch**（batch 越大，读 KV cache 的代价摊得越薄）。所以 decode 阶段 batch 越满，GPU 的显存带宽利用率越高，吞吐越高。Continuous batching 的核心就是**让 decode batch 尽量满**。

本模块模拟器的吞吐提升不大（378→384），是因为简化模型里 static 已经用了 batch=8；真实系统里 static 的 batch 利用率低得多（请求长度差异大时），continuous 的吞吐提升可达 2-5x。

---

## 4. PagedAttention：KV cache 分块管理

### 问题：contiguous KV cache 的碎片

如果每个请求预留 `max_output_len` 的**连续** KV cache：

```text
请求实际输出 10 token，但预留了 512 token 的空间 → 浪费 502 token 的显存
不同请求的输出长度差异大 → 大量预留空间浪费
```

### PagedAttention 的做法

借鉴操作系统的**分页虚拟内存**：KV cache 按固定块（如 16 token）分配，每个请求按需抓取块，用 block table 记录映射：

```text
contiguous：每个请求预留 max_len 连续空间
paged：     每个请求按需抓 block（16 token 一个），block table 记录
```

### 本模块实测（模拟器，block=16，预算 4096 块）

```text
contiguous：最多服务 128 个并发请求，显存浪费 20%
paged     ：最多服务 159 个并发请求，显存浪费 2%
```

**PagedAttention 用同样的显存服务了 24% 更多的请求，浪费从 20% 降到 2%**——浪费只剩每个请求的"不足一个 block"的尾部碎片。

这正是 vLLM 相比早期系统（预留连续 KV cache）吞吐高 2-4x 的核心原因之一。

---

## 5. vLLM 的其他核心机制（理论）

| 机制 | 作用 |
|---|---|
| **request scheduler** | continuous batching 的调度器（本模块核心） |
| **KV cache manager / block manager** | 显存块的分配/回收（PagedAttention） |
| **prefix cache** | 共享 prompt 前缀只算一次 prefill（system prompt） |
| **CUDA graph** | 消除 decode 的 launch 开销（对应 Stage 2 的 CUDA Graph） |
| **worker / tensor parallel** | 多卡切分模型，每个 worker 跑一个分片 |
| **SGLang 的 radix attention** | 更激进的 prefix 复用（树形前缀缓存） |

理解了本模块的 continuous batching + PagedAttention，就理解了 vLLM 的"调度层"和"显存层"两大支柱。

---

## 6. 工业故障与定位

| 现象 | 可能原因 |
|---|---|
| TTFT 高 | prefill 排队（batch 满才处理）/ 长 prompt |
| 吞吐低但 GPU 利用率高 | decode batch 小、KV cache 碎片 |
| 显存 OOM | KV cache 预留过多（contiguous）或 block 耗尽 |
| token 流抖动（ITL 大） | prefill 抢占 decode（新请求 prefill 时 decode 暂停） |

---

## 7. 本模块闭环小结

```text
问题：LLM 服务为什么不能简单静态 batch
      ↓
机制：continuous batching（动态 batch）+ PagedAttention（分块 KV cache）
      ↓
实测（模拟器）：TTFT 34x 下降；paged 服务 +24% 请求、浪费 20%→2%
      ↓
约束：vLLM 本机装不上（torch 升级），机制用模拟器演示，vLLM 实测 Not Validated
      ↓
下一步：Stage 13 VLM 推理（Image → Vision Encoder → LLM 的性能模型）
```

要继续就说「继续」。
