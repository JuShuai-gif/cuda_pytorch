# 09｜LLM 推理：Prefill vs Decode 与 KV Cache

## 本模块解决的问题

LLM 推理和其他模型推理最大的不同：它是**自回归**的，一次生成一个 token。理解"为什么 LLM 推理难优化"，必须先理解两阶段的本质差异。本章回答：

```text
Prefill 和 Decode 为什么性质完全不同？
KV Cache 解决什么问题、引入什么问题？
TTFT / TPOT / ITL 各指什么，由哪个阶段决定？
为什么 Decode 阶段 GPU 利用率低却是主要成本？
```

配套代码：`src/inference/llm/`（`roofline.py` 理论 + `model.py`/`benchmark.py` 实测）。

---

## 1. 自回归生成的两个阶段

LLM 生成一个回答的过程：

```text
输入 prompt（S 个 token）
   ↓ Prefill：一次性处理全部 S 个 token，算出第一个新 token 的概率
   ↓ Decode：每次用"已生成的 token + KV cache"算下一个 token，循环直到 <EOS>
输出（N 个 token）
```

| | Prefill | Decode |
|---|---|---|
| 处理什么 | 全部 prompt（S 个 token 并行） | 每次 1 个新 token |
| 计算量 | O(S²) 总量，但并行 | 每次 O(S)，串行循环 N 次 |
| 瓶颈 | compute（GEMM 吃满 Tensor Core） | memory（读整个 KV cache） |
| 决定指标 | TTFT | TPOT / ITL |

---

## 2. KV Cache：省计算，但吃显存

### 没有 KV Cache 会发生什么

Decode 每一步，attention 都要对**所有历史 token** 算 Q/K/V。第 t 步要对前 t 个 token 重新算 K/V，总计算 O(S²)——生成 N 个 token 的代价是 O(S*N) 的重复计算。

### KV Cache 的做法

把每个 token 算出来的 K、V **缓存起来**，Decode 时只算当前新 token 的 K/V，历史 K/V 直接从缓存读：

```text
第 t 步：只算新 token 的 q, k, v（O(1) 个 token 的投影）
       + 读缓存的历史 K/V（t 个 token）
       + attention(q, [k_cache; k_new], [v_cache; v_new])
```

这样生成 N 个 token 的投影计算是 O(S+N)，而不是 O(S*N)。

### 代价：显存

```text
KV cache = 2 × L × S × d × 2 bytes × B   （2 是 K/V，2 是 fp16，L 层，S 序列长）
```

本机实测（单层 d=1024，batch=1，fp16）：seq 128→8192，KV cache 0.5MB→33.6MB。**一个真实 7B 模型（L=32, d=4096）在 seq=2048、batch=8 下，KV cache 有 ~4GB**——这往往比权重还大。这就是 LLM 推理显存的主要去向，也是 PagedAttention 要解决的核心问题。

---

## 3. 实测：算术强度的两阶段分界

用单层 transformer（d=1024，batch=1）实测，关键看**算术强度**（FLOPs/byte）：

```text
seq    prefill_us  decode_us  p_tflops  d_tflops   p_ai      d_ai   kv_mb
128      405us      178us      8.11      0.14      126.7     1.0    0.5
512     1278us      168us     10.92      0.16      493.0     1.0    2.1
2048    7247us      432us      9.48      0.08     1820.4     1.0    8.4
8192   71393us     1765us      6.74      0.03     6371.6     1.0   33.6
```

### 读法

1. **Prefill 算术强度随 seq 增长（126 → 6371），compute-bound**：seq 越长，attention 的 O(S²) 计算越多，而权重读取代价被摊薄，FLOPs/byte 越来越高。prefill tflops 稳定在 8-11（接近峰值）。

2. **Decode 算术强度恒定 ~1.0，memory-bound**：这是 LLM 推理最核心的数字。decode 每次的 attention FLOPs ∝ S（读 S 个 KV），而 KV cache bytes 也 ∝ S，两者同阶，所以 AI 恒定在 ~1 FLOP/byte——远低于本机 ridge（~几十 FLOP/byte）。**decode 的 GPU 几乎都在等内存，Tensor Core 利用率极低（d_tflops 只有 0.03-0.16）**。

### 这就是为什么

```text
"GPU utilization 100%，但 throughput 低"
"Decode 阶段 GPU 利用率很低，却是延迟主因"
```

decode 阶段的 GPU "忙"在等 KV cache 的访存，而不是在算。理解了这一点，就理解了 vLLM/SGLang 几乎所有优化的目标：**提高 decode 的 arithmetic intensity**（更大 batch、更省 KV 的 cache 布局、更激进的内存调度）。

---

## 4. 性能指标：TTFT / TPOT / ITL

```text
TTFT（Time To First Token）= 请求到达 → 第一个 token 的时间
                            ≈ 排队 + prefill 时间
                            （compute-bound，输入越长越慢）

TPOT（Time Per Output Token）= 每个输出 token 的平均时间
                              = decode 总时间 / 输出 token 数
                              （memory-bound，KV cache 越大越慢）

ITL（Inter-Token Latency）= 相邻 token 的间隔
                           ≈ TPOT，但含调度抖动（排队、抢占）

Throughput = 单位时间生成的 token 数（tokens/s）
```

三者的工程含义：

- **TTFT 是用户体验的第一道坎**（首 token 快不快）。优化 = 缩短 prefill（长 prompt 分块、prefix cache）。
- **TPOT 是生成体验**（token 吐得稳不稳、快不快）。优化 = 提高 decode 吞吐（batch、KV cache 优化）。
- **ITL 的抖动是实时/流式场景的关键**（机器人、语音助手要稳定的 token 流）。优化 = 调度（continuous batching）+ 显存管理。

---

## 5. 现代 LLM 推理系统的核心机制（Stage 12 铺垫）

理解了两阶段差异，就能理解这些机制在解决什么：

| 机制 | 解决的问题 |
|---|---|
| **Continuous Batching** | 静态 batch 里长/短请求互相等；动态增删请求，提高吞吐 |
| **PagedAttention** | KV cache 按固定块分配，消除碎片，提高显存利用率 |
| **Prefix Cache** | 共享 prompt 前缀只算一次 prefill（system prompt、多轮对话） |
| **Speculative Decoding** | 小模型草稿 + 大模型验证，decode 从串行变并行，降 TPOT |

这些是 Stage 12（vLLM/SGLang）的核心内容。

---

## 6. 本模块闭环小结

```text
问题：LLM 推理为什么难优化
      ↓
原理：Prefill 并行 O(S²) compute-bound，Decode 串行 O(S) memory-bound
      ↓
实测：prefill AI 126→6371，decode AI 恒定 1.0
      ↓
指标：TTFT（prefill）、TPOT/ITL（decode）、tokens/s（吞吐）
      ↓
下一步：Stage 12 vLLM/SGLang（continuous batching、paged attention、KV cache 管理）
```

要继续就说「继续」。
