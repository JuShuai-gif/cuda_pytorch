# 01｜Edge AI：Jetson 平台与 Thermal/Power 约束

## 本模块解决的问题

边缘推理和云端推理最大的区别不是"算力小"，而是**物理约束**：功耗、散热、电池。本章用本机（Jetson Thor）回答：

```text
Edge AI 平台有哪些，各自什么架构？
统一内存对推理意味着什么？
为什么 benchmark 只测 latency 是不够的？
"跑 10 秒很快"和"连续 24 小时稳定"为什么是两回事？
```

配套代码：`src/edge/jetson/`（`platform.py` + `monitor.py` + `workload.py` + `benchmark.py`）。

---

## 1. Edge AI 平台分类

| 平台 | CPU | 加速器 | 特点 |
|---|---|---|---|
| ARM（Raspberry Pi 等） | ARM Cortex | 无/弱 GPU | 极低功耗，只跑轻模型 |
| x86（Intel/AMD） | x86 | iGPU | 通用，功耗高 |
| **NVIDIA Jetson** | ARM | **CUDA GPU** | 本机平台，CUDA 生态完整 |
| RK（瑞芯微） | ARM | NPU | 国产，NPU 专用 |
| Qualcomm | ARM | Hexagon NPU | 手机/无人机，NPU 强 |
| Apple Silicon | ARM | ANE | 统一内存，ANE 专用 |

**优先深度掌握一种平台，再理解差异**。本工程选 Jetson（有完整 CUDA + TensorRT 生态，和前面的 Stage 无缝衔接）。

---

## 2. Jetson Thor 的架构特征

本机实测（`platform.py`）：

```text
arch      = aarch64（ARM CPU）
GPU       = NVIDIA Thor（Blackwell 架构，sm_110，20 SM）
内存      = 统一内存 ~128GB（CPU/GPU 共享物理 DRAM）
power mode = MAXN（最高性能模式）
CPU max   = 2601 MHz
```

### 统一内存的含义

Jetson 的 CPU 和 GPU **共享同一块物理 DRAM**。所以：

1. **H2D/D2H 不是跨 PCIe 的搬运**，而是页迁移/一致性处理（Stage 2 已实测 pinned vs pageable 差距小）。
2. **Zero Copy 可行**：CPU 和 GPU 可以操作同一块内存，省掉拷贝。
3. **显存 = 系统内存**：`total_memory` 报告 ~128GB，不是离散 GPU 的独立显存。

这改变了"优化 H2D"的优先级——在 Jetson 上，**Zero Copy 比优化 memcpy 更有意义**。

---

## 3. 实测：idle vs 持续负载的 thermal/power

跑 30 秒持续 fp16 GEMM 负载，同时用 `tegrastats` 采样（`monitor.py`）：

```text
              gpu_temp       total_power    cpu_freq
idle          44.6°C         24.8W          2601 MHz
sustained     58.7→74.6°C    127.6W mean    2601 MHz（稳定）
load peak                    134.7W
```

### 读法

1. **功耗 5 倍增长**：24.8W → 127.6W。edge 设备的功耗预算很紧（电池/散热），持续高功耗意味着发热和续航问题。

2. **温度爬升 16°C**：负载期间 GPU 从 58.7°C 升到 74.6°C。温度上升不是瞬时的，是**热积累**——这正是"短跑 vs 长跑"的区别。

3. **本机未触发 throttling**：CPU 频率稳定 2601MHz，因为峰值 74.6°C 还没到 thermal 阈值（通常 90°C+）。这说明 Thor 的散热能承受 ~127W 持续负载到 ~75°C。**但这不意味着不会 throttling**——更长负载、更高环境温度、散热积灰都会触发。

---

## 4. Thermal Throttling 的机制

所有 edge SoC 都有 thermal envelope：

```text
温度上升 → 超过阈值（如 90°C）
        → 降频（CPU/GPU clock 下调）→ 温度回落 → 频率恢复 → 再升温 → 循环
```

**throttling 对推理的影响**：latency 突然变慢、且不稳定（频率来回跳）。这比"恒定慢"更危险，因为它制造 **jitter**（呼应 Stage 14 的实时性）。

```text
短跑（10 秒）：温度低，频率高，latency 低且稳
长跑（24 小时）：温度升高，可能进入 throttling，latency 变慢 + 抖动
```

### 监控什么（edge 比 server 多什么）

| 指标 | 为什么 |
|---|---|
| GPU/CPU 频率 | throttling 的直接证据 |
| 温度 | 逼近阈值的预警 |
| 功耗 | 续航、散热预算 |
| latency + jitter | 用户/机器人最终体验 |
| 内存 | 统一内存下显存=系统内存 |

**edge benchmark 必须同时采 latency + power + temp + freq**，单看 latency 会漏掉"温度在爬、频率要降"的隐患。

---

## 5. Power Mode 与 Clock（需要 root，本机记录命令）

```bash
# 查看当前 power mode
nvpmodel -q

# 切换 power mode（需要 root）
sudo nvpmodel -m 0    # MAXN（最高性能）
sudo nvpmodel -m 1    # 低功耗模式

# 锁定时钟（需要 root，本机需要 sudo 密码，Not Validated）
sudo jetson_clocks
```

本机 power mode 是 MAXN，`jetson_clocks`/`nvpmodel -m` 需要 sudo（本机无密码 sudo），所以**切换 power mode 和锁时钟的实测标记 Not Validated**，只记录命令。这是"正确提供方法 + 标记 Not Validated"原则。

---

## 6. 本模块闭环小结

```text
问题：edge 推理和 cloud 推理差在哪
      ↓
差异：统一内存 + power/thermal 物理约束
      ↓
实测：idle 24.8W/44.6°C → 持续负载 127.6W/74.6°C（峰值 134.7W）
      ↓
结论：edge 要采 power/temp/freq，throttling 制造 jitter 比恒定慢更危险
      ↓
下一步：Stage 16 Inference Server（HTTP/RPC → 模型 → GPU，batching/queue）
```

要继续就说「继续」。
