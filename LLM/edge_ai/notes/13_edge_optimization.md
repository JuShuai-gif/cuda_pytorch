# 边缘端性能优化实战：RK3588 机器人视觉管线调优

## 1. 边缘端内存访问的隐藏陷阱：uncached vs cached 内存

### 1.1 问题背景

在 RK3588 机器人视觉管线中，DMA buffer 用于 NPU/RGA/CPU 之间的数据传输。默认使用 `/dev/dma_heap/system-uncached` 分配的 DMA buffer，CPU 读取速度极慢：

| 内存类型 | CPU 单次读取延迟 | 6MB 帧读取时间 |
|----------|-----------------|---------------|
| cached   | 10-30ns/缓存行  | ~0.3ms |
| uncached | 100-300ns/缓存行 | ~15ms |

**15ms vs 0.3ms，差距高达 50 倍**。

### 1.2 根因分析

DMA 写入的数据直接进入 DRAM，完全绕过 CPU cache。CPU 读取 uncached 内存时，每次访问都要走完整的内存总线来回 DRAM，没有 cache 的缓冲加速。当 NPU、CPU、RGA、Display 多个模块同时争抢 DDR 带宽时，uncached 读的延迟抖动会急剧放大（从 ±5ms 变成 ±20ms）。

```
Cached 路径:   CPU → L1/L2 Cache（命中） → 10ns
Uncached 路径: CPU → 内存控制器 → DDR PHY → DRAM 颗粒 → 返回 → 100-300ns
```

### 1.3 带宽争抢放大效应

RK3588 的 DDR 带宽被多模块共享：
- NPU 推理：占用大量读带宽（权重 + 特征图）
- RGA 图像处理：读写各一份
- CPU 后处理：读 uncached buffer
- 显示控制器：持续读取 framebuffer

当 CPU 用 uncached 方式读取 6MB 帧时，不仅自身慢，还会挤占其他模块的 DDR 带宽，导致整体系统抖动加剧。

---

## 2. DMA_BUF_IOCTL_SYNC 的工作原理和正确使用

### 2.1 核心思路

使用 **cached dma-heap** 分配 buffer，在 DMA 写入完成后、CPU 读取之前，调用 `DMA_BUF_IOCTL_SYNC` 做 cache 失效（invalidate），让 CPU 看到最新的 DMA 数据。

```
时间线:
  [RGA DMA 写入] → [DMA_BUF_IOCTL_SYNC START_READ] → [CPU memcpy @ cached 速度] → [DMA_BUF_IOCTL_SYNC END_READ]
```

### 2.2 ioctl 参数

```c
struct dma_buf_sync {
    __u64 flags;
};

#define DMA_BUF_SYNC_READ      (1 << 0)  // CPU 读之前：cache invalidate
#define DMA_BUF_SYNC_WRITE     (1 << 1)  // CPU 写之后：cache writeback
#define DMA_BUF_SYNC_START     (0 << 2)  // 开始 CPU 访问
#define DMA_BUF_SYNC_END       (1 << 2)  // 结束 CPU 访问
```

常用组合：
- `DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ`：DMA→CPU 传输前，使 cache 失效
- `DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ`：CPU 读取完毕，释放控制
- `DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE`：CPU→DMA 传输前，回写 cache
- `DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE`：CPU 写入完毕，DMA 可以读

### 2.3 实际效果

优化前后 P50/P99 延迟对比（6MB 帧读取）：

| 指标 | uncached（优化前） | cached + SYNC（优化后） | 提升 |
|------|-------------------|------------------------|------|
| P50 延迟 | 15ms | 3.6ms | 4.2x |
| P99 延迟 | 45ms | 7ms | 6.4x |
| 抖动 | ±20ms | ±2ms | 10x |

---

## 3. Fail-closed 错误处理模式

### 3.1 问题

DMA_BUF_IOCTL_SYNC 调用可能失败（kernel 返回错误）。如果 sync 失败后仍然读取 buffer，CPU 可能读到的是 cache 中过期的脏数据而非 DMA 刚写入的新数据，导致后续推理结果完全错误。

### 3.2 Fail-closed 策略

```
sync_result = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync_args);
if (sync_result < 0) {
    origReady = false;  // 标记帧不可用，拒绝处理
    return;             // 不使用可能过期的数据
}
// sync 成功后，安全读取
memcpy(dst, src, size);
```

**关键原则**：
- 宁可丢帧（不处理），也绝不使用不确定状态的数据
- 绝不 fallback 到 virtual-address RGA 模式（在 RK3588 上会导致系统 freeze）
- 优雅降级：丢一帧不影响整体系统，使用过期数据会导致级联错误

### 3.3 Fail-open 的危害

Fail-open（忽略错误继续处理）的风险：
- 读到 cache 中的过期数据 → NPU 推理出错误结果
- 下游模块基于错误推理结果做决策 → 机器人误动作
- 偶发性 bug 极难复现和定位

---

## 4. NEON SIMD 在边缘端的实际效果和局限性

### 4.1 FP16→FP32 转换

RKNN 模型的输出默认是 FP16 格式。如果用 `rknn_outputs_get(want_float=1)` 让 NPU 驱动做转换，耗时 9.68ms（145 万 float）。切换为 `want_float=0`（直接获取 FP16）+ CPU NEON 转换：

```c
// NEON fcvtl 指令：一条指令转换 4 个 FP16→FP32（下半部分）
// NEON fcvtl2 指令：一条指令转换 4 个 FP16→FP32（上半部分）
// 145 万次转换 / 每指令 4 个 = 36 万条指令
float32x4_t lo = vcvt_f32_f16(vld1_f16(src));       // 低 4 个
float32x4_t hi = vcvt_f32_f16(vld1_f16(src + 4));    // 高 4 个（或用 fcvtl2）
```

### 4.2 实际收益的局限性

虽然 NEON 转换本身很快（~0.3ms），但整体延迟只能降到 ~9.3ms。瓶颈在 **NPU→CPU DMA 传输**（~9ms），而非转换计算。这揭示了一个重要洞察：

**带宽瓶颈 > 计算瓶颈**。在边缘端，DDR 带宽是稀缺资源，优化计算指令对端到端延迟的提升有限。

### 4.3 BGR→FP16 RGB 转换

`bgr_to_fp16_rgb_neon` 使用：
- `LD3` 指令：一次加载 3 个通道各自 8 个 uint8，完成 BGR→RGB 通道重排
- `fcvt` 指令：uint8→fp16
- `ST3` 指令：交错存储

40.96 万像素 × 3 通道的处理本身很快，但受限于 io_mem 写入带宽（NPU DDR 争抢）。

### 4.4 何时 NEON 有效

NEON 优化有效的条件：
- 数据已经在 cache 中（热数据）
- 计算密集（如矩阵乘法、卷积）
- 不受外部带宽限制

NEON 优化无效的场景：
- 数据在 uncached 内存中
- DDR 带宽已被其他模块占满
- 计算复杂度低，受内存墙限制

---

## 5. 带宽瓶颈 vs 计算瓶颈的区分方法

### 5.1 判断标准

| 特征 | 带宽瓶颈 | 计算瓶颈 |
|------|---------|---------|
| 增大数据量 | 延迟线性增长 | 延迟不变或亚线性 |
| 降低精度(FP32→FP16) | 延迟几乎不变 | 延迟明显下降 |
| 加 NEON/SIMD | 提升有限（<10%） | 提升显著（2-4x） |
| perf stat cache-miss | 高 | 低 |
| roofline 分析 | 低于带宽上限 | 低于计算峰值 |

### 5.2 实际管线瓶颈分布

RK3588 机器人视觉管线的 P0 阶段耗时：

```
P0: rknn_run(NPU推理)    → 39ms   ← 计算瓶颈（NPU 算力）
P1: output DMA(NPU→CPU)  → 9.4ms  ← 带宽瓶颈（DDR）
P2: letterbox io_mem     → 3ms    ← 带宽瓶颈（DDR + RGA）
```

总计 ~51ms = 约 19.6 FPS。要突破 25+ FPS：
1. INT8 量化模型减少 NPU 推理时间（39ms→15ms）
2. 异步流水线：double buffer 实现 run(N) 和 get(N-1) 并行

### 5.3 buffer 预分配

替换每帧 `std::vector` 动态分配为持久 `scaledBuf_` 成员变量。当输入尺寸不变时 `resize()` 是零开销（no-op），避免了 malloc/free 的抖动。

---

## 6. 从 20ms 到 7ms 的优化路径复盘

### 6.1 优化前状态

```
帧后处理延迟 P50: 20ms, P99: 50ms, 抖动 ±20ms
核心问题: uncached DMA buffer 导致 CPU 读取成为瓶颈
```

### 6.2 优化步骤

| 步骤 | 操作 | P50 | P99 | 抖动 |
|------|------|-----|-----|------|
| 0 | 基线（uncached DMA buffer） | 15ms | 45ms | ±20ms |
| 1 | 切换到 cached dma-heap + DMA_BUF_IOCTL_SYNC | 3.6ms | 7ms | ±2ms |
| 2 | 加入 fail-closed 错误处理 | 3.6ms | 7ms | ±2ms |
| 3 | NEON FP16→FP32 转换 | 3.3ms | 6.5ms | ±2ms |
| 4 | buffer 预分配消除 malloc | 3.1ms | 6.2ms | ±1.8ms |

**总提升：P50 15ms→3.1ms（4.8x），P99 45ms→6.2ms（7.3x）**。

### 6.3 优化收益分布

```
uncached→cached+SYNC: 贡献 80% 收益（最大头）
NEON 转换:            贡献 8% 收益
buffer 预分配:         贡献 5% 收益
其他:                 贡献 7% 收益
```

---

## 7. 关键教训

### 7.1 不要信默认配置

- `/dev/dma_heap/system-uncached` 是某些 BSP 的默认选项，但性能极差
- `rknn_outputs_get(want_float=1)` 的默认行为消耗 9.68ms，而自行 NEON 转换仅 0.3ms
- 默认配置往往是"能跑"而非"跑得快"

### 7.2 perf 数据说话

- 每个优化决策都要有测量数据支撑
- 实际瓶颈可能与直觉完全不同（以为 NEON 能大幅提速，实际受限于 DDR 带宽）
- P50 不够，要看 P99 和抖动，实时系统最怕长尾延迟

### 7.3 嵌入式特有的考虑

- DDR 带宽是共享资源，单个模块的优化可能被其他模块的争抢抵消
- RGA 的 virtual-address 模式在 RK3588 上有已知 bug（系统 freeze），必须用物理地址模式
- 驱动版本非常重要，不同 BSP 版本行为可能完全不同

### 7.4 优化是系统工程

- 局部优化不能解决系统级瓶颈（如 DDR 带宽）
- 需要从系统架构层面考虑：异步流水线、模型量化、数据布局
- 单一线程的串行处理架构最终会触碰 Amdahl 定律的墙
