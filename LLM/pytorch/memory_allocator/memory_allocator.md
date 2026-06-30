# CUDACachingAllocator 深度源码分析 + 用户最佳实践

> 源码: `c10/cuda/CUDACachingAllocator.cpp` (5390 行) — 核心分配器
> 常量: `c10/core/AllocatorConfig.h:15-24` — block size 阈值
> Python: `torch/cuda/memory.py` — 用户可见 API

## 0. 核心设计理念

PyTorch 的 CUDA 内存分配器不是对 `cudaMalloc/cudaFree` 的简单封装，而是一个**带缓存的分级分配系统**。设计目标：避免频繁的 CUDA driver 调用（每次 ~10-100μs），同时减少碎片化。

---

## 一、分配器的三级结构

### 1.1 Block 和 Pool 的层级

```
cudaMalloc (driver) — 昂贵, 只在必要时调用
  │
  ▼
DeviceCachingAllocator::alloc_block() — 按需申请新 Block
  │
  ▼
BlockPool (缓存池)
  ├─ large_blocks (size > 1 MB) — 独立的大块
  └─ small_blocks (size ≤ 1 MB) — 打包在 2 MB segment 中
```

### 1.2 关键常量 (`c10/core/AllocatorConfig.h:15-24`)

```cpp
constexpr size_t kMinBlockSize = 512;      // 最小分配粒度
constexpr size_t kSmallBuffer = 2097152;   // small block 的 segment 大小 = 2 MB
constexpr size_t kSmallSize = 1048576;     // 最大 "small" 分配 = 1 MB
constexpr size_t kMinLargeAlloc = 10485760;// 10 MB+ 用专用策略
constexpr size_t kRoundLarge = 2097152;    // large allocation 对齐 = 2 MB
```

这三层设计的作用:
- **< 512 bytes**: 取整到 512 (避免过多微小块)
- **512B ~ 1MB**: 从 2MB segment 中切分 (small pool)
- **1MB ~ 10MB**: 独立 large block
- **> 10MB**: 独立 large block + 按需从 driver 申请

---

## 二、`malloc()` 的 5 步查找链 (`:1722-1801`)

```cpp
// CUDACachingAllocator.cpp:1722
Block* malloc(size_t orig_size, cudaStream_t stream) {
    size_t size = round_size(orig_size);           // Step 0: 对齐
    auto& pool = get_pool(size, stream);           // Step 0: 选择池

    // Step 1: 从现有缓存池找可用的 block
    bool found = get_free_block(params);

    // Step 2: 触发 free callback, 再查一次
    if (!found) found = trigger_free_memory_callbacks(params)
                        && get_free_block(params);

    // Step 3: 没有 → cudaMalloc 申请新 block
    if (!found) found = alloc_block(params, ...);

    // Step 4: OOM → 尝试回收缓存
    if (!found) found = release_available_cached_blocks(params)
                        && alloc_block(params, ...);

    // Step 5: 还不行 → 清空所有非 split 缓存
    if (!found) found = release_cached_blocks(...)
                        && alloc_block(params, true, ...);

    if (!found) throw OOM;  // 真 OOM
}
```

### 2.1 `round_size()` — 对齐策略 (:3062)

```cpp
size_t round_size(size_t size) {
    if (size < kMinBlockSize) return kMinBlockSize;    // 最小 512
    if (size < kSmallSize) {
        // power_of_2 对齐 (减少碎片)
        return round_up_power2(size, divisions);
    }
    // > 1 MB: 对齐到 2 MB 倍数
    return kRoundLarge * ceil_div(size, kRoundLarge);
}
```

**为什么用 power-of-2**: 减少块种类 → 复用率更高 → 碎片化降低。

---

## 三、显存指标的物理含义

| 指标 | API | 含义 |
|------|-----|------|
| `allocated_bytes` | `torch.cuda.memory_allocated()` | 所有 tensor storage 的 sum |
| `reserved_bytes` | `torch.cuda.memory_reserved()` | allocator 向 CUDA 申请的 block 总大小 |
| `active_bytes` | 内部统计 | allocated + 还未 free 但 tensor 已废弃 |
| `inactive_split_bytes` | snapshot | split 产生的碎片 |

```
total GPU memory
 ├─ PyTorch reserved (缓存池)
 │   ├─ allocated (tensor 在用)
 │   │   ├─ active (tensor 仍在作用域)
 │   │   └─ inactive (tensor del 但 block 未 free)
 │   └─ cached (未被 tensor 使用, 缓存池中)
 │       ├─ 可复用的完整 block
 │       └─ split 碎片 (单个 block 太小)
 └─ 其他 (driver, other processes)
```

---

## 四、GPU 内存层级与访存优化

### 4.1 GPU 内存层级

```
延迟    带宽       容量
~1 cycle  最快      ~256 KB/MP    寄存器 (per-thread, 最快)
~20 cyc    ~10 TB/s  ~164 KB/SM    L1/Shared Memory
~200 cyc   ~4 TB/s   ~40 MB        L2 Cache
~400 cyc   ~2 TB/s   up to 80 GB   HBM (Global Memory)

关键: HBM → 寄存器 延迟差 > 400x!
     → 减少 global memory 读写 = 核心优化目标
```

### 4.2 Memory Coalescing (合并访存)

```
线程:     T0 T1 T2 T3 ... T31
访问地址: A0 A1 A2 A3 ... A31  (连续 → 1 次 128B transaction)
访问地址: A0 A8 A16 ...         (不连续 → 多次 transaction)

规则: warp (32 threads) 访问连续地址 → 最少 transaction
      torch 保证: contiguous tensor → 自然 coalescing
```

### 4.3 Channels Last (NHWC) 优化

```
NCHW (默认): [N, C, H, W] strides = [C*H*W, H*W, W, 1]
  访存模式: 沿着 W 维度连续 → 同一个 channel 内连续
  对 Conv 不利: 读取 H×W patch 时跨越多个 cache line

NHWC (channels_last): [N, H, W, C] strides = [H*W*C, W*C, C, 1]
  访存模式: 沿着 C 维度连续 → 一个像素的所有 channel 在一起
  对 Conv 有利: 读取一个 patch 时 channel 在连续内存中
```

**何时用 NHWC**: Conv 占主导的模型 (ResNet, EfficientNet)。
**何时不用**: 全连接层为主、或 1×1 Conv（NCHW 和 NHWC 差异不大）。

---

## 五、用户内存最佳实践

### 5.1 Pin Memory 的正确使用

```python
# ✅ 正确: 数据在 CPU, 目标设备是 GPU
data = torch.randn(N, C, H, W)  # CPU tensor
dl = DataLoader(ds, pin_memory=True)
for x, y in dl:
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    # non_blocking=True: DMA 异步传输, CPU 同时做其他事

# ❌ 错误: 数据已经在 GPU 上
x = torch.randn(N, device="cuda")
x.pin_memory()  # 无效果! pin 只对 CPU tensor 有意义
```

### 5.2 减少 alloc/free 的抖动

```python
# ❌ 常见但低效
for step in range(N):
    x = torch.randn(B, D, device="cuda")  # alloc
    y = model(x)
    # x 离开作用域 → free → small block 缓存
    # 下次循环 → alloc → 从缓存取回来

# ✅ 复用 buffer
x = torch.empty(B, D, device="cuda")  # 一次 alloc
for step in range(N):
    x.normal_()  # 覆写内容
    y = model(x)
```

### 5.3 控制碎片化

```python
# ❌ 大 tensor 和小 tensor 交替分配 → 碎片化
for _ in range(100):
    big = torch.randn(1024, 1024, device="cuda")   # 4 MB
    small = torch.randn(1, 16, device="cuda")       # 64 B
    del big  # 留下 4 MB 空洞
    # small 分配在 4 MB 空洞中 → 未来 4 MB 分配可能 OOM

# ✅ 先分配大块, 再小块; 或者定期 empty_cache()
torch.cuda.empty_cache()  # 每 N 步清理一次碎片
```

### 5.4 正确理解 memory_allocated vs reserved

```python
# 训练循环常见的误解:
# "我 del model 后 allocated 归零, 为什么 reserved 还是 20GB?"

del model, optimizer
torch.cuda.empty_cache()
# 现在 reserved 才会下降
# 因为 del 只释放 tensor → block 回池; cudaFree 要显式调用
```

---

## 六、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| 常量定义 | `c10/core/AllocatorConfig.h` | 15-24 |
| `DeviceCachingAllocator` 类 | `CUDACachingAllocator.cpp` | 1426 |
| `BlockPool` (large/small) | `CUDACachingAllocator.cpp` | 1437-1440 |
| `malloc()` 入口 | `CUDACachingAllocator.cpp` | 1722 |
| `round_size()` 对齐 | `CUDACachingAllocator.cpp` | 3062 |
| `get_free_block()` 缓存查找 | `CUDACachingAllocator.cpp` | 1751 |
| `alloc_block()` 新分配 | `CUDACachingAllocator.cpp` | 1770 |
| `release_cached_blocks()` | `CUDACachingAllocator.cpp` | — |
| `memory_allocated()` Python | `torch/cuda/memory.py` | — |
| `memory_summary()` | `torch/cuda/memory.py` | — |

---

## 七、实战常见坑点

### 1. memory_reserved 只增不减
**原因**: allocator 缓存策略 — 会保留已申请的 block 不复用。cached block 被视为「免费」。
**解决**: 如果 reserved - allocated > 50% GPU memory → 考虑 `empty_cache()` 或减少 alloc/free 频率。

### 2. 相同模型, fp16 省显存但 OOM?
**原因**: fp16 矩阵乘法内部使用 fp32 累加 → 临时显存比单精度更大。
**排查**: 用 `torch.cuda.memory_summary()` 查看 `inactive_split_bytes`。

### 3. 多进程共享 GPU 时的一个进程 OOM
**原因**: PyTorch allocator 不知道其他进程的 usage — 每个进程独立 `cudaMalloc`。
**解决**: 限制每个进程的显存上限:
```python
torch.cuda.set_per_process_memory_fraction(0.4)  # 只用 40%
```

### 4. num_workers > 0 时每个 worker 也分配 CUDA 显存
**原因**: worker 初始化时 PyTorch 的 CUDA context 被 fork → 每个 worker 有自己的 allocator。
**解决**: 用 `spawn` 方式 + `worker_init_fn` 限制显存:
```python
def worker_init_fn(worker_id):
    torch.cuda.set_per_process_memory_fraction(0.1)
```
