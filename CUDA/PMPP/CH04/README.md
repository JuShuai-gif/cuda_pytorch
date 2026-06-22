好的 👍 我来把你上面那整段 **Chapter 4 的讲解与习题答案** 翻译成中文，保持结构、格式和技术细节不变：

---

# 第 4 章

## 代码

在本章中，我们几乎没有实际代码，只是查询 CUDA 设备的属性。

直接运行 [Makefile](code/Makefile)：

```bash
cd code
```

```bash
make
```

你会看到类似这样的输出：

```
Detected 2 CUDA capable device(s)

Device 0: "NVIDIA GeForce RTX 4090"
  Major revision number:         8
  Minor revision number:         9
  Total amount of global memory: 23.65 GB
  Number of multiprocessors:     128
  Total amount of constant memory: 65536 bytes
  Total amount of shared memory per block: 49152 bytes
  Total number of registers available per block: 65536
  Warp size:                     32
  Maximum number of threads per block: 1024
  Maximum sizes of each dimension of a block: 1024 x 1024 x 64
  Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
  Clock rate:                    2.57 GHz
  Memory clock rate:             10501 MHz
  Memory bus width:              384-bit
  L2 cache size:                 75497472 bytes

Device 1: "NVIDIA GeForce RTX 4090"
  Major revision number:         8
  Minor revision number:         9
  Total amount of global memory: 23.65 GB
  Number of multiprocessors:     128
  Total amount of constant memory: 65536 bytes
  Total amount of shared memory per block: 49152 bytes
  Total number of registers available per block: 65536
  Warp size:                     32
  Maximum number of threads per block: 1024
  Maximum sizes of each dimension of a block: 1024 x 1024 x 64
  Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
  Clock rate:                    2.57 GHz
  Memory clock rate:             10501 MHz
  Memory bus width:              384-bit
  L2 cache size:                 75497472 bytes
```

---

## 练习题

### 练习 1

**考虑以下 CUDA kernel 以及调用它的主机函数：**

```cpp
01 __global__ void foo_kernel(int* a, int* b) {
02     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03     if(threadIdx.x < 40 || threadIdx.x >= 104) {
04         b[i] = a[i] + 1;
05     }
06     if(i%2 == 0) {
07         a[i] = b[i]*2;
08     }
09     for(unsigned int j = 0; j < 5 - (i%3); ++j) {
10         b[i] += j;
11     }
12 }
13 void foo(int* a_d, int* b_d) {
14     unsigned int N = 1024;
15     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
16 }
```

**a. 每个 block 中有多少个 warp？**

一个 warp 有 32 个线程；每个 block 有 128 个线程（第二个参数 `128`），因此每个 block 有 `128/32=4` 个 warp。

**b. 整个 grid 中有多少个 warp？**

一共有 `(N + 128 - 1)/128 = 8` 个 block，每个 block 4 个 warp，因此总共有 `32` 个 warp。

**c. 关于第 04 行的语句：**

**i. grid 中有多少个 warp 是活跃的？**

* warp 0 (线程 0–31)：全部满足 `threadIdx.x < 40` → 全部执行 → warp 活跃。
* warp 1 (线程 32–63)：部分满足 `threadIdx.x < 40` → 整个 warp 需要执行（部分线程空转）→ warp 活跃。
* warp 2 (线程 64–95)：不满足条件 → warp 不执行 → warp 不活跃。
* warp 3 (线程 96–127)：部分满足 `threadIdx.x >= 104` → warp 活跃。

因此每个 block 有 3 个 warp 活跃，总共 `3 × 8 = 24` 个 warp 活跃。

**ii. grid 中有多少 warp 出现了分歧 (divergence)？**

warp 1 和 warp 3 出现分歧（部分线程执行，部分不执行）。每个 block 有 2 个分歧 warp，总共有 `8 × 2 = 16` 个分歧 warp。

**iii. block 0 的 warp 0 的 SIMD 效率 (%)？**

warp 0 的线程 0–31 全部执行 → `32/32 = 100%`。

**iv. block 0 的 warp 1 的 SIMD 效率 (%)？**

warp 1 的线程 32–63 中，只有 `32–39` 执行，总共 8 个线程。效率为 `8/32 = 25%`。

**v. block 0 的 warp 3 的 SIMD 效率 (%)？**

warp 3 的线程 96–127 中，`96–103` 不执行，`104–127` 执行，总共 24 个线程。效率为 `24/32 = 75%`。

**d. 关于第 07 行的语句：**

**i. grid 中有多少 warp 活跃？**

每两个线程中就有一个线程执行 → 所有 warp 都有部分线程活跃 → 32 个 warp 全部活跃。

**ii. grid 中有多少 warp 出现分歧？**

所有 warp 都分歧 → 一共有 32 个分歧 warp。

**iii. block 0 的 warp 0 的 SIMD 效率 (%)？**

warp 中一半线程执行 → `16/32 = 50%`。

**e. 关于第 09 行的循环：**

`i` 范围 `0–1023`，所以 `i%3` 的可能值是 `{0,1,2}`。

* 342 个 `i%3=0` → 执行 5 次循环
* 341 个 `i%3=1` → 执行 4 次循环
* 341 个 `i%3=2` → 执行 3 次循环

**i. 有多少次迭代无分歧？**

前三次迭代所有线程都会执行 → 3 次迭代无分歧。

**ii. 有多少次迭代有分歧？**

第 4 次和第 5 次迭代时部分线程退出 → 有分歧。

---

### 练习 2

向量加法，向量长度为 2000，每个线程计算一个输出元素，block 大小为 512。
需要多少线程？

最少需要 4 个 block → `4 × 512 = 2048` 个线程。

---

### 练习 3

上一题中，有多少个 warp 会因边界检查产生分歧？

一共有 `2048 / 32 = 64` 个 warp。

* warp `[2016–2047]` 全部超界 → 不活跃。
* warp `[1984–2015]` 部分有效 (1984–1999)，部分无效 (2000–2015) → 分歧。

因此只有 1 个 warp 分歧。

---

### 练习 4

一个 block 有 8 个线程，执行时间分别是：
`2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9` 微秒。
剩余时间都在等待 barrier。

最长执行时间为 `3.0` 微秒。等待时间和：

```
(3.0-2.0) + (3.0-2.3) + (3.0-3.0) + (3.0-2.8) 
+ (3.0-2.4) + (3.0-1.9) + (3.0-2.6) + (3.0-2.9) 
= 4.0
```

总执行时间为 `8 × 3.0 = 24.0` 微秒。

等待比例：`4.0 / 24.0 = 16%`。

---

### 练习 5

一位 CUDA 程序员说，如果每个 block 只启动 32 个线程，就可以省略 `__syncthreads()`。

分析：

* 因为一个 warp 内的线程同步执行，所以理论上确实不需要 barrier。
* 但涉及内存读写时，实际硬件可能有延迟，仍可能需要同步。
* 此外，Nvidia 并不保证 warp 大小永远是 32，将来架构可能变化。

结论：不推荐省略 `__syncthreads()`。

---

### 练习 6

某 CUDA 设备：每个 SM 最多 1536 个线程，最多 4 个 block。
以下 block 配置哪个能得到最多线程？

* a. 128 线程/block → `min(4,12) × 128 = 512`
* b. 256 线程/block → `min(4,6) × 256 = 1024`
* c. 512 线程/block → `min(4,3) × 512 = 1536`
* d. 1024 线程/block → `min(4,1) × 1024 = 1024`

答案：**c. 512 线程/block → 1536 线程**。

---

### 练习 7

设备：每个 SM 最多 64 个 block，最多 2048 个线程。
判断以下配置是否可能，并给出占用率：

* a. 8 × 128 = 1024 → 可行，占用率 50%
* b. 16 × 64 = 1024 → 可行，占用率 50%
* c. 32 × 32 = 1024 → 可行，占用率 50%
* d. 64 × 32 = 2048 → 可行，占用率 100%
* e. 32 × 64 = 2048 → 可行，占用率 100%

---

### 练习 8

设备：每个 SM 最多 2048 线程，32 个 block，65536 寄存器。

**a. 128 线程/block，30 寄存器/线程**

* 基于线程：`2048/128=16 block` → OK
* 基于寄存器：`128×30=3840` → `65536/3840≈17 block` → OK
* 实际：`16×128=2048` → 满占用 → **100%**

**b. 32 线程/block，29 寄存器/线程**

* 基于线程：`2048/32=64 block` → 超过 32 block 限制 → 限制
* 基于寄存器：`32×29=928` → `65536/928≈70` → 超过 32 block 限制
* 实际：`32×32=1024` → 占用 **50%**
* 限制因素：block 数量

**c. 256 线程/block，34 寄存器/线程**

* 基于线程：`2048/256=8 block` → OK
* 基于寄存器：`256×34=8704` → `65536/8704≈7 block`
* 实际：`7×256=1792` → 占用 `1792/2048≈87%`
* 限制因素：寄存器

---

### 练习 9

学生声称：

* 使用 32×32 的 thread block（=1024 线程）
* CUDA 限制：512 线程/block，8 block/SM
* 每个线程计算一个结果元素
* 乘两个 1024×1024 矩阵

分析：

* 矩阵有 `1024×1024=1,048,576` 个元素
* 学生 grid 有 `32×32=1024 block`，每个 block 512 线程
* 总线程数 = `1024×512=524,288`
* 但结果矩阵需要 `1,048,576` 个线程 → 线程数不够

结论：学生的说法错误。

