# 14 性能测试与Benchmark

> 本笔记对应 PDF 第 16 章 Testing speed（第 167～171 页），覆盖 16.1～16.3，并衔接第 3.2 节（profiler，第 16～18 页）。

## 1. 本章解决什么问题

优化的每一步都要回答："我改得真的更快了吗？" 本章教你**正确地测速**：

1. 用什么测量？（时间戳计数器 / 性能计数器）
2. 首次调用与后续调用差在哪？（冷/热缓存）
3. 单测有什么陷阱？（缓存被低估、代码体积被忽视）
4. 什么场景要测最坏情况？（实时音频视频、用户响应）

核心结论：**测量要在最终程序的上下文里做，用真实数据、多轮、考虑缓存与代码体积，不能只看单测。**

## 2. 核心概念

| 术语 | 含义 | 出处 |
|------|------|------|
| 时间戳计数器（TSC） | CPU 自带的按主频计数的计数器（`rdtsc`） | PDF 第 168 页 |
| 性能监视计数器（PMC） | CPU 内可配置计数的事件计数器（cache miss、分支预测失败等） | PDF 第 169 页 |
| 预热（warm-up） | 让 CPU 升频、代码/数据进缓存后再测 | PDF 第 168 页 |
| 冷启动（first call） | 首次执行，代码/数据未缓存 | PDF 第 168 页 |
| 单元测试陷阱 | 单函数测试忽略缓存与代码体积 | PDF 第 170 页 |
| 最坏情况测试 | 在资源紧张/干扰下测试 | PDF 第 171 页 |

## 3. 工作原理

### 3.1 用时钟周期测量（PDF 第 168 页）

读 `rdtsc`（时间戳计数器）前后差值，得到精确的时钟周期数。注意：跨核心会让 TSC 失效（每个核心有独立计数器），需固定线程亲和性；现代 CPU 频率动态变化，周期数不等于秒。

> 补充说明：本机 i9-14900HX 的 TSC 是 `invariant_tsc`（不随主频变化），适合测周期；但要转成秒需乘以真实频率。现代 Intel 用 `unhalted core cycles`（core clock cycles）计数器更稳定（PDF 第 169 页）。

### 3.2 测量结果怎么解读（PDF 第 168 页）

第一次执行通常比后续慢（代码/数据未缓存）。"最坏情况"（第一次）和"最好情况"（后续）分别对应程序中被调用一次 vs 频繁调用的场景。优化 CPU 效率看"最好情况"，优化缓存看"最坏情况"（PDF 第 168 页）。

### 3.3 频率不稳定怎么办（PDF 第 168 页）

- 测试前给 CPU 重活热身；
- BIOS 关节能选项；
- Intel 用 core clock cycles 计数器（不受频率影响）。

## 4. PDF 核心观点

### 第 16 章引言（第 167～168 页）

- 测试速度是优化的一部分：必须确认改动真的提速了（PDF 第 167 页）。
- profiler 不准（见 03 笔记）；热点定位后，隔离热点单独测量（PDF 第 167 页）。
- 用 TSC（`ReadTSC`）在代码前后读时钟，重复多次存数组（PDF 第 168 页，Example 16.1/16.2）。
- 测量含函数调用开销，可减去空测（PDF 第 168 页）。
- 首次调用（最坏情况）vs 后续调用（最好情况）的解读（PDF 第 168 页）。
- 任务切换会造成异常高值：测试前提高线程优先级可减少（PDF 第 168 页）。
- 结果波动大：现代 CPU 动态变频；用预热、关节能、core clock cycles 计数器（PDF 第 168 页）。

### 16.1 用性能监视计数器（第 169～170 页）

- 性能计数器可数：执行的指令数、缓存未命中、分支误预测等（PDF 第 169 页）。
- 厂商工具：Intel VTune、AMD CodeAnalyst；作者自研 testp 工具（PDF 第 169 页）。
- **core clock cycles 计数器**（Intel）按实际频率计数，几乎不受变频影响，非常适合对比两个版本谁快（PDF 第 169 页）。
- 不需要测试时关闭计数器读取，否则会崩溃（PDF 第 170 页）。

### 16.2 单元测试的陷阱（第 170～171 页）

- 单函数测试必要，但**不能给出真实性能**：测试程序内存总占用小于缓存，未命中的惩罚看不到（PDF 第 170 页）。
- 最终程序中，代码缓存/微操作缓存/数据缓存往往是瓶颈；**单位测试最快的版本可能不是最终最优**（内存足迹大）（PDF 第 170 页）。
- 例：大循环展开是否值得，单测不反映缓存影响（PDF 第 170 页）。
- 看函数占用多少内存：链接 map 文件或汇编清单（PDF 第 170 页）。
- **真实性能测试应包含内层循环（含被调用函数与热点），用真实数据**（分支误预测才真实），排除用户输入等待，文件 I/O 单独测（PDF 第 170 页）。
- 单测陷阱非常常见，连一些优秀函数库也过度展开循环导致内存足迹过大（PDF 第 170 页）。

### 16.3 最坏情况测试（第 171 页）

- 最好情况测试可复现、适合对比实现；但有些场景必须测最坏情况（PDF 第 171 页）。
- 需要最坏情况测试的场景：用户响应时间上限、实时音频/视频流不能丢帧（PDF 第 171 页）。
- 最坏情况手段（PDF 第 171 页）：
  - 首次激活代码（懒加载/缓存未命中/分支误预测）；
  - 测整个软件包而非单函数，切换功能迫使代码换出缓存；
  - 网络/服务器满负荷时测；
  - 大数据文件与数据库；
  - 旧电脑 + 内存不足 + 后台进程多 + 慢/碎片化磁盘；
  - 不同品牌 CPU、不同显卡；
  - 开启全盘病毒扫描；
  - 多进程/多线程并发（超线程同核跑两线程）；
  - 分配超过内存总量的内存强制换页；
  - 让代码/数据超过缓存或 `_mm_clflush` 主动失效缓存；
  - 让数据更随机引发分支误预测。

## 5. 简单示例

用 `std::chrono` 做预热 + 多轮测量（PDF 第 168 页的"重复测试、存数组"思想，结合现代 C++）：

```cpp
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>

// Warm-up + multiple rounds, report min/median (PDF p168 spirit).
template <class Fn>
void benchmark(const char *name, Fn fn, int rounds = 7) {
    for (int i = 0; i < 3; ++i) fn();          // warm-up: cache + turbo

    std::vector<double> us;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        volatile auto sink = fn();             // prevent elimination
        auto t1 = std::chrono::steady_clock::now();
        (void)sink;
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    std::printf("%-12s min=%8.2f us  median=%8.2f us\n",
                name, us.front(), us[us.size() / 2]);
}
```

## 6. 未优化代码（错误的测法）

```cpp
// BAD benchmarking practices (PDF p168-171):
//   * no warm-up          -> cold cache inflates the first reading
//   * single shot         -> turbo/OS noise hides the true value
//   * -O0 build           -> measuring debug code, not the release
//   * tiny data           -> everything fits in cache, misses invisible
auto t0 = std::chrono::steady_clock::now();
result = heavy_function();
auto t1 = std::chrono::steady_clock::now();
// draw conclusions from ONE sample -> meaningless (PDF p168)
```

## 7. 优化后代码（正确的测法）

```cpp
// GOOD: warm-up + multiple rounds + median + release build + realistic data
// (see the benchmark helper in section 5 and src/common/benchmark).
//   - compile with -O3 (release), never -O0/-g debug for speed tests
//   - data size larger than the cache if cache behavior matters
//   - verify results of baseline vs optimized are identical (checksum)
```

## 8. 为什么会更快（测准了才算数）

测速本身不改快代码，但**决定优化决策是否正确**：

- **预热**：跳过冷缓存/变频噪声，得到稳定读数（PDF 第 168 页）。
- **多轮 + 中位数**：抗任务切换与频率毛刺（PDF 第 168 页）。
- **Release 构建**：`-O0` 下的结论对最终程序无意义（PDF 第 85 页）。
- **真实数据规模**：小数据全在缓存里，看不出内存优化效果（PDF 第 170 页）。
- **校验和**：防止编译器消除代码、防止优化版算错（`src/common/benchmark` 实现）。

## 9. 如何验证

```bash
# 编译（必须 Release；Debug 结果不算数）
g++ -O3 -std=c++17 benchmark_demo.cpp -o benchmark_demo
g++ -O0 -std=c++17 benchmark_demo.cpp -o benchmark_demo_debug

# 对比：Debug 与 Release 结论可能完全不同（PDF p85）
./benchmark_demo_debug
./benchmark_demo

# 用 perf 交叉验证
sudo perf stat -e cycles,instructions,cache-misses ./benchmark_demo

# 用性能计数器看 cache-misses / branch-misses（定位瓶颈类型）
sudo perf stat -e cache-misses,branch-misses ./benchmark_demo
```

- 编译命令：`g++ -O3 -std=c++17`（本机 g++ 13.3.0）
- 运行命令：`./benchmark_demo`
- Benchmark 方法：`src/common/benchmark`（预热、多轮、min/median、校验和）
- perf 命令：`sudo perf stat -e ...`（本机需 root，见 README）
- 批量基准：`scripts/benchmark_all.sh`（阶段三实现）

## 10. 常见误区

- **误区一：单次测量就能下结论。** 任务切换/频率毛刺会污染单样本（PDF 第 168 页）。
- **误区二：Debug 版测性能。** 未优化代码，结论无意义（PDF 第 85 页）。
- **误区三：单测即真相。** 单测程序小于缓存，看不到内存足迹代价（PDF 第 170 页）。
- **误区四：最快的就是最好的。** 内存足迹大的版本在最终程序中可能输给单测较慢的版本（PDF 第 170 页）。
- **误区五：只测最好情况。** 实时应用必须测最坏情况（PDF 第 171 页）。
- **误区六：忽略频率变化。** 用预热、core clock cycles 计数器或本机 invariant TSC（PDF 第 168～169 页）。

## 11. 实践任务

1. 用一个计算函数跑"无预热单次" vs "预热多轮取中位数"，观察差距。
2. 用 `-O0` 与 `-O3` 分别编译同一程序，对比运行时间（Debug vs Release）。
3. 数据规模从小到大（如 1K/64K/1M/16M 元素）测求和，找出缓存临界点（L1/L2/L3 变化）。
4. 用 `sudo perf stat -e cache-misses,branch-misses` 分析热点函数，判断瓶颈是内存还是分支。
5. 把大循环展开 8 次 vs 不展开，对比"单测"与"放进大程序上下文"的结果，体会 16.2 节陷阱。

## 12. 本章总结

- 正确测速：预热 + 多轮 + 中位数 + Release 构建 + 真实数据 + 校验和。
- 解读：冷/热缓存分别对应最坏/最好情况，按应用场景取舍。
- 单测陷阱：缓存大小与代码体积在单测中失真；要测内层循环与最终上下文。
- 实时应用测最坏情况。
- 工具：TSC、性能计数器（cache-misses/branch-misses/instructions）、perf。

## 13. 对应代码

本章对应实验（阶段三实现）：

- `src/01_profiling/` —— 手动计时、perf stat/record/report
- `src/18_benchmark/` —— Debug vs Release、预热 vs 冷启动、规模过小、消除代码、频率、干扰、结果波动
- `src/common/benchmark.h/.cpp` —— 共用的计时与防消除工具

> 状态：上述实验代码尚未实现（阶段三完成），届时更新本节链接。
