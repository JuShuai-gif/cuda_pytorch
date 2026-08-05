# 17 Linux NUMA 支持

> 对应 PDF：第 5.2 节 OS Support for NUMA（PDFp43~44）、第 5.3 节（进程/线程与内存迁移，PDFp44~45）、第 5 章"Published Information"部分（PDFp44~46，表 5.1~5.4）
> 本篇回答：操作系统在 NUMA 机器上要做什么？默认内存分配策略为什么是条带化？进程/内存迁移的代价？怎么从 sysfs 读出完整拓扑？

## 1. 本章要解决的问题

- OS 如何让 NUMA 机器的内存分配"尽量本地"？
- 为什么 DSO 只存在一份会让多数处理器远程访问？OS 的理想做法（镜像）是什么？
- 进程/线程迁移为什么被限制？内存迁移为什么昂贵？
- 默认分配策略为什么是条带（stripe）而非全本地？
- 如何从 /sys 组合出完整机器拓扑（缓存共享 + 处理器拓扑 + 节点距离）？

## 2. 前置知识

- note/16：NUMA 概念、hop、节点。
- note/09：缓存一致性、共享缓存。
- 基本 OS 调度概念：进程迁移、上下文切换。

## 3. 核心概念

- **Local Allocation（本地分配）**：内存尽可能从运行处理器所在节点分配。
- **DSO Mirroring（DSO 镜像）**：把共享库副本复制到各节点，减少远程访问（优化，非必需）。
- **Migration（迁移）**：进程迁移（换 CPU）或内存迁移（页搬到新节点）。
- **Page Migration（页迁移）**：把进程内存页搬到离新 CPU 更近的节点，昂贵、需停进程。
- **Stripe / Interleave（条带化/交错分配）**：内存按节点轮转分配，均衡利用全部内存。
- **Copy-On-Write（COW）**：写时复制；多进程共享只读页，写时才复制。
- **sysfs 拓扑文件**：`cpu*/cache`、`cpu*/topology`、`node*/cpumap`、`node*/distance`。

## 4. 硬件工作流程

### 4.1 OS 的 NUMA 职责

```text
进程跑在 nodeX 的 CPU 上
  └─► 内存尽量从 nodeX 分配（本地）
否则每个指令都远程取代码/数据 → 慢

特殊情形：DSO 只存在一份
  多个处理器共用 libc → 多数处理器要远程读
  OS 理想：把 DSO 镜像到各节点（优化，难实现，支持有限）
```

### 4.2 迁移的两难

```text
进程迁移（换 CPU）：
  缓存内容丢失 → 尽量避免
  NUMA 下还要保证新 CPU 的访存代价不高于旧 CPU

内存迁移（页搬到新节点）：
  可能复制海量内存、要短暂停进程
  OS 应避免，除非万不得已
```

### 4.3 默认策略：条带化

```text
只按本地分配 → 大进程可能耗尽本地节点内存
默认改为条带（interleave）：
  保证所有节点内存均衡使用
  副作用：进程可在节点间自由迁移（平均访存代价不变）
  小 NUMA factor 下可接受，但非最优
```

### 4.4 用 sysfs 拼完整拓扑（表 5.1~5.4，PDFp44~45）

```text
① cpu*/cache/index*：type(Data/Instruction/Unified)、level(1/2/3)、shared_cpu_map(哪些 CPU 共享)
② cpu*/topology：core_id、thread_siblings、physical_package_id
③ node*/cpumap：节点包含哪些 CPU
④ node*/distance：到各节点的相对代价

组合即可得出：哪些核共享 L1/L2/L3、哪个核属哪个包、哪个节点管哪些 CPU、节点间距离
```

示例：Core 2 QX6700（表 5.1）→ L1d/L1i 各自独立（shared_cpu_map 单 bit）、L2 由 cpu0+cpu1 或 cpu2+cpu3 共享。
Opteron 8 路（表 5.2~5.4）→ 4 处理器、每处理器双核、无共享缓存、每处理器一个节点、distance 本地 10/远程 20。

## 5. PDF 核心观点

> 来源：PDF 第 43~46 页；对应章节 5.2、5.3、"Published Information"、表 5.1~5.4、图 5.2。以下为概括。

1. **内存应尽量本地分配**（PDFp43）：否则每个指令都远程访问代码与数据。
2. **DSO 镜像**（PDFp43~44）：libc 等只存在一份，多数处理器远程访问；OS 理想是每节点放本地副本，但难实现、支持有限，这是优化而非需求。
3. **避免进程迁移**（PDFp44）：迁移丢缓存；NUMA 下新 CPU 访存代价不应高于旧 CPU；实在要迁则优先选代价不变的目标。
4. **内存迁移昂贵**（PDFp44）：可能复制海量内存、需短暂停进程、还有一堆前提条件；OS 应尽量不迁移。
5. **默认条带化**（PDFp44）：内存使用不均会导致本地内存耗尽；条带保证均衡、允许自由迁移；但小 factor 下仍非最优。
6. **sysfs 缓存拓扑**（PDFp44，表 5.1）：`cpu*/cache/index*` 的 type/level/shared_cpu_map 告诉你缓存类型、级别、共享关系。
7. **处理器拓扑**（PDFp44，表 5.3）：`cpu*/topology` 的 core_id、thread_siblings、physical_package_id 帮你分辨超线程/核心/物理包。
8. **节点信息**（PDFp45，表 5.4）：`node*/cpumap` 与 `node*/distance` 补全 NUMA 图景；distance 是相对代价（示例本地 10 远程 20）。
9. **numa_maps**（PDFp46，图 5.2）：`/proc/PID/numa_maps` 展示进程各映射在 N0~N3 的页数，可用于分析本地性。
10. **信息已齐全但难用**（PDFp46）：论文预告第 6.5 节会给出更易用的接口。

## 6. 通俗解释

Linux 在 NUMA 机器上的角色，像**配送调度员**：

> 原则 1：工人（进程）在哪干活，货（内存）就尽量放哪个仓库（本地节点）——否则每次拿料都要跑远路。
> 原则 2：共享工具（DSO/libc）最好每个仓库放一份（镜像）——但成本高，一般做不到。
> 原则 3：别让工人随便换车间（进程迁移丢缓存）；真要换，别换到离仓库更远的地方。
> 原则 4：默认"每家仓库轮流放货"（条带化）——保证所有仓库都不空，虽然对单进程不一定最优。

为什么要条带化而不是全本地？

> 一个大程序把本地仓库塞爆，别人没货用；轮流放至少每家都有。代价是每个进程平均一半货在远程，
> 访问略慢但"能跑"且允许自由换车间。

怎么知道机器长什么样？

> 把三个文件拼起来：缓存共享（谁和谁共用 L2/L3）、处理器拓扑（谁是超线程/核心/包）、
> 节点信息（哪个 CPU 归哪个节点、节点间多贵）。论文用表 5.1~5.4 演示了怎么拼。

## 7. 示例分析

### 7.1 为什么 libc 只有一份是问题

- libc 被所有进程使用，但物理内存里只有一份。
- 跑在 node0 的进程能本地读，node1~3 的进程每次读 libc 代码都是远程（1~2 hop）。
- OS 若在每节点放副本（镜像），所有节点本地读；但维护一致性/内存翻倍是代价。

### 7.2 条带化 vs 本地分配

- 本地分配：进程 A 在 node0 用 30GB，node0 只有 32GB → node0 耗尽，其它节点空闲。
- 条带化：30GB 平均摊到 4 节点（每节点 ~7.5GB），不耗尽；但进程访问一半内存是远程。
- 论文结论：均衡优先，非最优可接受；需要最优时用第 6.5 节接口干预。

### 7.3 用表 5.2~5.4 推断机器

- 表 5.2：8 个 CPU、每 CPU 3 个缓存（L1i/L1d/L2），无共享 → 每核独立缓存。
- 表 5.3：thread_siblings 单 bit（无超线程）、physical_package_id 0~3（4 个物理包）、core_id 0/1（每包双核）。
- 表 5.4：cpumap 00000003/0c/30/c0 → 每节点 2 个 CPU；distance 本地 10、其余 20。
- 综合：4 路双核 Opteron，每处理器一个节点，共 4 节点。

## 8. 未优化代码

对应"默认分配，不做任何 NUMA 感知"的程序（由内核条带化 + 可能迁移）。

```cpp
// bad.cpp: 大数组 + 多线程，不控制节点与绑定
#include <vector>
#include <thread>

int main() {
    constexpr int N = 1 << 26;
    std::vector<int> data(N, 1);
    unsigned n = std::thread::hardware_concurrency();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n; ++t)
        pool.emplace_back([&, t] {
            long long s = 0;
            for (int i = t; i < N; i += (int)n) s += data[i];
            return s;
        });
    for (auto &th : pool) th.join();
    return data[0] == 0;
}
```

## 9. 优化后代码

对应"NUMA 感知"的程序：绑定节点 + 本地分配（numactl 外层绑定，代码层面用 first-touch）。

```bash
# 外层绑定：进程与内存都绑定到 node0
numactl --cpunodebind=0 --membind=0 ./build/25_numa_first_touch/numa_first_touch
```

```cpp
// good.cpp: 每线程触碰自己的局部块（first-touch → 本地页）
#include <vector>
#include <thread>

int main() {
    constexpr int N = 1 << 26;
    unsigned n = std::thread::hardware_concurrency();
    std::vector<int> data(N, 0);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n; ++t)
        pool.emplace_back([&, t] {
            int lo = (int)(N / n * t), hi = (int)(N / n * (t + 1));
            for (int i = lo; i < hi; ++i) data[i] = 1;  // 先触碰 → 页落本地节点
        });
    for (auto &th : pool) th.join();
    return data[0] == 0;
}
```

> first-touch 的完整机制与验证见 src/25。注意：绑核与内存分配必须在 numactl/libnuma 支持下才能生效。

## 10. 为什么会更快

| 角度 | 默认（条带+可迁移） | 绑定+本地分配 |
|---|---|---|
| 远程访问 | 平均约一半页远程 | 基本全本地 |
| 缓存本地性 | 迁移丢缓存 | 固定核心缓存热 |
| 平均访存延迟 | 混入 hop 代价 | 0 hop |
| 带宽 | 受互连影响 | 本地满带宽 |

论文数据：1-hop 读慢约 9%~20%、2-hop 慢 30%（图 5.3/5.4）——具体取决于机器，需实测。

## 11. 如何验证

```bash
numactl --hardware
./scripts/numa_test.sh                        # 检测并跳过单节点
./build/25_numa_first_touch/numa_first_touch
./build/24_numa_local_remote/numa_local_remote
cat /proc/self/numa_maps
cat /sys/devices/system/node/node*/distance
```

## 12. 实验结果应该怎么看

- 单节点机器：numa_test.sh 与 NUMA 实验应提示"单节点，跳过远程测试"，这是正确行为。
- 多节点机器：对比绑定前/后的访问时间与 numa_maps 中页分布，验证 first-touch 生效。
- 关注"页迁移是否发生"：`/proc/self/numa_maps` 的页分布变化能反映。

## 13. 常见误区

- **误区 1：Linux 默认就最优**。默认条带是为均衡，不是为单进程最优。
- **误区 2：内存迁移很常见**。它昂贵且需停进程，OS 尽量避免。
- **误区 3：DSO 镜像一定存在**。是优化、难实现、支持有限；多数系统 libc 仍单份远程共享。
- **误区 4：distance 值 = 真实延迟比**。是相对代价的估计，且 ACPI 数据可能不准（论文脚注 26）。
- **误区 5：绑定只对内存分配有意义**。绑核还影响缓存局部性（避免迁移丢缓存）。

## 14. 实践练习

1. 用 sysfs 三个来源（cache/topology/node）拼出本机完整拓扑，与 `lscpu`、`numactl` 对照。
2. 解释"为什么默认条带分配比全本地更稳"。
3. 在多节点机器上跑 src/25，比较绑定前后 numa_maps 的页分布。
4. 阅读 /proc/self/numa_maps，解释为什么只读映射（如 locale-archive）可能在其他节点。
5. 讨论：什么情况下"进程迁移"的代价可以接受？（负载均衡 vs 缓存/NUMA 损失）

## 15. 本章总结

- OS 在 NUMA 上的目标：内存尽量本地、避免无谓迁移、均衡利用全部内存。
- DSO 镜像与页迁移都是昂贵优化，支持有限，OS 尽量不做。
- 默认条带化保证均衡与可迁移，但非单进程最优。
- sysfs（cache + topology + node）提供完整拓扑；numa_maps 提供进程级分布。
- 程序员要主动绑定节点与分配本地内存，才能超越默认策略。

## 16. 对应代码

- src/24_numa_local_remote/、src/25_numa_first_touch/、src/26_numa_replication/
- scripts/numa_test.sh、scripts/system_info.sh
