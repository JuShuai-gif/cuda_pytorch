# 16 NUMA 架构

> 对应 PDF：第 5 章 NUMA Support（PDFp43）、第 5.1 节 NUMA Hardware（PDFp43）、第 5.4 节 Remote Access Costs（PDFp45~46）、图 5.1、图 5.3、图 5.4
> 本篇回答：NUMA 为什么出现？NUMA 拓扑（超立方体/hop）如何组织？本地 vs 远程访问差多少？远程带宽比本地慢多少？

## 1. 本章要解决的问题

- 什么是 NUMA，为什么它是集成内存控制器的必然结果？
- NUMA 硬件如何连接（超立方体、hop、直径）？
- 1-hop/2-hop 访问的延迟与带宽代价（图 5.3、5.4）。
- NUMA factor 与节点距离。
- Linux 如何发布 NUMA 信息（sysfs node*、numa_maps）？

## 2. 前置知识

- note/01：集成内存控制器、NUMA 概念、NUMA factor。
- note/09：多核一致性、共享总线带宽。
- 超立方体（hypercube）拓扑概念。

## 3. 核心概念

- **NUMA（Non-Uniform Memory Architecture）**：不同 CPU 访问不同物理内存区域代价不同的架构。
- **NUMA Node（节点）**：CPU + 本地内存 + 本地互连接口的组合。
- **NUMA Factor**：远程访问相对本地访问的额外代价倍数。
- **Hop（跳数）**：访问远端节点需要跨越的互连链路数。
- **Diameter（直径）**：任意两节点间的最大距离（跳数）。
- **Hypercube（超立方体）**：一种高效连接拓扑，2^C 个节点、每节点 C 个互连、直径 C。
- **HyperTransport**：AMD 使用的处理器间互连（论文时代）。
- **Stripe（条带化）**：Linux 默认把内存按节点条带分配，均衡利用全部内存（见 note/17）。
- **`/sys/devices/system/node`**：sysfs 中的 NUMA 节点目录。
- **`/proc/PID/numa_maps`**：进程各映射在节点上的页数分布。

## 4. 硬件工作流程

### 4.1 NUMA 拓扑（论文图 2.3 + 图 5.1，PDFp4、p43）

```text
超立方体示例（C=3，8 节点，直径 3）：
       N0 ---- N1
      /  \    /  \
    N2 ---- N3   （二维超立方 = 四边形，直径 2）
     \  /    \  /
       N4 ---- N5
        ...
```

- 每节点有 C 个互连接口；节点数 = 2^C；直径 = C。
- 超立方体在"2^n 节点、n 条互连"的系统中直径最小。
- AMD 第一代 Opteron 每处理器 3 条 HyperTransport；至少一条要给 Southbridge → 实际可实现 C=2 超立方（4 节点）。

### 4.2 访问本地 vs 远程

```text
本地内存：CPU ──► 自己的内存控制器 ──► 本地 DRAM   （0 hop）
远程内存：CPU ──► 互连 ──► 对方内存控制器 ──► 对方 DRAM （1+ hop）
```

- 访问 CPU2 的内存：跨 1 条互连；访问 CPU4 的内存：跨 2 条（note/01 图 2.3）。
- 每跳都有额外成本 → NUMA factor。

### 4.3 Linux 发布 NUMA 信息

```text
/sys/devices/system/node/node*/cpumap   → 该节点包含哪些 CPU
/sys/devices/system/node/node*/distance  → 到各节点的相对代价（如本地 10，远程 20）
/proc/PID/numa_maps                      → 进程内存各映射分布在哪些节点（N0~N3 页数）
```

## 5. PDF 核心观点

> 来源：PDF 第 43、45~46 页；对应章节 5、5.1、5.4、图 5.1、5.3、5.4。以下为概括。

1. **NUMA 是集成内存控制器的必然结果**（PDFp43）：各处理器有本地内存后，访问远端内存要跨互连，代价不均。
2. **简单 NUMA 的 NUMA factor 低**（PDFp43）：处理器有本地内存、访问它比访问别处便宜；此类系统差异不大。
3. **共享北桥是大机器瓶颈**（PDFp43）：所有内存流量过北桥；多端口 RAM 太贵几乎不用。
4. **AMD 的互连模型**（PDFp43）：HyperTransport（源于 Digital）让非直连处理器的处理器也能访问内存；结构规模受直径限制。
5. **更大结构需要专业硬件**（PDFp43）：crossbar（如 Newisys Horus）、IBM x445（4U 8 路 ×2~4）、SGI Altix（NUMAlink，数千 CPU）——NUMA factor 动态变化，HPC/MPI 看重低延迟高带宽。
6. **集群不是 NUMA**（PDFp43）：无共享地址空间的联网机器不属于本文讨论范畴。
7. **超立方体拓扑**（PDFp43）：2^C 节点、直径 C，是同规模互连系统中直径最小的。
8. **远程访问代价**（PDFp45，图 5.3）：2-hop 读比 0-hop 慢 30%、2-hop 写慢 49%（相对 0-hop 读）；2-hop 写比 1-hop 写慢 17%。处理器与内存节点的相对位置影响很大。
9. **远程带宽**（PDFp46，图 5.4）：远程（1-hop）读总是慢 20%；工作集在缓存内时写/复制也慢 20%；工作集超缓存后写不再明显变慢（互连足够快，主存等待主导）。
10. **numa_maps 用途**（PDFp46，图 5.2）：可看出程序代码/脏页在哪个节点、只读映射在哪些节点，用于分析本地性。
11. **ACPI 数据可能不准**（PDFp45 脚注 26）：示例中声称所有远程代价相同（20），但实际至少有一对节点距离应更大——不要盲目相信系统报告。

## 6. 通俗解释

NUMA 就像**连锁超市的仓储**：

> 每个分店（CPU）都有自己后仓（本地内存），货架东西就近拿（快）。
> 但一个分店缺货要去别的分店调货（远程内存），得经过中转站（互连），慢一些。
> 调一次货叫 1 hop，转两次手叫 2 hop——越远越慢，这就是 NUMA factor。

超立方体拓扑，像"办公室的走廊设计"：

> 8 间办公室每人 3 条走廊，任何两人之间最多转 3 次身就到（直径 3）。
> 这种设计在同人数的办公室里是最省的（直径最小）。

Linux 怎么告诉你这些？

> `/sys/devices/system/node/node*` 里写着"这个节点管哪些 CPU"（cpumap）和"去每个节点要花多少钱"（distance）。
> `/proc/PID/numa_maps` 则告诉你你程序的每个内存区域分别在哪个节点上有多少页。

## 7. 示例分析

### 7.1 1-hop vs 2-hop 的代价（图 5.3）

- 基准：0-hop 读（本地读）。
- 1-hop 读：慢约 9%（论文正文）/图 5.4 实测慢 20%（不同机器/测量）。
- 2-hop 读：慢 30%。
- 2-hop 写：相对 0-hop 读慢 49%，比 1-hop 写慢 17%。
- 结论：跳数越多越慢，且写比读更敏感（需要所有权转移）。

### 7.2 远程写在大工作集下"不慢"的原因

- 工作集 > 缓存时，本地与远程都以主存等待为主，互连带宽足够跟上。
- 因此远程写的额外代价被主存等待淹没（图 5.4）。

### 7.3 从 numa_maps 判断程序局部性

- 若程序代码与脏页集中在 node3、只读共享库在其他节点：说明程序跑在 node3 的 CPU 上，代码本地化好。
- 若大量页分布在各节点（条带化默认），读时一半是远程 → 平均变慢。

## 8. 未优化代码

对应"忽略 NUMA"的程序：不绑核、默认条带分配，随机访问被摊到多节点。

```cpp
// bad.cpp: 多线程访问共享大数组，不关心节点/绑定
#include <vector>
#include <thread>
#include <numeric>

int main() {
    constexpr int N = 1 << 26;
    std::vector<int> data(N, 1);
    unsigned n = std::thread::hardware_concurrency();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n; ++t)
        pool.emplace_back([&, t] {
            long long s = 0;
            for (int i = t; i < N; i += (int)n) s += data[i];
            return s;   // 返回值被丢弃（示意）
        });
    for (auto &th : pool) th.join();
    return data[0] == 0;
}
```

## 9. 优化后代码

对应"尊重 NUMA"的程序：每线程本地分配数据 + 绑定节点（numactl 外层控制；代码见 src/24~26）。

```cpp
// good.cpp: 每线程独立局部数据（first-touch 引导到本地节点）
#include <vector>
#include <thread>

int main() {
    constexpr int N = 1 << 22;
    unsigned n = std::thread::hardware_concurrency();
    std::vector<std::vector<int>> per(n);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n; ++t)
        pool.emplace_back([&, t] {
            per[t].assign(N, 1);        // 由该线程触碰 → 落在本地节点
            long long s = 0;
            for (int v : per[t]) s += v;
            return s;
        });
    for (auto &th : pool) th.join();
    return per[0][0] == 0;
}
```

> 真正的 NUMA 行为必须用 numactl 绑定后实测，见 src/24、25、26。单节点机器上无法测量远程代价，须提示并跳过。

## 10. 为什么会更快

| 角度 | 忽略 NUMA | 本地分配+绑定 |
|---|---|---|
| 远程访问次数 | 高（条带/随机散布） | 极少 |
| 平均延迟 | 混入 1~2 hop 代价 | 基本 0 hop |
| 带宽 | 受互连限制 | 本地内存满带宽 |
| 缓存本地性 | 差 | 好 |

论文数据：1-hop 读慢约 9%~20%，2-hop 读慢 30%，2-hop 写慢 49%（图 5.3/5.4，具体取决于机器与测量）。

## 11. 如何验证

```bash
./scripts/system_info.sh
numactl --hardware                # 节点数、CPU 分布、distance
./build/24_numa_local_remote/numa_local_remote   # 多节点机器
./build/25_numa_first_touch/numa_first_touch     # first-touch 效应
cat /proc/self/numa_maps
cat /sys/devices/system/node/node*/distance
```

## 12. 实验结果应该怎么看

- 本机为单节点（`numactl --hardware` 只显示 node0）：NUMA 实验应自动跳过并提示，绝不编造远程数据。
- 多节点机器：对比本地/远程绑定的延迟与带宽，看是否与 distance 文件（如 10 vs 20）方向一致。
- first-touch 实验：先由哪个线程分配内存，页就落哪个节点——看绑定后是否改善。

## 13. 常见误区

- **误区 1：NUMA 只在服务器上存在**。现代桌面 CPU 也有多 CCD/CCX 的 NUMA 效应（现代补充，见 note/30），只是 factor 可能较小。
- **误区 2：numactl 不绑也能"自动优化"**。Linux 默认条带分配是为了均衡，不是最优；程序员需主动绑定/分配。
- **误区 3：远程访问永远慢很多**。大工作集下主存等待主导，远程写可能"不慢"（图 5.4）；小工作集远程代价明显。
- **误区 4：系统报告的 distance 一定准**。论文自己标注 ACPI 数据可能错误（脚注 26）。
- **误区 5：集群=NUMA**。集群无共享地址空间，不归 NUMA 讨论。

## 14. 实践练习

1. 运行 `numactl --hardware` 与 `cat /sys/devices/system/node/node*/distance`，记录本机拓扑。
2. 读论文图 5.3/5.4，解释"为什么 2-hop 写比 1-hop 写慢 17%"。
3. 若本机为多节点，用 numactl 分别绑定本地/远程跑 src/24，记录差异。
4. 解释 Linux 为什么默认条带分配内存而不是全本地（均衡 vs 最优）。
5. 阅读 `/proc/self/numa_maps`，识别哪些映射在本地、哪些在远程。

## 15. 本章总结

- NUMA 是集成内存控制器的必然结果；节点 = CPU + 本地内存 + 互连。
- 超立方体（2^C 节点、直径 C）是同规模中直径最小的互连拓扑。
- 远程访问代价随跳数增加：2-hop 读慢 30%、2-hop 写慢 49%（相对 0-hop 读）。
- 远程带宽：1-hop 读慢约 20%；大工作集下写不再明显慢（主存等待主导）。
- Linux 用 sysfs node*（cpumap/distance）与 /proc/PID/numa_maps 发布 NUMA 信息。
- 系统报告（ACPI）可能不准，须实测。

## 16. 对应代码

- src/24_numa_local_remote/（本地/远程访问对比）
- src/25_numa_first_touch/（first-touch 效应）
- src/26_numa_replication/（数据复制/镜像策略）
- scripts/numa_test.sh、scripts/system_info.sh（NUMA 环境探测）
