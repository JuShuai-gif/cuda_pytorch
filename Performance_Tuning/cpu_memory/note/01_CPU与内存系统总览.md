# 01 CPU 与内存系统总览

> 对应 PDF：第 1 章 Introduction（PDFp1）、第 2 章 Commodity Hardware Today（PDFp3~4）、第 2.3 节 Other Main Memory Users（PDFp12~13）
> 本篇解决"这套知识体系的出发点"和"历史背景下的硬件蓝图"两大问题。

## 1. 本章要解决的问题

- 为什么现在的程序性能瓶颈在内存，而不是 CPU？
- 一台计算机的主板/芯片组级硬件是怎么组织 CPU、内存、I/O 的？
- 为什么系统性能工程师需要理解内存子系统？
- 谁在"抢"内存带宽？（不只是 CPU）

## 2. 前置知识

- CPU 与寄存器：CPU 内部最快的存储，一条指令即可访问。
- 内存（RAM）：程序运行时存放代码与数据的地方。
- 总线（Bus）：连接多个硬件组件的共享通信线路。
- 进程与虚拟地址空间：每个进程"以为"自己独占内存（第 12 章展开）。
- 基本概念词：Northbridge、Southbridge、FSB、DMA、NUMA（本章先建立直觉，细节后文展开）。

## 3. 核心概念

- **Commodity Hardware（通用硬件）**：大规模量产、价格便宜、兼容性强的 PC/服务器硬件。论文声明只讨论这一类硬件，不讨论专用大型机。
- **Northbridge（北桥）**：连接 CPU、内存与高速 I/O 的芯片，是 2000 年代芯片组中靠近 CPU 的那一半。
- **Southbridge（南桥 / I/O 桥）**：负责与慢速设备（SATA、USB、PCI、声卡等）通信的另一半芯片。
- **Front Side Bus（FSB，前端总线）**：CPU 与北桥之间的共享总线，当年 CPU 所有访存和核间通信都走这里。
- **Memory Controller（内存控制器）**：决定用什么类型 RAM、负责把 CPU 的访存请求转换为对 RAM 芯片的行列操作。
- **DMA（Direct Memory Access，直接内存访问）**：允许设备绕过 CPU 直接读写内存的技术。
- **Integrated Memory Controller（集成内存控制器）**：把内存控制器放进 CPU 内部，每个 CPU 就近挂本地内存。
- **NUMA（Non-Uniform Memory Architecture，非均匀内存架构）**：不同 CPU 访问不同物理内存区域成本不同的架构。
- **NUMA Factor**：访问远端内存相对本地内存额外花费的时间倍数。
- **Single-port RAM**：同一时刻只能被一个请求者访问的内存端口。

> 术语注：Northbridge/Southbridge/FSB 属于论文编写年代（2007）的结构，见"历史背景"说明。

## 4. 硬件工作流程

### 4.1 2007 年的经典双桥结构（论文图 2.1，PDFp3）

```text
              +--------------------------------------+
              |              CPU1       CPU2         |
              |                  |        |          |
              +------------------+--------+----------+
                                 |      FSB
              RAM ---------------+  Northbridge(含内存控制器)
                                 |  |
              PCI-E ---------------|  Southbridge(I/O桥)
                                  |     |  |
                                 SATA  USB
```

关键结论（论文 PDFp4 原文观点概括）：

- CPU 之间通信、CPU 与 RAM 通信、CPU 与南桥设备通信，**全部都要经过北桥**；
- RAM 只有单端口；
- FSB 是唯一瓶颈通道。

### 4.2 现代（集成内存控制器）结构（论文图 2.3，PDFp4，也是 NUMA 的起源）

```text
  RAM --- CPU1 ---+--- CPU2 --- RAM
                  | interconnect
  RAM --- CPU3 ---+--- CPU4 --- RAM
                  |
              Southbridge --- SATA/USB
```

- 每个 CPU 自带内存控制器，本地内存访问快，远端内存访问要经过互联线，出现 NUMA。

> 历史背景：图 2.2（PDFp4）还展示了"北桥外接多个内存控制器"的过渡方案，用于增加内存通道数。
> 现代补充：现代 CPU（Intel/AMD）全部使用集成内存控制器，FSB/北桥结构已不存在（详见 note/30）。

## 5. PDF 核心观点

> 来源：PDF 第 1~4 页、第 12~13 页；对应章节 1、2、2.3。以下为概括，非逐句翻译。

1. **CPU 与内存速度失衡是根本问题**（PDFp1）：早期各部件性能平衡，后来硬件开发商各自优化子系统，内存子系统因成本原因提升缓慢，成为瓶颈。
2. **解决存储瓶颈用软件，解决内存瓶颈必须动硬件**（PDFp1）：磁盘慢可以用 OS 缓存 + 存储设备自带缓存缓解；而内存慢几乎只能靠硬件手段（缓存、内存控制器设计、DMA）解决。
3. **缓存需要程序员配合**（PDFp1）：硬件缓存设计得再巧妙，也需要程序写出有局部性的访问模式才能发挥最大作用——这是全书的出发点。
4. **通用硬件的标准形态（2007 年预测）**（PDFp3）：四路插槽 × 四核 CPU × 超线程 = 最高 64 个虚拟处理器，是当时数据中心"甜点位"。
5. **DMA 缓解 CPU 负担但加剧带宽竞争**（PDFp4、PDFp13）：设备直接访问内存降低了 CPU 工作量，但 DMA 请求与 CPU 访存争抢同一带宽。
6. **双通道/多通道增加带宽**（PDFp4）：DDR2 时代两个内存通道带宽翻倍，FB-DRAM 可达 6 通道。
7. **NUMA 是集成内存控制器的必然结果**（PDFp4）：每个 CPU 有本地内存后，跨 CPU 访问内存就存在不同代价。
8. **时序数字示例**（PDFp12, 2.2.5 Conclusions）：Core 2 @ 2.933GHz + 1.066GHz 四倍泵 FSB，时钟比 11:1——内存总线上每停滞 1 个周期，CPU 损失 11 个周期。
9. **不止 CPU 用内存**（PDFp13, 2.3）：网卡、存储控制器用 DMA；共享显卡（无显存）系统把主存当显存用，1024×768@60Hz 16bpp 就要 94MB/s 带宽。

## 6. 通俗解释

用一句话概括全书的动机：

> CPU 就像一位极其高效的员工，而内存像一个大仓库。员工去仓库取一次货要花很长时间（几百个周期）。
> 于是硬件设计师在员工工位旁放了几层小货架（Cache），货架越近越快但越小。
> 但这套货架系统只有在"你用得巧"时才有用——如果你每次都跳着乱翻仓库（随机访问），货架形同虚设。

本章讲的"南北桥 + FSB"就是这台机器的老版物流管线：所有货（数据）都要先到北桥中转站，再进 CPU。现代机器把中转站拆到每个 CPU 家门口（集成内存控制器），于是"离你近的仓库"和"离你远的仓库"访问速度不同了——这就是 NUMA。

## 7. 示例分析

### 7.1 为什么时钟比 11:1 很致命

- 假设 CPU 频率 2.933GHz，一个时钟周期约 0.34ns。
- 内存总线频率 1.066GHz（四倍泵），内存周期约 0.94ns。
- 内存侧 1 个周期 ≈ CPU 侧 11 个周期。

这意味着一次纯内存访问（约几百个内存周期）会让 CPU 白白等待几千个 CPU 周期，必须靠 Cache 挡住绝大多数访问。

### 7.2 为什么 DMA 是"双刃剑"

- 网卡收包直接 DMA 进内存：CPU 不参与，省了中断/搬运开销。
- 但 DMA 占用了内存控制器带宽；若 CPU 正好在等内存数据，就会多等。

## 8. 未优化代码

对应"忽视了内存子系统存在"的典型程序：每次循环都直接访问一个跨越大数组的元素，没有任何结构设计。

```cpp
// bad.cpp: 无序、大跨度、无法利用任何缓存效果
#include <vector>
#include <numeric>
#include <random>

int main() {
    constexpr int N = 1 << 24;
    std::vector<int> data(N);
    std::mt19937 rng(42);
    for (int i = 0; i < N; ++i) data[i] = i;

    // 固定随机种子，乱序累加，破坏空间局部性
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);

    long long sum = 0;
    for (int v : order) sum += data[v];
    return sum == 0;  // 防优化
}
```

## 9. 优化后代码

对应"意识到内存子系统"的程序：用连续顺序访问 + 局部变量累加，最大限度利用缓存与硬件预取。

```cpp
// good.cpp: 顺序访问，空间局部性极佳，预取友好
#include <vector>
#include <numeric>

int main() {
    constexpr int N = 1 << 24;
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) data[i] = i;

    long long sum = 0;          // 寄存器/栈局部变量
    for (int i = 0; i < N; ++i) // 顺序遍历
        sum += data[i];
    return sum == 0;            // 防优化
}
```

（更完整的顺序 vs 随机对比见 src/02_sequential_random_access。）

## 10. 为什么会更快

| 角度 | bad.cpp（随机） | good.cpp（顺序） |
|---|---|---|
| Cache Line 利用率 | 每访问一个 int 就浪费整条 64B Cache Line | 每条 Cache Line 的 16 个 int 全部用上 |
| L1/L2/LLC 命中率 | 命中率极低，几乎每次 miss | 顺序访问几乎全命中 + 硬件预取 |
| DRAM 访问次数 | 每个元素一次 DRAM 访问 | 每 64B 才一次 |
| 硬件预取 | 无法识别随机模式 | 完美匹配顺序预取 |
| TLB Miss | 每条新页面都可能 miss | 顺序访问一条页面后连续命中 |
| Page Fault | 两者初次分配相近 | 相同 |
| 内存带宽 | 被延迟而非带宽限制 | 接近峰值带宽 |

这里 bad/good 只是概念示例，真正的量化对比交给 src/02 实验用多轮统计输出，不在此编造数字。

## 11. 如何验证

编译：

```bash
cd src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

运行顺序/随机对比：

```bash
./build/02_sequential_random_access/sequential_random_access
```

查看系统硬件蓝图：

```bash
./scripts/system_info.sh
lscpu | grep -E 'Model name|Cache|Socket|NUMA'
numactl --hardware
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
grep -E 'HugePages|Hugepagesize' /proc/meminfo
```

用 perf 观察内存事件（事件名随 CPU 而异，脚本会容错）：

```bash
./scripts/perf_stat.sh ./build/02_sequential_random_access/sequential_random_access
```

## 12. 实验结果应该怎么看

- **ns/element**：每个元素平均访问耗时。顺序访问应远小于随机访问；若两者接近，说明工作集太大或数据未命中缓存。
- **GB/s**：吞吐。顺序访问应接近平台内存带宽上限；随机访问受延迟限制，GB/s 会很低。
- **checksum**：用于确认两次对比计算的是同一份数据，避免优化器做手脚。
- **cache-misses / dTLB-load-misses**：如果随机访问的 cache-misses 比例远高于顺序访问，正印证了第 10 节的分析。
- 不要只看一次运行；本项目所有实验都输出 mean/median/min/stddev，看趋势而非单点。

## 13. 常见误区

- **误区 1：内存访问就是"访问内存"这么简单**。实际上它要经过 TLB → Cache → 内存控制器 → DRAM 多级流水，每一级都有命中/未命中之分。
- **误区 2：SSD/网络快就能解决一切**。存储和网络瓶颈能用软件缓解，但内存带宽是硬件墙，绕不开。
- **误区 3：现代机器没有北桥 FSB，所以论文过时**。恰恰相反：FSB/南北桥只是论文给的历史蓝图，其"共享带宽、多路争抢"的思想直接延续到现代的片上互联与 NUMA。
- **误区 4：DMA 越多越好**。DMA 减轻 CPU 负载，但抢占内存带宽，需统筹。
- **误区 5：以为"延迟=带宽"**。二者独立：延迟决定单次访问多快返回，带宽决定单位时间能搬运多少。随机访问撞延迟墙，顺序访问撞带宽墙。

## 14. 实践练习

1. 运行 `lscpu`，找出本机 Cache 各级大小、Cache Line 大小、NUMA 节点数；把它们与论文图 2.1/2.3 的结构对应起来。
2. 查看 `numactl --hardware` 与 `/proc/meminfo` 的 Huge Pages 信息，判断本机是否 NUMA。
3. 读论文 PDFp3 图 2.1 与 PDFp4 图 2.3，画出"南北桥结构"与"集成内存控制器结构"两张 ASCII 图并互相对照。
4. 用 `/sys/devices/system/cpu/cpu0/cache/` 下列出 index0~indexN 的 type/level/size/ways，对应 L1d/L1i/L2/L3。
5. 运行 src/02 顺序/随机对比，记录两组 ns/element 与 GB/s，解释差异来源。

## 15. 本章总结

- 全书动机：CPU 与内存速度失衡，内存成为性能瓶颈，硬件缓存需要程序员配合。
- 历史蓝图：CPU ←FSB→ 北桥（含内存控制器）←→ 南桥 ←→ 设备；单端口 RAM、共享总线是核心约束。
- 演进方向：外置多内存控制器 → 集成内存控制器 → NUMA。
- 竞争带宽的还有 DMA 设备与共享显存系统。
- 从本章开始，后续笔记将逐层深入 DRAM、Cache、TLB、NUMA，并最终回到"程序员怎么写代码才快"。

## 16. 对应代码

- src/02_sequential_random_access/（顺序 vs 随机访问的完整对比实验）
- src/24_numa_local_remote/（NUMA 本地/远程访问，需 libnuma 且多节点机器）
- scripts/system_info.sh（硬件蓝图探测）
