# 14 TLB 原理与优化

> 对应 PDF：第 4.3 节 Optimizing Page Table Access（PDFp38）、第 4.3.1 节 Caveats Of Using A TLB（PDFp39）、第 4.3.2 节 Influencing TLB Performance（PDFp40）、图 3.17（PDFp24）
> 本篇回答：TLB 到底是什么、为什么必要？TLB 刷新与 ASID 怎么权衡？大页如何缓解 TLB 压力？哪些因素决定 TLB 性能？

## 1. 本章要解决的问题

- 为什么不能只把页表项缓存进 L1d？
- TLB 缓存什么、用什么做 tag？
- 上下文切换/进出内核时 TLB 怎么处理？ASID 如何避免全刷？
- 页面大小如何影响 TLB 条目数与命中率？
- 大页的代价与 Linux hugetlbfs。

## 2. 前置知识

- note/12：页表遍历、TLB 概念。
- note/13：多级页表结构、层级数。
- 上下文切换、系统调用基本概念。

## 3. 核心概念

- **TLB（Translation Look-Aside Buffer）**：缓存"虚拟页→物理页"完整翻译结果的小缓存。
- **ITLB / DTLB**：指令/数据 TLB。
- **L1TLB / L2TLB**：一级（小、快、常全相联）与二级（大、慢、统一）TLB。
- **TLB Flush（TLB 刷新）**：使缓存条目失效。
- **ASID（Address Space ID）**：TLB tag 扩展，标识地址空间，避免上下文切换全刷。
- **Page Size（页面大小）**：4KB 常规页、2MB/4MB 大页、64KB（IA-64/PowerPC）。
- **hugetlbfs**：Linux 预留大页的文件系统。
- **`dTLB-load-misses`**：数据 TLB 加载未命中计数（perf 事件名，平台相关）。

## 4. 硬件工作流程

### 4.1 为什么需要 TLB（PDFp38）

```text
方案A：页表项缓存在 L1d
  每次翻译 = 4 级串行 L1d 查找 ≥ 12 周期，且 miss 概率高、抢缓存带宽 → 不行

方案B：TLB 缓存完整翻译结果
  用虚拟地址(去页内偏移)做 tag，命中即得物理页号 + 偏移 → 极快
```

- TLB 通常很小且必须极快；L1TLB 曾全相联 + LRU，近年变大并改为组相联。
- tag 匹配 → 直接合成物理地址（供指令与 L2 索引使用）。
- TLB miss → 页表遍历，昂贵。

### 4.2 TLB 与上下文切换（PDFp39）

```text
页表树改变后两种策略：
  ① 全刷 TLB：简单但贵（系统调用进出内核就可能触发）
  ② 扩展 tag（ASID）：给每个地址空间唯一 ID，切换不刷

单条目失效：按地址范围只作废相关条目（如 munmap 时）→ 便宜
ASID 问题：tag 位数有限，地址空间多 → ID 复用 → 需部分刷新
```

- Core2：128 ITLB + 256 DTLB，全刷浪费 100+/200+ 条目。
- 好处：同一进程线程切换不刷 TLB；进出内核/虚拟机翻译可保留。
- 现状：AMD Pacifica ASID（1 位，区分 VMM 与客户）、Intel VPID（每客户域一个，不能细分进程）。

### 4.3 影响 TLB 性能的因素（PDFp40）

```text
① 页面大小：页越大 → 每个翻译覆盖更多数据 → 需要 TLB 条目越少
   x86/x86-64: 4KB / 2MB / 4MB；IA-64/PowerPC: 64KB 基础页
② 页表层级数：页越大 → 偏移位多 → 目录级数少 → miss 遍历更便宜
③ 数据摆放：同时使用的数据放更少页面 → 少占 TLB 条目
④ 大页物理连续性：需连续物理内存；平均浪费半页；碎片化后难分配
```

- Linux 用 hugetlbfs 在启动时预留大页池；改池大小常需重启。
- 大页典型用户：数据库服务器。

## 5. PDF 核心观点

> 来源：PDF 第 38~40 页；对应章节 4.3、4.3.1、4.3.2、图 3.17。以下为概括。

1. **目录项当普通数据缓存不够**（PDFp38）：4 级 × 串行 ≥ 12 周期、miss 概率高、偷缓存带宽，流水线藏不住。
2. **TLB 缓存完整翻译**（PDFp38）：只缓存物理页号计算，tag 用虚拟地址去掉页内偏移；大量指令/对象共享同一 tag。
3. **TLB 要小且快**（PDFp38）：L1TLB 常全相联 + LRU，现转组相联；多级 TLB（ITLB/DTLB/L2TLB）与缓存类似。
4. **硬件预取不预取 TLB**（PDFp38）：可能发起无效页表遍历；TLB 要显式预取。
5. **TLB 全刷昂贵**（PDFp39）：系统调用进出内核可能触发；Core2 全刷浪费 100+ 条目；单条目/地址范围失效更便宜。
6. **ASID 扩展 tag**（PDFp39）：避免进出内核/虚拟机全刷；ID 位数有限需复用。
7. **大页减少翻译与层级**（PDFp40）：更多数据/指令进一个页；更少 TLB 项；miss 遍历更便宜。
8. **大页的代价**（PDFp40）：物理连续、平均浪费半页、碎片化后难找 512 连续页；hugetlbfs 启动预留、固定池。
9. **最小页大小增大也有问题**（PDFp40）：ELF 对齐限制（图 4.3 Align=2MB）；超过设计值可能无法装载。
10. **把同时使用的数据放更少页面**（PDFp40）：TLB 条目少，值得做的优化（类似缓存局部性，但对齐要求大）。

## 6. 通俗解释

TLB 就是**前台的"快速拨号本"**：

> 每次要打电话（翻译地址）都要翻 4 层总机目录（页表遍历），太慢。
> 于是前台把查过的结果记在小本子上：目标分机段 → 直接线路。
> 小本子小（TLB 小）但翻得飞快；翻不到（TLB miss）才回去翻 4 层目录。

上下文切换就像**换班**：

> 换一个班（进程）前台就要把小本子撕掉重记（TLB 全刷），代价大。
> 聪明做法是给每班发一个专属小本子（ASID），换班不撕，回来接着用。
> 问题是专属小本子的标签位有限，人太多就得共用、共用就得部分重记。

大页就像**把整层楼的电话合并成一条热线**：

> 一条热线（一个大页翻译）覆盖 512 个房间（4KB 页）。要打的电话都在一层时，
> 前台记一条就够（TLB 条目少）。代价：你得保证这一整层都是同一个人的（物理连续），
> 而且空房间也算你的（浪费半页）。

## 7. 示例分析

### 7.1 页面大小与 TLB 覆盖

- 4KB 页、256 个 DTLB 条目 → 最多覆盖 256×4KB = 1MB。
- 2MB 页、256 个 DTLB 条目 → 最多覆盖 512MB。
- 论文图 3.17：随机访问 + 大工作集，TLB miss 主导性能；用小块随机化（限制活跃页数）可显著改善。

### 7.2 大页 vs 4KB 页的 TLB 命中

- 4KB 页下 64MB 工作集 → 16384 个页 → 远超 DTLB → 大量 miss。
- 2MB 页下 64MB 工作集 → 32 个页 → 全部可驻留 TLB → 极少 miss。
- 论文 7.5 节示例：512MB 工作集大页快 38%（note/29 会引用）。

### 7.3 为什么 TLB 不能太大

- TLB 命中必须极快（每个绝对寻址指令都要用）。
- 大 TLB 变慢 → 无法做全相联 → 组相联 → 命中率反而受冲突影响。
- 上下文切换频繁全刷时，大 TLB 也填不满（论文原意）。

## 8. 未优化代码

对应"高 TLB 压力"的程序：随机访问覆盖大量页面。

```cpp
// bad.cpp: 随机访问，dTLB miss 高
#include <vector>
#include <random>

int main() {
    constexpr int N = 1 << 24;
    std::vector<int> data(N, 1);
    std::mt19937 rng(42);
    long long sum = 0;
    for (int i = 0; i < N; ++i)
        sum += data[static_cast<int>(rng() & (N - 1))];
    return sum == 0;
}
```

## 9. 优化后代码

对应"低 TLB 压力"的程序：顺序访问（页内连续）或显式使用大页。

```cpp
// good.cpp: 顺序访问，页连续
#include <vector>

int main() {
    constexpr int N = 1 << 24;
    std::vector<int> data(N, 1);
    long long sum = 0;
    for (int i = 0; i < N; ++i)
        sum += data[i];
    return sum == 0;
}
```

大页版本（见 src/20）：用 `mmap` + `MAP_HUGETLB`/`madvise(MADV_HUGEPAGE)`，需系统支持。

## 10. 为什么会更快

| 角度 | 随机访问 | 顺序访问 |
|---|---|---|
| 活跃页数 | 覆盖整个工作集 | 少量连续页 |
| dTLB-load-misses | 高 | 低 |
| 页表遍历次数 | 频繁 | 极少 |
| 每页访问元素 | 1 个 | 多个（页内连续） |
| 缓存命中 | 差 | 好 |

大页叠加效果：页覆盖范围 ×512，TLB miss 进一步骤减；若系统不预分配大页则无法验证（程序会提示）。

## 11. 如何验证

```bash
./build/18_tlb_capacity/tlb_capacity            # 页数 vs TLB 台阶
./build/19_page_size/page_size                  # 4KB vs 大页对比
./build/20_huge_pages/huge_pages                # THP / hugetlbfs
./scripts/perf_stat.sh ./build/18_tlb_capacity/tlb_capacity
```

查看 TLB 相关系统信息：

```bash
cat /proc/meminfo | grep -iE 'hugepages|directmap'
lscpu | grep -i tlb   # 部分 CPU 报告 TLB 大小
```

> 说明：许多现代 CPU 不直接暴露 TLB 容量；读不到则标注"当前环境或资料未验证"。

## 12. 实验结果应该怎么看

- src/18：曲线应在"活跃页数超过 TLB 容量"处出现台阶——跳升点对应 TLB 条目数 × 页大小。
- src/19/20：若大页可用，dTLB-load-misses 显著下降，延迟台阶右移。
- 不要只报"数字变大"；要解释台阶位置与 TLB 容量、页大小的乘积关系。
- perf 事件名（dTLB-load-misses）平台相关，脚本会自动容错。

## 13. 常见误区

- **误区 1：TLB miss = 缓存 miss**。TLB miss 触发页表遍历（可达 4 次主存访问），代价独立于数据缓存 miss。
- **误区 2：TLB 越大越好**。快是硬约束；上下文切换全刷使大 TLB 未必填满。
- **误区 3：硬件预取会预热 TLB**。不会——可能触发无效遍历，必须显式预取。
- **误区 4：大页零成本**。物理连续、半页浪费、碎片化、hugetlbfs 固定池都是代价。
- **误区 5：所有平台 TLB 行为相同**。条目数、关联度、ASID 支持因厂商/型号而异，需实测。

## 14. 实践练习

1. 运行 src/18，找出本机 DTLB 容量台阶，与 TLB 条目数×页大小对照。
2. 运行 src/19/20（若大页可用），比较 dTLB-load-misses 差异。
3. 计算：128 个 DTLB 条目，4KB vs 2MB 页各能覆盖多少工作集？
4. 解释论文图 3.17 中"小块随机化限制活跃页数"为何提升性能。
5. 讨论 ASID 位受限时会发生什么，以及为什么多进程场景仍受益。

## 15. 本章总结

- TLB 缓存完整地址翻译，tag 用虚拟地址（去页内偏移），小而快。
- 页表项缓存进 L1d 太慢（4 级串行），故需专用 TLB。
- TLB 刷新（上下文切换/进出内核）昂贵；ASID 扩展 tag 避免全刷，但 ID 有限。
- 大页减少翻译次数与页表层级，但需物理连续与 hugetlbfs 预留。
- 布局优化（集中放置、大页）与顺序访问是降低 TLB 压力的核心手段。

## 16. 对应代码

- src/18_tlb_capacity/（TLB 容量实验）
- src/19_page_size/（页面大小）
- src/20_huge_pages/（THP / hugetlbfs）
- src/04_stride_access/（步长与页边界的 TLB 交互）
