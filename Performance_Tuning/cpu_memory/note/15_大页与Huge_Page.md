# 15 大页与 Huge Page

> 对应 PDF：第 6.2.4 节 Optimizing TLB Usage（PDFp59）、第 7.5 节 Page Fault Optimization 的大页部分（PDFp87~89）、图 7.9（PDFp88~89）
> 本篇回答：为什么大页能大幅提速？TLB 优化的两个方向是什么？怎么在 Linux 上使用大页（hugetlbfs / SHM_HUGETLB / mmap）？大页的代价与限制？

## 1. 本章要解决的问题

- TLB 优化的两个方向：减少页面数、减少页表层级。
- 大页如何同时减少 TLB miss 与页错误？
- Linux 上获取大页的三种途径：hugetlbfs、System V 共享内存 SHM_HUGETLB、透明大页（现代补充）。
- 大页的物理连续性与 hugetlbfs 预留机制。
- 图 7.9：4KB vs 2MB 页的实际收益（2^20 字节 57% 快、512MB 38% 快）。

## 2. 前置知识

- note/14：TLB 容量、页面大小。
- note/13：页表层级。
- note/29（将写）：page fault 与 mmap/MAP_POPULATE。
- Linux 内存接口：mmap、madvise、shmget/shmat。

## 3. 核心概念

- **Huge Page（大页）**：比常规页（4KB）大得多的页（x86-64 常用 2MB，也有 1GB）。
- **THP（Transparent Huge Pages，透明大页）**：内核自动为大映射选择大页（现代补充，见 note/30）。
- **hugetlbfs**：预留大页的伪文件系统，系统管理员通过 `/proc/sys/vm/nr_hugepages` 预留。
- **MAP_HUGETLB**：mmap 使用大页的 flag。
- **SHM_HUGETLB**：System V 共享内存使用大页的 flag。
- **`/proc/meminfo` HugePages 字段**：Total/Free/Reserved/Surp 与 Hugepagesize。
- **TLB 覆盖**：TLB 条目数 × 页大小 = 可覆盖的内存总量。

## 4. 硬件工作流程

### 4.1 TLB 优化的两个方向（PDFp59）

```text
方向①：减少页数
   更少页面 → 更少 TLB miss → 更少页错误
   要求：任意时刻活跃的 TLB 条目尽量少

方向②：减少页表层级
   更少高层目录 → 更省内存 → 目录查找缓存命中率更高
   要求：地址空间使用区域尽量集中（大页/紧凑布局）
```

- TLB miss 是"持续惩罚"（TLB 小且频繁被刷），页错误是"一次性代价"。
- 程序跑得够久、热点执行够频繁时，TLB miss 总代价可超过页错误。
- 页优化不能只从 page fault 角度，还要从 TLB miss 角度考虑。

### 4.2 大页如何加速（图 7.9，PDFp88~89）

```text
4KB 页：大工作集 → 成千上万页 → TLB miss 泛滥
2MB 页：同一工作集 → 页数 ÷512 → TLB 覆盖骤增
```

- 2^20 字节工作集完全落在单个 2MB 页内 → 无 DTLB miss → 快 57%。
- 512MB 工作集：大页快 38%。
- 2MB 页、64 个 TLB 条目 → 覆盖 2^27 字节后曲线再次上升（图 7.9 平台结束点）。

### 4.3 Linux 获取大页的途径

```text
① hugetlbfs（推荐，明确）:
   echo N > /proc/sys/vm/nr_hugepages   （预留 N 个大页）
   mount -t hugetlbfs hugetlbfs /dev/hugetlb
   fd = open("/dev/hugetlb/file1", ...)
   p = mmap(NULL, LENGTH, PROT_READ|PROT_WRITE, fd, 0)

② System V 共享内存:
   id = shmget(ftok(key), LENGTH, SHM_HUGETLB|IPC_CREAT|...)
   p = shmat(id, NULL, 0)

③ 透明大页 THP（现代补充）:
   madvise(addr, len, MADV_HUGEPAGE)  或 mmap + MAP_HUGETLB
```

- LENGTH 必须是系统大页大小（`/proc/meminfo` Hugepagesize）的倍数。
- hugetlbfs 挂载点可用 `getmntent` 在运行时探测（论文给出示例函数）。
- 大页可用作共享（多进程 open 同一文件）与可执行（PROT_EXEC）。

## 5. PDF 核心观点

> 来源：PDF 第 59、87~89 页；对应章节 6.2.4、7.5、图 7.9。以下为概括。

1. **工作集应与缓存大小匹配**（PDFp59）：数据只需一次时不需匹配；需要多次的数据若超缓存，即使预取成功也慢。
2. **LLC 与 L1 优化可叠加**（PDFp59）：矩阵乘法数据放不进 LLC 时，可同时优化 L1 与 LLC 访问；LLC 数据块可更大。
3. **L1 行大小可硬编码，LLC 不行**（PDFp59）：L1 行大小跨代基本稳定；LLC 大小变化可达 8 倍以上，代码须动态适配。
4. **用 /sys 获取共享缓存信息**（PDFp59）：`/sys/devices/system/cpu/cpu*/cache` 取最后一级缓存（level 最大者），size ÷ shared_cpu_map 位数为每执行单元的安全下限。
5. **TLB 优化方向**（PDFp59）：(a) 减少使用页数 → 少 TLB miss；(b) 减少高层目录 → 省内存、目录查找命中率高。
6. **TLB miss vs 页错误**（PDFp59）：页错误单次更贵，但 TLB miss 是持续惩罚；长期运行且热点频繁的程序，TLB miss 总代价可超页错误。
7. **mmap 只改页表**（PDFp86）：demand-paging 下，mmap 不分配实际内存，首次访问触发页错误才分配。
8. **pagein 工具**（PDFp86~87）：基于 valgrind，记录页错误顺序与原因；可据此重排代码，让"先后执行的代码同页"，减少/推迟页错误；[17] 中仅重排函数就降低启动成本 5%。
9. **MAP_POPULATE 预故障**（PDFp87）：mmap 加 MAP_POPULATE 一次性预分配所有页，避免多次页错误；但粒度粗、若大量页长期不用则浪费，系统忙时可丢弃该优化。
10. **POSIX_MADV_WILLNEED**（PDFp87）：madvise 提示内核近期会用某些页，粒度比 MAP_POPULATE 细；对含大量未使用数据的映射文件优势大。
11. **大页减少页数与页错误**（PDFp87）：DSO/映射用页越少，页错误越少；IA-64/PPC64 常用 64KB 基础页。
12. **2MB 大页减少 511 个页错误**（PDFp87）：相比同容量 4KB 页（每个大页省 511 次页错误）。
13. **大页代价**（PDFp87）：物理内存必须连续；碎片化后 512 个连续页难找，尤其系统运行后；故启动时预留 hugetlbfs。
14. **图 7.9 数据**（PDFp88~89）：2^20 字节工作集大页快 57%（单 2MB 页、无 DTLB miss）；512MB 大页快 38%；2MB 页 64 个 TLB 条目覆盖 2^27 字节后曲线再升。
15. **文件映射用大页仍未落地**（PDFp89）：透明大页需内核判断映射大小，若后续要 4KB 粒度（如 mprotect）会浪费线性物理内存。

## 6. 通俗解释

大页就是把"内存的包装单位"变大：

> 4KB 页像一箱 512 瓶装的水瓶；2MB 页像一辆卡车。前台（TLB）一次只能记一条"货在哪个仓库"。
> 用 4KB 页，一车水要记 512 条；用 2MB 页，一条就够。
> 仓库地址表（TLB）本来就只有几十行，自然记大包装更划算。

为什么不能全部用大页？

> 因为大包装要求"一整片连续仓库"（物理连续），还要对齐——就像卡车必须停在整块场地正中，
> 停不进去就浪费半块场地。系统跑久了内存碎片化，很难凑出 512 个连续 4KB 页，
> 所以只能开机时（内存最整）预留一批（hugetlbfs）。

怎么用？

> 跟管理员预借几辆卡车（echo N > nr_hugepages），挂载专用停车场（mount hugetlbfs），
> 然后 mmap 一块大小必须是卡车容积整数倍的内存。现代内核还能"自动判断要不要用卡车"（THP）。

## 7. 示例分析

### 7.1 TLB 覆盖计算

- 256 个 DTLB 条目：
  - 4KB 页 → 覆盖 1MB；
  - 2MB 页 → 覆盖 512MB；
  - 1GB 页 → 覆盖 256GB。
- 图 7.9：64 个 TLB 条目 × 2MB = 2^27 字节（128MB）——超过后曲线再升。

### 7.2 页错误节省

- 2MB 大页 vs 4KB 页：2MB/4KB = 512 页，节省 511 次页错误/大页。
- 1GB 数据：4KB 页 262144 次页错误；2MB 页 512 次——差 512 倍。

### 7.3 图 7.9 转折点解读

- 2^20 字节（1MB）：落在单个 2MB 页 → 0 DTLB miss → 快 57%。
- 2^20~2^27：多个 2MB 页，但都在 64 个 TLB 条目覆盖内 → 平台约 250 周期。
- >2^27：TLB 条目耗尽 → 延迟再升。
- 4KB 页曲线：TLB 覆盖只有 64×4KB=256KB，更早开始 miss → 整体更慢。

## 8. 未优化代码

对应"默认 4KB 页 + 大工作集"的程序（TLB miss 高）。

```cpp
// bad.cpp: 大工作集，普通 malloc（4KB 页）
#include <vector>

int main() {
    constexpr int N = 1 << 26;   // 256MB
    std::vector<int> data(N, 1);
    long long sum = 0;
    for (int i = 0; i < N; ++i)
        sum += data[i];
    return sum == 0;
}
```

## 9. 优化后代码

对应"使用大页"的程序（THP hint，系统需支持；完整见 src/20）。

```cpp
// good.cpp: 使用透明大页提示（需内核支持 THP）
#include <vector>
#include <sys/mman.h>

int main() {
    constexpr int N = 1 << 26;
    void* p = mmap(nullptr, N * sizeof(int),
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED)
        madvise(p, N * sizeof(int), MADV_HUGEPAGE);  // 提示使用 2MB 页
    int* data = static_cast<int*>(p);
    for (int i = 0; i < N; ++i) data[i] = 1;
    long long sum = 0;
    for (int i = 0; i < N; ++i) sum += data[i];
    return sum == 0;
}
```

> 显式 hugetlbfs/HugeTLB 用法的完整可运行代码见 src/20_huge_pages。
> 程序必须先检查系统是否支持大页，不支持则输出提示并跳过，不得假设计算成功。

## 10. 为什么会更快

| 角度 | 4KB 页 | 2MB 页 |
|---|---|---|
| TLB 条目需求 | 256MB→65536 条 | 256MB→128 条 |
| dTLB-load-misses | 大量 | 极少 |
| 页表遍历 | 频繁 | 少（层级也更少） |
| Page Fault | 65536 次 | 128 次 |
| 页表目录内存 | 多 | 少 |

论文实测：1MB 工作集快 57%，512MB 快 38%。注意这些数字来自 2007 年机器与特定测试，仅作量级参考；本机以实测为准。

## 11. 如何验证

```bash
./build/20_huge_pages/huge_pages        # 自动检测并选择可用的大页路径
./build/19_page_size/page_size          # 4KB vs 大页对比
./scripts/perf_stat.sh ./build/20_huge_pages/huge_pages

# 检查大页配置
cat /proc/meminfo | grep -i hugepages
cat /sys/kernel/mm/transparent_hugepage/enabled
grep -i huge /proc/mounts
sudo sysctl vm.nr_hugepages              # 需 root 才能写
```

## 12. 实验结果应该怎么看

- 若 `HugePages_Total=0` 且 THP 关闭：程序应提示"无法验证大页"并跳过，这是正确行为，不是失败。
- 若可用：对比 dTLB-load-misses 与延迟台阶位置；大页应显著降低 TLB miss、右移台阶。
- 复现图 7.9 思路：固定工作集，分别用 4KB/2MB 页跑同一访问，记录每元素周期。

## 13. 常见误区

- **误区 1：大页谁都能随时用**。需要 hugetlbfs 预留或 THP 支持；HugePages_Total=0 时显式大页不可用。
- **误区 2：透明大页一定自动生效**。THP 的 enabled 值（always/madvise/never）决定行为；默认值因发行版而异。
- **误区 3：大页只省页错误**。它同时大幅减少 TLB miss 与页表遍历，这正是大工作集提速的主因。
- **误区 4：文件映射也能透明用大页**。论文时代不行；现代 THP 对文件映射支持仍有限（需实测/标注未验证）。
- **误区 5：2MB 页无浪费**。物理连续 + 对齐 → 平均浪费半页（1MB）。

## 14. 实践练习

1. 运行 src/20，确认本机大页可用性，记录 HugePages_Total/Hugepagesize。
2. 若可用，对比 4KB 与 2MB 页下的 dTLB-load-misses 与运行时间。
3. 计算：本机 DTLB 条目（如未知则用 64 假设并标注）分别能覆盖多少 4KB/2MB 工作集。
4. 解释图 7.9 中 2^20 字节处为何最快、2^27 后为何再升。
5. 尝试 `sudo sysctl vm.nr_hugepages=8`（若允许），用 hugetlbfs 分配一次大页并回滚。

## 15. 本章总结

- TLB 优化两方向：减少活跃页数、减少页表层级。
- 大页同时带来：更少 TLB miss、更少页错误、更少页表目录。
- Linux 途径：hugetlbfs（预留+挂载+mmap）、SHM_HUGETLB、THP/madvise（现代）。
- 大页代价：物理连续、半页浪费、碎片化后难分配 → 启动时预留。
- 论文实测：1MB 快 57%、512MB 快 38%（量级参考）。
- 对程序员：大工作集 + 性能关键时，大页是高性价比优化；须先探测系统支持。

## 16. 对应代码

- src/19_page_size/（页面大小影响）
- src/20_huge_pages/（THP / hugetlbfs / SHM_HUGETLB，环境检查与跳过）
- src/18_tlb_capacity/（TLB 容量，大页效果的上游现象）
