# 29 Page Fault 分析

> 对应 PDF：第 7.5 节 Page Fault Optimization（PDFp86~88）、图 7.8、图 7.9（大页部分在 note/15 详述）
> 本篇回答：页错误何时发生、为什么贵？minor/major 怎么区分？怎么测量与优化启动页错误？MAP_POPULATE 与 POSIX_MADV_WILLNEED 怎么用？代码重排如何减少页错误？

## 1. 本章要解决的问题

- demand paging：mmap 只改页表，内存何时真正分配？
- 页错误处理为什么昂贵（进内核、找页、清页、改页表、同步）？
- minor vs major 页错误。
- pagein 工具与代码重排减少启动页错误。
- MAP_POPULATE 与 posix_madvise(POSIX_MADV_WILLNEED) 的取舍。

## 2. 前置知识

- note/12、13：页表、demand paging。
- note/15：大页减少页错误（本笔记引用图 7.9）。
- mmap、madvise 基础。

## 3. 核心概念

- **Demand Paging（按需分页）**：页在首次访问时才分配/读入。
- **Page Fault（页错误）**：访问未映射/未分配页时触发，内核处理。
- **Minor Page Fault**：页内容已在内存（首次匿名页、COW、已有内容）。
- **Major Page Fault**：需从磁盘读入（文件映射、换出页）。
- **MAP_POPULATE**：mmap 时预分配所有页。
- **POSIX_MADV_WILLNEED / posix_madvise**：提示内核近期需要某些页。
- **pagein 工具**：记录页错误的顺序与原因（论文作者基于 valgrind 开发）。

## 4. 硬件工作流程

### 4.1 mmap 与首次访问（PDFp86）

```text
mmap()：只改页表
  文件映射 → 记录底层数据位置
  匿名映射 → 标记"访问时提供清零页"
  不分配实际内存！

首次读/写某页 → Page Fault
  内核：进内核 → 用页表树确定内容 → 找页/读盘 → 清页/填充 → 改页表 → 同步 → 返回
```

- 每页使用一次就发生一次页错误处理。

### 4.2 页错误代价（PDFp80、86）

- major：需读磁盘（交换/文件）→ 显著更贵。
- minor：内容已在内存，但也不便宜：进内核、找页、清页（或填充）、改页表、与读写页表的任务同步。

### 4.3 pagein 输出（图 7.8，PDFp86）

```text
第1列: 序号
第2列: 页地址
第3列: C(代码)/D(数据)
第4列: 距首个页错误的周期数
其余: valgrind 尝试命名的触发地址
```

- 示例：第一条指令在 `3000000B50`，触发 `3000000000` 页入；第 3320 周期就访问了变量页（call 指令写返回地址到栈）。
- 注意：valgrind 会引入伪影（如它自己的内部栈），解读时需留意。
- 用途：找出"先后执行的代码"应在同一页，重排可避免/推迟页错误。

### 4.4 减少页错误的主动手段（PDFp87）

```text
MAP_POPULATE：mmap 时预分配所有页
  优点：避免多次页错误（一次较贵的 mmap）
  缺点：粒度粗；大量页若不用则浪费；过早占用内存
  注意：是优化，系统忙时可丢弃（届时照常页错误）

posix_madvise(POSIX_MADV_WILLNEED)：
  提示内核近期需要某些页 → 可预取
  优点：粒度细（单页/页范围），适合"映射文件但只用到一部分"
```

### 4.5 被动手段：大页（PDFp87）

- 页越大 → 需要的页数越少 → 页错误越少。
- 2MB 大页 vs 4KB 页：每个大页少 511 次页错误。
- 图 7.9：2^20 字节工作集完全落入单 2MB 页 → 0 DTLB miss → 快 57%（详见 note/15）。

## 5. PDF 核心观点

> 来源：PDF 第 86~88 页；对应章节 7.5、图 7.8、图 7.9。以下为概括。

1. **mmap 只改页表**（PDFp86）：文件映射记底层位置，匿名映射记"访问时清零"；实际分配在首次访问时。
2. **页错误处理不便宜**（PDFp86）：进内核、找页、清页/填充、改页表、同步——每页用一次发生一次。
3. **减少页总数**（PDFp86）：优化代码大小；重排代码让特定路径（如启动）触碰的页最少。
4. **pagein 工具**（PDFp86~87，图 7.8）：记录页错误顺序与原因；可据此让"先后执行的代码同页"，避免/推迟页错误。
5. **调用图分析 + 对象文件重排**（PDFp87）：从入口点追踪依赖，把对象文件按调用顺序排列填满页；[17] 仅重排函数就降 5% 启动成本。
6. **MAP_POPULATE**（PDFp87）：一次较贵 mmap 换多次页错误；粒度粗、未用页浪费、过早占用内存；系统忙时可丢弃该优化。
7. **POSIX_MADV_WILLNEED**（PDFp87）：细粒度预取提示；对"映射文件但只用部分"优势大。
8. **大页减少页错误**（PDFp87）：2MB 页比 4KB 少 511 次页错误/大页；物理连续限制 → hugetlbfs（详见 note/15）。

## 6. 通俗解释

Demand paging 像**"预约制取货"**：

> mmap 只是"预约了这批货"（登记地址），货并不在你手里。真正第一次去拿（访问）那页，
> 仓库才开工：查订单（页表）、找货/备货（找页/清零/读盘）、入账（改页表）、通知各相关部门（同步）。
> 每拿一页都有这套开销，所以"一页一个动作"会慢。

页错误分两种：

> minor：货其实已经在仓库里（首次匿名页/COW/内容已存在），只要"找出来"——便宜些但也要走流程。
> major：货要去隔壁城市调（读盘）——贵得多。

MAP_POPULATE 与 WILLNEED 的区别：

> MAP_POPULATE 像"下单时就要求把整批货全搬到门口"——一次搞定但浪费（可能用不到）。
> WILLNEED 像"先告诉我你近期要哪几件"——精确预取，适合只读大文件的一小部分。

## 7. 示例分析

### 7.1 图 7.8 解读

- 第一条指令 `3000000B50` 触发代码页 `3000000000` 入内存。
- 第 3320 周期就碰了数据页 `7FF000000`（call 指令压返回地址）。
- 含义：启动序列很短的时间内就触碰多个页 → 这些页按序重排到一起可减少页错误。

### 7.2 MAP_POPULATE 取舍

```cpp
// 全部页都要马上用：MAP_POPULATE 划算
mmap(ptr, len, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE|MAP_POPULATE, -1, 0);

// 大文件映射但只用到一部分：用 madvise(WILLNEED)
posix_madvise(addr, range, POSIX_MADV_WILLNEED);
```

- 若映射后只有一部分页会被用，MAP_POPULATE 浪费；WILLNEED 粒度更合适。

### 7.3 大页减少页错误

- 1GB 数据：4KB 页 → 262144 页错误；2MB 页 → 512 页错误。
- 图 7.9：1MB 工作集（单 2MB 页）无 DTLB miss → 快 57%。

## 8. 未优化代码

对应"启动时大量页错误"的程序（立即触碰整个大映射）。

```cpp
// bad.cpp: 顺序触碰大数组，逐页页错误
#include <vector>

int main() {
    constexpr int N = 1 << 26;
    std::vector<int> data(N, 0);
    for (int i = 0; i < N; ++i) data[i] = i;   // 每 4KB 页一次页错误
    return data[0] == 0;
}
```

## 9. 优化后代码

对应"预分配/预取"的程序。

```cpp
// good.cpp: MAP_POPULATE 预分配（全量使用场景）
#include <cstddef>
#include <sys/mman.h>

int main() {
    constexpr std::size_t BYTES = std::size_t(1) << 26;
    int* data = static_cast<int*>(mmap(nullptr, BYTES,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0));
    if (data != MAP_FAILED)
        for (std::size_t i = 0; i < BYTES / sizeof(int); ++i) data[i] = (int)i;
    return data[0] == 0;
}
```

> 完整页错误对比（malloc/首次/二次访问/MAP_POPULATE/madvise）见 src/21_page_fault。

## 10. 为什么会更快

| 角度 | 普通 mmap | MAP_POPULATE |
|---|---|---|
| 页错误次数 | N 页 N 次 | 1 次 mmap（预分配） |
| 页错误开销 | 每次进内核+同步 | 摊销到一次调用 |
| 内存占用时机 | 用才占 | 立刻占（若不用则浪费） |
| 适用 | 部分页使用 | 全部页即将使用 |

论文数据：图 7.9 大页 1MB 快 57%（单 2MB 页、无 DTLB miss）——页错误+TLB 双重收益。

## 11. 如何验证

```bash
./build/21_page_fault/page_fault
perf stat -e page-faults,minor-faults,major-faults ./build/21_page_fault/page_fault
\time ./build/21_page_fault/page_fault          # 显示 minor/major pagefaults
cat /proc/self/stat | awk '{print "minflt="$10" majflt="$12}'
```

## 12. 实验结果应该怎么看

- 对比"malloc 后不访问"（0 页错误）、"首次访问"（每页一次 minor）、"二次访问"（0 新增）。
- MAP_POPULATE：页错误几乎全在 mmap 期间一次发生。
- major vs minor：访问文件映射但文件不在 page cache 时才有 major；不要擅自清理系统 Page Cache。
- 大页实验须先确认系统支持（note/15），否则跳过。

## 13. 常见误区

- **误区 1：mmap 就分配了内存**。demand paging 下 mmap 只登记，首次访问才分配。
- **误区 2：minor 页错误免费**。它也要进内核、清页、改页表、同步，只是比 major 便宜。
- **误区 3：MAP_POPULATE 总是好**。粒度粗、未用页浪费、过早占内存；系统忙时可被丢弃。
- **误区 4：madvise 一定会预取**。它是提示，内核可忽略。
- **误区 5：可以随便清 Page Cache 测 major**。需 root 且影响系统，只能提示、不擅自执行。

## 14. 实践练习

1. 运行 src/21，对比不访问/首次/二次访问的页错误数与耗时。
2. 用 perf 与 `\time` 统计 minor/major 页错误。
3. 对比 mmap 普通版与 MAP_POPULATE 版的页错误分布。
4. 解释图 7.8 中"call 指令也触发数据页错误"的原因。
5. 讨论：为什么代码重排（让启动函数同页）能降低启动页错误。

## 15. 本章总结

- mmap 只改页表，页在首次访问时分配（demand paging）。
- 页错误处理昂贵；minor（内存中）比 major（读盘）便宜但都不免费。
- pagein 工具记录页错误顺序，指导代码重排。
- MAP_POPULATE 预分配（粗粒度）、POSIX_MADV_WILLNEED 细粒度预取。
- 大页（note/15）减少页数从而减少页错误与 TLB miss。

## 16. 对应代码

- src/21_page_fault/（页错误对比实验）
- src/22_memory_mapping/（mmap vs read）
- src/19_page_size/、src/20_huge_pages/（大页减少页错误）
