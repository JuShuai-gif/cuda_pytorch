# 25 NUMA 编程实践

> 对应 PDF：第 6.5 节 NUMA Programming（PDFp72~77）、第 6.5.1~6.5.8 节、附录 D libNUMA Introduction（PDFp108~110）、图 6.15
> 本篇回答：程序员在 NUMA 上要做什么？内存策略（policy）怎么设？mbind/set_mempolicy 怎么用？只读数据怎么复制、可写数据怎么处理？libnuma/libNUMA 接口？

## 1. 本章要解决的问题

- 内存策略分层（系统默认 → 任务 → VMA/共享内存）。
- 四种策略：BIND / PREFERRED / INTERLEAVE / DEFAULT。
- set_mempolicy 与 mbind（含 MPOL_MF_MOVE/STRICT）的用法。
- get_mempolicy 查询页所在节点。
- 显式 NUMA 优化：只读数据复制（replication）、可写数据的分区累加。
- libnuma（numactl 包）与作者建议的 libNUMA 接口。
- 附录 D：libNUMA 的 CPU/节点集合与层级接口示例。

## 2. 前置知识

- note/16、17：NUMA 概念、Linux 支持、numa_maps。
- note/15：线程亲和性基础（pthread_setaffinity_np 等）。
- 进程/线程、mmap、共享内存概念。

## 3. 核心概念

- **Memory Policy（内存策略）**：决定新分配内存落在哪个节点的规则。
- **Policy Hierarchy（策略分层）**：VMA 策略 > 任务策略 > 系统默认。
- **MPOL_BIND / MPOL_PREFERRED / MPOL_INTERLEAVE / MPOL_DEFAULT**：四种分配模式。
- **set_mempolicy**：设置当前线程的任务策略。
- **mbind**：为地址区间设置 VMA 策略（可带 MOVE/STRICT 标志）。
- **MPOL_MF_MOVE / MOVEALL / STRICT**：迁移已有页 / 迁移所有页 / 严格校验。
- **get_mempolicy**：查询地址策略或页所在节点。
- **First-Touch（首次触碰）**：页在实际第一次读写时分配，分配节点由当时策略决定。
- **Replication（复制）**：只读数据每节点一份副本，避免跨节点访问。
- **libnuma（numactl 包）**：现有 NUMA 库，系统调用薄封装。
- **libNUMA**：论文作者提出的新库，封装 CPU/节点集合与层级信息（未正式发布）。

## 4. 硬件工作流程

### 4.1 策略分层（图 6.15，PDFp72~73）

```text
地址 A 的分配节点确定顺序：
  ① 有 VMA 策略（mbind 设置）？ → 用它
  ② 否则有任务策略（set_mempolicy）？ → 用它
  ③ 否则系统默认 → 本地分配（请求线程所在节点的本地内存）
```

- 默认：内存本地分配；任务/VMA 策略默认不存在。
- 多线程进程的"本地节点" = 首先运行进程的节点（home node）。

### 4.2 四种策略（PDFp73）

| 策略 | 行为 |
|---|---|
| MPOL_BIND | 只从指定节点分配，否则失败 |
| MPOL_PREFERRED | 优先从指定节点，失败才考虑其他节点 |
| MPOL_INTERLEAVE | 按偏移/计数器在指定节点间轮流分配 |
| MPOL_DEFAULT | 用区域默认策略 |

### 4.3 接口使用

```cpp
#include <numaif.h>
long set_mempolicy(int mode, unsigned long *nodemask, unsigned long maxnode);
long mbind(void *start, unsigned long len, int mode,
           unsigned long *nodemask, unsigned long maxnode, unsigned flags);
long get_mempolicy(int *policy, const unsigned long *nmask,
                   unsigned long maxnode, void *addr, int flags);
```

- set_mempolicy：只影响当前线程、只影响**未来**分配（已有页不迁移）。
- mbind：地址须页对齐，len 上取整到页；flags=0 只设未来；MPOL_MF_MOVE 尝试迁移本进程独享页；MPOL_MF_MOVEALL 迁移所有页（特权）；MPOL_MF_STRICT 严格（不满足则失败）。
- get_mempolicy：flags=0 查询策略；MPOL_F_NODE 查询 INTERLEAVE 下一个分配节点；MPOL_F_ADDR 查询 addr 所在页的节点。

### 4.4 First-Touch 与分配时机

```text
mmap 只保留地址区间（不分配内存）
第一次读/写该页 → 页错误 → 按当前策略分配 → 页落某节点
若策略在访问不同页之间变化，或允许多节点 → 同一区间可能散落多节点
```

### 4.5 显式 NUMA 优化（PDFp76~77）

```text
只读数据 → 复制（replication）：
  每节点一份副本，线程用本地副本（libNUMA: NUMA_memnode_self_current_idx 选节点）

可写数据 → 分区累加：
  各节点各自累加区域，最后合并（要求累加无状态/顺序无关）
  或（访问量很大时）用页迁移把页搬到本地节点
```

### 4.6 libNUMA（附录 D，PDFp108~110）

```text
CPU/节点集合宏：MEMNODE_*（类似 CPU_*）
查询：NUMA_cpu_system_count、NUMA_cpu_self_current_idx、
      NUMA_memnode_self_current_idx、NUMA_cpu_to_memnode、NUMA_memnode_to_cpu、
      NUMA_mem_get_node_idx / NUMA_mem_get_node_mask
层级：NUMA_cpu_level_mask(..., level)：
      level=1 → 超线程兄弟；level=2 → 同核心其他核心；更高级 → 缓存/节点层级
```

- 附录 D 示例：找超线程兄弟（level 1）、找同包其他核心（level 2 XOR level 1），用于调度辅助线程/绑定。
- 注意事项：返回的当前 CPU/节点信息可能已过期，只能当提示；CPU hot-plug 会改变。

## 5. PDF 核心观点

> 来源：PDF 第 72~77、108~110 页；对应章节 6.5、D。以下为概括。

1. **NUMA 改变了页的"平等性"**（PDFp72）：均匀内存下只优化页错误；NUMA 下要优化页的本地性。
2. **NUMA 不可避免**（PDFp72）：Intel CSI / AMD Opteron 都用；家用单处理器无 NUMA，但程序员不能忽略（缓存层级也是"非均匀"的）。
3. **缓存也是 NUMA**（PDFp72）：共享缓存的核协作更快（Core 2 四核有两条独立 L2）。
4. **策略分层**（PDFp72~73）：VMA > 任务 > 系统默认；默认本地分配；"本地"= 进程首次运行节点。
5. **四种策略**（PDFp73）：BIND / PREFERRED / INTERLEAVE / DEFAULT。
6. **set_mempolicy 只影响未来**（PDFp73）：已有页不迁移；mmap 不分配内存，首次访问才分配。
7. **mbind 与迁移**（PDFp74）：flags=0 设未来；MPOL_MF_MOVE 迁移本进程独享页；MOVEALL 特权且影响其他进程；STRICT 严格校验。
8. **swap 丢失节点信息**（PDFp73~74）：换出页再换入，节点可能变化 → 节点关联是"提示"不是绝对事实，需要准确时用 get_mempolicy。
9. **只读数据复制**（PDFp76）：每节点一份副本，无跨节点访问；writable 不能简单复制。
10. **可写数据分区累加**（PDFp77）：各节点各自累加、最后合并（需无状态）；访问量大时可页迁移。
11. **利用全部带宽**（PDFp77）：缓存失效时远程访问并不慢（图 5.4）——可把不重读的数据写到其他节点的内存，本地/远程带宽并行；需确认缓存失效且远端节点空闲。
12. **libnuma 不充分**（PDFp72）：只是系统调用封装，无架构信息；作者提出 libNUMA（附录 D），未正式发布，依赖 /sys。
13. **信息会过期**（PDFp75、109）：当前 CPU/节点只是提示；CPU hot-plug、负载均衡都会改变。

## 6. 通俗解释

NUMA 编程就像**在多仓连锁店里调度"货往哪放"**：

> 默认规矩：谁在哪个店干活，货就放哪个店（本地分配）。
> 但可以定制：只放指定店（BIND）、优先放指定店（PREFERRED）、轮流放几家店（INTERLEAVE）、恢复默认（DEFAULT）。
> 改规矩只影响"新进的货"（未来分配），已摆好的货不会自己动（除非你用迁移）。

只读数据怎么共享？

> 一本大家都要查的书（只读数据），每个店放一本（复制/复制）——大家查自己店里的，不用跑隔壁店。
> 可写数据不能这样——两家店各记各的账最后对不上。所以要么各记各的账最后合并（分区累加），
> 要么把账本搬到干活的那家店（页迁移）。

libNUMA 是"店员通讯录"：

> 告诉你哪个店员（CPU）和哪个仓库（内存节点）是一家、谁是同一个工位的（超线程兄弟）、
> 谁在同一个车间（核心/缓存层级）——方便你把线程绑对地方。

## 7. 示例分析

### 7.1 内存策略分层示例

```cpp
// 系统默认：本地分配
set_mempolicy(MPOL_BIND, nodemask, maxnode);   // 任务策略：只从 node0..3 中选
mmap(...); mbind(p, len, MPOL_INTERLEAVE, ...); // VMA 策略：该区间轮流分配
```

- 分配 `p` 的页时：VMA 策略（INTERLEAVE）优先 → 页轮流落在指定节点。
- 若某地址无 VMA 策略 → 用任务策略（BIND）；都没有 → 系统默认本地。

### 7.2 只读复制（论文代码 6.5.7）

```cpp
void *local_data(void) {
    static void *data[NNODES];
    int node = NUMA_memnode_self_current_idx();   // 当前节点
    if (node == -1) node = 0;
    if (data[node] == NULL) data[node] = allocate_data();  // 每节点一份
    return data[node];
}
void worker(void) {
    void *data = local_data();   // 用本地副本，无跨节点访问
    for (...) compute using data;
}
```

### 7.3 可写数据的处理

- 分区累加：各节点各自 `sum_local += ...`，结束后合并 → 无跨节点写。
- 页迁移：`mbind(addr, len, MPOL_BIND, node, maxnode, MPOL_MF_MOVE)` 把页搬到本地节点。
- 前提：累加无状态（顺序无关）才可分区；迁移是拷贝，成本要摊薄。

## 8. 未优化代码

对应"忽略 NUMA"的多线程程序（共享一个大数组、条带分配、远程访问多）。

```cpp
// bad.cpp: 共享大数组，所有线程访问同一内存区域
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

对应"NUMA 感知"的程序：每线程访问本地分配的区域（first-touch + 绑定）。

```cpp
// good.cpp: 每线程触碰自己的分区（first-touch → 本地节点），配合 numactl 绑定
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
            for (int i = lo; i < hi; ++i) data[i] = 1;   // first-touch
        });
    for (auto &th : pool) th.join();
    return data[0] == 0;
}
```

外层绑定（多节点机器）：`numactl --cpunodebind=0 --membind=0 ./build/25_numa_first_touch/numa_first_touch`。

> first-touch 的机制验证见 src/25；本地/远程对比见 src/24；数据复制见 src/26。

## 10. 为什么会更快

| 角度 | 忽略 NUMA | first-touch + 绑定 |
|---|---|---|
| 页落在哪个节点 | 条带/随机 | 本地（触碰线程所在节点） |
| 远程访问 | 多 | 少 |
| 平均访存延迟 | 混入 hop | 0 hop |
| 带宽 | 受互连限制 | 本地满带宽 |

论文数据：远程读慢 9%~20%（1 hop）、30%（2 hop）（图 5.3/5.4），需实测确认。

## 11. 如何验证

```bash
numactl --hardware
./scripts/numa_test.sh
./build/24_numa_local_remote/numa_local_remote
./build/25_numa_first_touch/numa_first_touch
./build/26_numa_replication/numa_replication
cat /proc/self/numa_maps
```

libnuma 检测（CMake ENABLE_NUMA_EXAMPLES）：

```bash
ldconfig -p | grep libnuma     # 存在则可用
```

## 12. 实验结果应该怎么看

- 单节点机器：NUMA 实验应提示跳过，不编造远程数据。
- 多节点：对比绑定前后运行时间、numa_maps 页分布、访问延迟。
- first-touch：先由哪个节点线程触碰，页就落哪；绑定 + 触碰顺序决定本地性。
- replication：只读数据每节点一份后，跨节点访问显著减少。

## 13. 常见误区

- **误区 1：NUMA 只影响服务器**。缓存层级共享问题在桌面多核同样存在（论文原意：缓存也是"非均匀"的）。
- **误区 2：set_mempolicy 会迁移已有内存**。只影响未来分配；迁移要用 mbind+MOVE。
- **误区 3：内存策略全局生效**。分层：VMA > 任务 > 默认；set_mempolicy 只影响当前线程。
- **误区 4：复制适合可写数据**。writable 不能简单复制，需分区累加或迁移。
- **误区 5：libNUMA 是可用的生产库**。论文时代它未发布、依赖 /sys；现代用 libnuma/numactl 与内核接口。
- **误区 6：当前节点信息可靠**。调度器可能迁移线程，当前 CPU/节点只能当提示。

## 14. 实践练习

1. 用 numactl 在多节点机器上跑 src/24/25，记录绑定前后的差异。
2. 用 mbind 给一块内存设置 INTERLEAVE，用 get_mempolicy 验证页分布。
3. 实现论文 6.5.7 的 `local_data` 复制模式（src/26），对比跨节点访问。
4. 用 /proc/self/numa_maps 观察 first-touch 前后页分布变化。
5. 讨论：为什么页迁移是"拷贝操作"，其成本何时能被摊薄。

## 15. 本章总结

- 策略分层：VMA > 任务 > 系统默认（本地分配）。
- 四种策略：BIND / PREFERRED / INTERLEAVE / DEFAULT。
- set_mempolicy/mbind/get_mempolicy 控制与查询；mbind 可迁移（MOVE/STRICT）。
- 页在首次访问时分配（first-touch），策略变化会让区间散落多节点。
- 只读数据复制、可写数据分区累加或页迁移。
- libnuma 是薄封装；libNUMA 提供完整 CPU/节点/层级查询（论文时代未发布）。
- 节点/CPU 信息会过期，只能当提示。

## 16. 对应代码

- src/24_numa_local_remote/（本地/远程访问）
- src/25_numa_first_touch/（first-touch）
- src/26_numa_replication/（只读数据复制）
- scripts/numa_test.sh（检测与跳过）
