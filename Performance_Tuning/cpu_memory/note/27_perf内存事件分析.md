# 27 perf 内存事件分析

> 对应 PDF：第 7.1 节 Memory Operation Profiling（PDFp78~80）、附录 B Some OProfile Tips（PDFp102~104）、图 7.1~7.3
> 本篇回答：如何在现代 Linux 上用 perf 观测内存行为？哪些事件可用？CPI、cache-miss、TLB miss 怎么读？OProfile（历史工具）与 perf 的对应关系？

## 1. 本章要解决的问题

- 性能计数器的正确使用（比例而非绝对值）。
- CPI 的三个台阶与缓存层级的关系。
- cache-misses、dTLB 等事件在 perf 里怎么统计。
- 论文时代 OProfile 与 modern perf 的对应。
- 如何用 opannotate/perf annotate 定位热点行。

## 2. 前置知识

- note/26：OProfile、性能计数器、CPI 概念。
- 全书缓存/内存概念。
- Linux 命令行基础。

## 3. 核心概念

- **CPI（Cycles Per Instruction）**：`CPU_CLK_UNHALTED / INST_RETIRED`。
- **perf stat**：一次性统计事件。
- **perf record / perf report / perf annotate**：采样记录、报告、源码行标注。
- **cache-references / cache-misses**：perf 的通用缓存事件名。
- **L1-dcache-loads / L1-dcache-load-misses / dTLB-loads / dTLB-load-misses**：perf 的内存相关事件。
- **Hardware Prefetch 事件**：如 Core 2 的 SSE_HIT_PRE/SSE_PRE_MISS/LOAD_PRE_EXEC（评估软件预取）。
- **Demand Miss vs Total Miss**：需求未命中 vs 含硬件预取的未命中（论文区分）。

## 4. 硬件工作流程

### 4.1 CPI 计算与解读（图 7.1，PDFp78）

```text
CPI = 周期数 / 已退休指令数
小工作集（≤L1d）→ ≤1.0
L1d 不足 → ~3.0（L2 惩罚被平均到所有指令）
L2 不足 → >20（主存惩罚）
```

### 4.2 缓存事件解读（图 7.2/7.3，PDFp79）

```text
随机 Follow：
  L1d miss：L1d(32k) 后上升，预取把 miss 压到 1%（≤64k 工作集）
  L2 miss：L2(2^21) 后上升；L2 demand miss 非零（预取不完美）
  DTLB miss：在 L2 miss 之前就开始显著

顺序 Follow：
  L2 demand miss ≈0（硬件预取完美）
  L1d 与 L2 miss 率相同（所有 L1d miss 都由 L2 满足）
  DTLB ≈0
```

### 4.3 perf 命令

```bash
perf stat -e cycles,instructions,branches,branch-misses,
          cache-references,cache-misses,page-faults,
          minor-faults,major-faults,context-switches,cpu-migrations \
          ./program

perf stat -e L1-dcache-loads,L1-dcache-load-misses,
          dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses \
          ./program

perf record -g ./program && perf report
perf annotate
```

- 事件名平台相关：Intel/AMD 名称可能不同，脚本需容错（见 scripts/perf_stat.sh）。
- 某些事件（如 LLC-loads）在部分 CPU 上不可用，需检查并跳过。

## 5. PDF 核心观点

> 来源：PDF 第 78~80、102~104 页；对应章节 7.1、B、图 7.1~7.3。以下为概括。

1. **计数器要用比例**（PDFp78）：绝对值无意义；相除得到可比指标（如 CPI）。
2. **事件因 CPU 而异**（PDFp78）：OProfile 接口简单但需自己查手册理解事件。
3. **CPI 台阶**（PDFp78，图 7.1）：L1d 内 ≤1、L1d 不足 ~3、L2 不足 >20。
4. **miss 率要把非访存指令从 INST_RETIRED 减去**（PDFp78~79）：否则实际 miss 率更高。
5. **L1d miss 在包含缓存下隐含 L2 miss**（PDFp79）：Intel 包含缓存，L2 miss 必然 L1d miss。
6. **随机 vs 顺序的 L2 demand miss**（PDFp79，图 7.2/7.3）：随机非零（预取不完美）、顺序≈0（预取完美）。
7. **DTLB 先于 L2 miss 变高**（PDFp79）：随机访问 TLB 压力在 L2 压力之前出现。
8. **软件预取有效性**（PDFp79）：useful/late prefetch 事件评估；论文例 5.5% 有用、48% 未及时。
9. **opannotate 定位**（PDFp79~80）：按指令/源码行标注事件；随机采样有遗漏风险。
10. **附录 B OProfile 技巧**（PDFp102~104）：现代工具参考价值有限，但"先看 miss 大户、再看有用预取"的方法论仍适用。

## 6. 通俗解释

perf 就像**给程序装的仪表盘**：

> `perf stat` 给你一份"仪表读数"：周期数、指令数、缓存 miss、TLB miss、页错误……
> `perf record` 像行车记录仪，`perf report`/`perf annotate` 告诉你"哪个路口最堵"（哪行代码 miss 最多）。

CPI 是"平均每走一步花多少拍"：

> 数据都在桌面（L1d）时，一步一拍（≤1）；要到文件柜（L2）取，两三拍（~3）；
> 要下楼去仓库（主存）取，二十多拍（>20）。指标一目了然。

怎么判断缓存/预取好不好？

> 随机访问：硬件预取"猜不中"，L2 demand miss 非零；顺序访问：预取完美，demand miss 接近零。
> DTLB miss 就像"频繁翻通讯录"——随机访问时它先于缓存问题冒头。

## 7. 示例分析

### 7.1 CPI 读法

- `perf stat` 输出 `# 3.5 cycles per instruction`：说明平均每指令 3.5 周期 → 缓存 miss 拖累明显。
- 结合工作集：若小工作集 CPI 也高，则问题不在缓存而在代码依赖/分支。

### 7.2 缓存 miss 事件

- `cache-misses`（通用名）在 Intel 上通常映射到 LLC miss。
- `L1-dcache-load-misses` 看 L1 命中。
- 论文强调：Intel 包含缓存下 L2 miss ⇒ L1d miss，所以 L1d 与 L2 曲线形状相关。

### 7.3 预取事件

- 若 CPU 暴露 `useful prefetch`/`prefetch misses` 类事件，可评估软件预取有效性（论文 SSE_HIT_PRE/SSE_PRE_MISS）。
- 无法暴露时：对比"有无软件预取"两版代码的 cache-misses 与耗时。

## 8. 未优化代码

对应"缓存 miss 高"的随机访问程序。

```cpp
// bad.cpp: 随机访问大数组
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

对应"缓存 miss 低"的顺序访问程序。

```cpp
// good.cpp: 顺序访问
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

## 10. 为什么会更快

| 角度 | 随机 | 顺序 |
|---|---|---|
| CPI | 高（主存惩罚） | 低（缓存命中） |
| cache-misses | 大量 | 少 |
| dTLB-load-misses | 高 | 低 |
| L1/L2/LLC 命中 | 低 | 高 |

perf 数据会量化这些差异；论文图 7.1 给了 2007 年的 CPI 台阶（≤1/~3/>20）。

## 11. 如何验证

```bash
./scripts/system_info.sh
./scripts/perf_stat.sh ./build/02_sequential_random_access/sequential_random_access
./scripts/perf_stat.sh ./build/05_cache_capacity/cache_capacity
./scripts/perf_record.sh ./build/09_matrix_traversal/matrix_traversal

# 手动指定事件
perf stat -e cycles,instructions,page-faults \
         ./build/02_sequential_random_access/sequential_random_access
```

## 12. 实验结果应该怎么看

- 先看 CPI 量级：≤1 说明计算受限于指令依赖；3+ 说明缓存 miss 主导。
- 对照工作集变化：cache-misses 在超出某级缓存时跳升，台阶位置对应缓存大小。
- 事件不可用时：脚本应跳过并说明，不硬编码假设（CPU 事件平台相关）。
- 用比例解读，不要用绝对值。

## 13. 常见误区

- **误区 1：所有 CPU 事件名一样**。Intel/AMD 差异大；perf 通用名在部分平台不可用。
- **误区 2：cache-misses 一定是 L1 miss**。常映射到 LLC miss；需确认。
- **误区 3：perf 需要 root 才能用**。基础事件在 paranoid 值较低时可用户态运行；root 可获得更全事件。
- **误区 4：OProfile 现在还能直接用**。论文时代工具；现代用 perf（OProfile 已基本被取代）。
- **误区 5：只跑一次就下结论**。本项目所有实验多轮统计，perf 也应多轮观察。

## 14. 实践练习

1. 运行 `perf stat` 对比顺序/随机访问的 CPI 与 cache-misses。
2. 用 `perf record -g` + `perf report` 定位矩阵乘法的热点行。
3. 若 CPU 暴露 L1/TLB 事件，记录工作集变化时 miss 台阶。
4. 用 perf annotate 查看热循环的指令级 miss。
5. 阅读论文图 7.1~7.3，把"CPI 三台阶"与你的 perf 数据对照。

## 15. 本章总结

- perf 是现代 Linux 的性能计数器接口；OProfile 是历史工具。
- CPI = cycles/instructions，台阶对应 L1d/L2/主存。
- cache-misses/dTLB 事件按平台变化，脚本要容错。
- 随机访问 L2 demand miss 非零、DTLB 提前恶化；顺序访问近零。
- 预取有效性可用专用事件或对比实验评估。
- 用比例解读、多轮验证、标注平台。

## 16. 对应代码

- scripts/perf_stat.sh、scripts/perf_record.sh（事件统计与记录）
- src/02_sequential_random_access/、src/05_cache_capacity/、src/18_tlb_capacity/（分析对象）
