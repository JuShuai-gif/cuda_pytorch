# 19 数据布局优化 AoS 与 SoA

> 对应 PDF：第 6.2.1 节 Optimizing Level 1 Data Cache Access 的数据结构部分（PDFp51~54）、图 6.2~6.5、表 6.2
> 本篇回答：结构体在内存里怎么摆才不浪费缓存行？为什么"字段对齐 + 紧凑"很重要？什么时候该把一个大结构体拆开（AoS→SoA）？未对齐访问代价有多大？

## 1. 本章要解决的问题

- 结构体成员顺序与对齐如何影响缓存行利用？
- pahole 怎么用，怎么通过重排成员消除空洞？
- 结构体对象如何按缓存行对齐（posix_memalign / __attribute__((aligned))）？
- 未对齐访问的代价（图 6.4：顺序 L2 内慢约 300%）。
- 什么时候把大结构体拆成 AoS/SoA？关键字段前置？

## 2. 前置知识

- note/05、06：缓存行、局部性。
- note/18：矩阵乘法的缓存优化。
- C/C++ 结构体对齐规则（alignof、填充）。
- pahole 工具（可选安装）。

## 3. 核心概念

- **Alignment（对齐）**：对象地址按类型要求对齐；对齐不足导致跨缓存行访问。
- **Structure Padding（结构体填充）**：成员按对齐规则留下的空洞。
- **Cache Line Utilization（缓存行利用率）**：一条缓存行中有多少字节被真正用到。
- **Critical Word（关键字段）**：程序最先等待的字段，应放结构体开头。
- **AoS（Array of Structures，结构体数组）**：一个结构体包含所有字段，数组连续。
- **SoA（Structure of Arrays，数组结构体）**：每个字段一个数组，按字段收集。
- **posix_memalign**：按指定对齐分配内存。
- **`__attribute__((aligned(n)))`**：gcc 变量/类型对齐属性。
- **Conflict Miss（冲突未命中）**：多个对象落入同一缓存集合导致的 miss（L1d 用虚拟地址索引，程序员可影响）。

## 4. 硬件工作流程

### 4.1 结构体如何占用缓存行

```cpp
struct foo {          // 64 位编译
    int a;            // offset 0
    long fill[7];     // offset 8..56（int 后补 4 字节空洞）
    int b;            // offset 64..68
};                    // size 72 → 2 条缓存行
```

pahole 输出（图 6.3）会明确标注：`XXX 4 bytes hole`、`cacheline 1 boundary`、`size: 72, cachelines: 2`。

优化：把 `b` 移到 `a` 后面填掉 4 字节空洞 → size 64 → 正好 1 条缓存行。

### 4.2 对象级对齐

```text
动态分配：posix_memalign(&p, 64, size);
编译器分配（全局/栈）：__attribute__((aligned(64))) 变量 或 类型属性
```

- 结构体默认对齐 = 成员最大对齐（常 < 缓存行），所以光排成员还不够，对象本身要对齐。
- 类型属性 `__attribute__((aligned(64)))` 让所有该类型对象（含数组）对齐。
- 动态对象用 `posix_memalign(p, alignof(...), size)`。
- 代价：对齐分配可能造成碎片化/更高内存占用。

### 4.3 AoS vs SoA（PDFp53~54）

```cpp
// AoS：order 数组，字段聚在一起
struct order {
    double price;
    bool paid;
    const char* buyer[5];
    long buyer_id;
};

// 若频繁任务只累加 price：buyer/buyer_id 白占缓存 → 慢最多 5 倍
// SoA：拆开，price/paid 一个数组，buyer/buyer_id 另一个
```

- AoS：概念上聚在一起、写代码直观，但"只访问部分字段"时浪费缓存行。
- SoA：把常用字段收集到连续数组，缓存行利用率高、SIMD 友好；代码更复杂。

### 4.4 未对齐访问（图 6.4，PDFp53）

- 顺序访问 + 工作集在 L2 内：未对齐慢约 300%——增量操作常触碰两条缓存行，L1↔L2 连接拥塞。
- 大工作集：仍慢 20%~30%（对齐访问本身已很慢，但未对齐额外代价可观）。
- 结论：即使 CPU 支持未对齐访问，也不代表它们"和对齐一样快"。

## 5. PDF 核心观点

> 来源：PDF 第 51~54 页；对应章节 6.2.1（结构体部分）、图 6.2~6.5。以下为概括。

1. **缓存行利用率**（PDFp51）：矩阵乘法优化的本质是"所有缓存行字节都被用上，在逐出前"。
2. **跨缓存行的惩罚**（PDFp51，图 6.2）：两个元素一个在行首一个在行尾 → 顺序访问 L2 内慢约 17%、主存级约 27%；随机访问 L2 内慢 25%~35%。
3. **用 pahole 查看布局**（PDFp51~52，图 6.3）：结构体大小 72、2 条缓存行、4 字节空洞；把 b 移进空洞可压缩到 1 条缓存行。
4. **关键字段前置**（PDFp52）：两条规则——(a) 最可能是 critical word 的成员放开头；(b) 按成员定义顺序访问。
5. **对象要对齐到缓存行**（PDFp52）：否则重排成员白费；用 posix_memalign 或 `__attribute__((aligned))`。
6. **对齐有成本**（PDFp53）：对齐要求可能导致碎片化/内存占用上升；数组类型对齐需类型属性（否则只有首元素对齐）。
7. **未对齐访问很贵**（PDFp53，图 6.4）：顺序访问 L2 内慢约 300%（触两条缓存行），大工作集仍慢 20%~30%。
8. **栈对齐由 ABI 保证**（PDFp53）：编译器按 ABI 让调用方对齐；VLA/alloca 时栈帧大小运行时才知，需主动对齐控制。
9. **AoS 拆分**（PDFp53）：`order` 结构体若频繁只累加 `price`，`buyer/buyer_id` 白占缓存 → 最多慢 5 倍；拆成两部分可避免。
10. **冲突未命中可控**（PDFp54）：L1d 用虚拟地址索引，把一起使用的变量放一起可降低落入同一集合的概率（图 6.5）。

## 6. 通俗解释

数据布局优化的核心：**让"一起用的数据"睡在同一张床（缓存行）上**。

> 缓存行是 64 人一间的大床房。你想让"经常一起干活"的成员住一间房，
> 这样搬一次房（load 一条缓存行）就都到齐了。可如果结构体里有大块不常用的字段
> （比如每个订单的买家名单），它们占着房间让常用的 price 都住不进来——一次搬房只用到 1/8。

AoS vs SoA 就像：

> AoS：每个人的所有资料装一个信封，一摞信封摆桌上。你只查每个人"工资"这一项，
> 就得拆开每个信封、翻半天（整条缓存行只有 1/8 用上）。
> SoA：把所有人的"工资"单独订成一摞，另一摞放"名字"。查工资时只翻那一摞，页页用满。

对齐为什么重要？

> 如果某人（对象）站的位置正好卡在两间房之间（未对齐、跨缓存行），他做事要先开两间房，
> 邻居也被占。图 6.4 显示这能慢 3 倍。

## 7. 示例分析

### 7.1 用 pahole 压缩结构体

```cpp
// 原始：size 72，2 条缓存行，4 字节空洞
struct foo { int a; long fill[7]; int b; };

// 重排：b 填入 a 后空洞
struct foo { int a; int b; long fill[7]; };   // size 64，1 条缓存行
```

- `pahole --reorganize` 可自动做这类优化（含位域、合并空洞）。

### 7.2 AoS 拆分示例

```cpp
// 频繁任务：累加所有订单 price
struct Order { double price; bool paid; const char* buyer[5]; long buyer_id; };
// AoS 数组：每 64B 行只有 price(8)+paid(1) 有用 → 利用率 ~14%
// SoA：double* price; bool* paid; → 顺序数组，利用率 ~100%，且 SIMD 友好
```

### 7.3 未对齐代价

- 顺序自增 + 工作集 L2 内：未对齐慢约 300%（每次触碰两条缓存行，L1↔L2 拥塞）。
- 这就是为什么多媒体/SSE 指令要求 16 字节对齐，未对齐变体更慢。

## 8. 未优化代码

对应 AoS + 只访问部分字段的程序（粒子数据）。

```cpp
// bad.cpp: AoS，只访问 x/y 但整行加载全部字段
#include <vector>

struct Particle {
    float x, y, z;
    float velocity;
    float mass;
};

int main() {
    constexpr int N = 1 << 20;
    std::vector<Particle> p(N);
    for (auto &e : p) e.x = e.y = e.z = e.velocity = e.mass = 1.0f;

    float sum = 0;
    for (auto &e : p) sum += e.x + e.y;   // 只用到 8B/行，行里还有 12B 无用
    return sum == 0.0f;
}
```

## 9. 优化后代码

对应 SoA + 只访问部分字段的程序。

```cpp
// good.cpp: SoA，字段收集为连续数组
#include <vector>

int main() {
    constexpr int N = 1 << 20;
    std::vector<float> x(N), y(N), z(N), velocity(N), mass(N);
    for (int i = 0; i < N; ++i) x[i] = y[i] = z[i] = velocity[i] = mass[i] = 1.0f;

    float sum = 0;
    for (int i = 0; i < N; ++i) sum += x[i] + y[i];  // 连续数组，SIMD 友好
    return sum == 0.0f;
}
```

> 完整 AoS/SoA 对比（含 SIMD 与内存带宽测量）见 src/11_aos_soa。

## 10. 为什么会更快

| 角度 | AoS 部分字段 | SoA |
|---|---|---|
| 缓存行利用率 | 只用 ~14%（8/56B） | ~100% |
| L1d 命中 | 低 | 高 |
| 预取 | 受无用字段拖累 | 纯顺序 |
| SIMD 友好 | 不友好（字段间隔） | 连续向量 |
| 内存带宽 | 浪费 | 全利用 |

论文数据：`order` 场景最多慢 5 倍（图 3.11 思想）。这是"只访问部分字段"时 AoS 的典型代价。

## 11. 如何验证

```bash
./build/11_aos_soa/aos_soa            # AoS vs SoA
./scripts/perf_stat.sh ./build/11_aos_soa/aos_soa
pahole ./build/11_aos_soa/aos_soa     # 查看结构体布局（若已安装 pahole）
```

查看对齐/布局：

```bash
echo 'int main(){struct foo{int a;long f[7];int b;};return sizeof(struct foo);}' \
  | gcc -x c - -o /tmp/sz && /tmp/sz; echo $?   # 打印 sizeof
getconf LEVEL1_DCACHE_LINESIZE
```

## 12. 实验结果应该怎么看

- AoS vs SoA 只访问部分字段：SoA 的耗时与 cache-misses 显著更低。
- 若访问全部字段：两者差异可能很小甚至 AoS 略好（少一次间接）——所以优化要针对真实访问模式。
- 未对齐对比（若实验含）：顺序访问场景未对齐应明显慢，印证图 6.4。
- 用 perf 看 `cache-references/cache-misses`，解释利用率差异。

## 13. 常见误区

- **误区 1：AoS 永远比 SoA 慢**。取决于访问模式：访问全部字段时 AoS 可能更简单直接；只访问部分字段才该拆。
- **误区 2：对齐=浪费内存**。对齐可能多占内存（碎片化），但缓存利用率与速度收益通常更大；需权衡。
- **误区 3：CPU 支持未对齐访问=未对齐免费**。论文图 6.4：慢 300%（顺序 L2 内）。
- **误区 4：结构体重排只是整洁问题**。它决定哪些字段睡同一缓存行，直接决定 critical word 的等待。
- **误区 5：pahole 能自动解决一切**。工具提供帮助（含 --reorganize），但"哪些字段一起用"仍需程序员判断。

## 14. 实践练习

1. 用 pahole（或手动）分析 `struct order`，把 `b` 移入空洞，对比 sizeof。
2. 运行 src/11：分别只访问部分字段/全部字段，对比 AoS 与 SoA。
3. 构造一个未对齐访问基准，验证图 6.4 的 300% 量级（顺序访问、L2 内工作集）。
4. 讨论：什么场景下 AoS 更合适？（如对象常整体复制/传递、访问全部字段）
5. 用 `__attribute__((aligned(64)))` 与 posix_memalign 各做一次对齐分配，说明区别。

## 15. 本章总结

- 缓存行利用率决定数据布局优劣；"一起用的字段放同一缓存行"是目标。
- pahole 揭示结构体空洞与缓存行占用，重排成员可压缩。
- 对象要对齐到缓存行；posix_memalign 与 aligned 属性提供手段。
- 未对齐访问昂贵（顺序 L2 内慢约 300%）。
- 关键字段前置、按定义顺序访问。
- AoS→SoA：只访问部分字段时收益显著（可达 5 倍），访问全部字段时需重新评估。

## 16. 对应代码

- src/11_aos_soa/（AoS vs SoA 对比）
- src/12_pointer_chasing/（大结构体/NPAD 效应衔接）
- src/09_matrix_traversal/、src/10_cache_blocking/（行主序数据布局）
