# hotspot_profiling

用 profiler 识别热点函数。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 97-102 页：

- **帕累托法则（80/20）**：约 20% 的代码消耗 80% 的资源；
- **插桩 profiler**：在函数入口/出口插入计数代码，精确但影响被测程序
  （本项目的 `scoped_timer` 是手工插桩的例子）；
- **采样 profiler**：按固定间隔（~10ms）采样调用栈，几乎不干扰程序，
  但函数必须出现在采样点上才被记录；
- **gprof**：混合两者（插桩 + 采样）。

本实验构造一个失衡程序：`compute_heavy` 承担 90%+ 时间，`helper_a/b` 很轻。

## 构建与运行

```bash
cmake --build build --target ch03_hotspot_profiling
./build/chapter03_measurement/ch03_hotspot_profiling
```

## perf（采样 profiler）

```bash
# 需要 root 或 perf_event_paranoid <= 1
sudo ./scripts/perf_stat.sh ./build/chapter03_measurement/ch03_hotspot_profiling
sudo ./scripts/perf_record.sh ./build/chapter03_measurement/ch03_hotspot_profiling
sudo perf report
```

## gprof（混合式，无需 root）

```bash
cd /tmp
g++ -std=c++17 -O0 -g -pg -o hotspot \
    src/chapter03_measurement/hotspot_profiling/example.cpp
./hotspot && gprof ./hotspot gmon.out
```

## 结果解释（本环境实测）

`-O0 -pg` 下 gprof 显示 `compute_heavy` 占约 **94%** 的时间，
`main→compute_heavy` 与 `main→helper_a/b` 调用关系清晰——符合 80/20 设计。

**重要**：`-O2` 编译时 gprof 会失真——`compute_heavy` 被内联进 `main`，
采样只看到 main。这正是书中提醒的"插桩 profiler 可能阻止编译器优化"
的实证。采样 profiler（perf）则不受此影响（对优化代码也准确）。

## 观察点

- Total vs Self 列（书中 p.101-102 的表）：Total 是该函数出现在调用栈的
  采样占比，Self 是它位于栈顶（自己执行）的占比；
- 采样 profiler 可能漏掉短暂/低频函数（书中 f4 例子）。
