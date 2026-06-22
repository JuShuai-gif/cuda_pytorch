# 09 · 内存规划：生命周期分析 + 区间图着色复用

> 对应代码：`tools/edge-memplan/edge-memplan.cpp`
> 验证：`ninja -C build check-edge`（memplan 测试已通过，节省 50% 峰值内存）

---

## 1. 中文原理讲解

内存规划解决的问题：**如何让生命周期不重叠的张量共享同一块内存**，从而把峰值内存降到最低。
这与寄存器分配是同一个数学问题——**区间图着色**：每个张量是一个节点，生命周期重叠的两个张量之间
连边，"颜色"对应内存槽，目标是用最少的总字节覆盖所有节点。

`edge-memplan` 的算法（与 TFLite `GreedyMemoryPlanner` / XLA buffer assignment 同思路）：
1. **生命周期分析**：按程序顺序给每个 Op 编号；每个张量值的生命周期 `[birth, death]` =
   `[定义点, 最后一次被使用的点]`；被 `return` 使用的值活到结尾。
2. **大小计算**：`numElements * 元素字节宽`。
3. **贪心 by-size first-fit 分配**：按张量大小降序处理；放置每个张量时，只与**生命周期重叠**且已分配
   的张量做空间避让，找最低可放下的对齐偏移。
4. **报告**：朴素峰值（所有张量大小之和，无复用）vs 规划峰值（max offset+size），与节省比例。

验证：4 个 100×100×f32（各 40000 B）链式 relu，同一时刻最多 2 个活跃 → 规划峰值 80000 B（2 个槽），
相对朴素 160000 B **节省 50%**；分配结果显示 tensor 2/3 正确复用了 tensor 0/1 已释放的偏移。

## 2. 工业背景

边缘 SoC 的 DRAM/SRAM 极其有限，峰值内存常是能否上板的硬约束。内存规划把"按需 malloc"变成
"编译期一次性规划好的静态 arena"，既降峰值又消除运行期分配抖动（实时性关键）。

## 3. TensorRT 对应模块

≈ TensorRT 的 **workspace 内存复用** + builder 的内存分配策略；TensorRT 会为整网规划一块 workspace,
不同 layer 的临时张量在其中复用。

## 4. TVM 对应模块

≈ TVM 的 `StorageRewrite` / graph runtime 的 storage pool 规划（同样基于生命周期复用 storage id）。

## 5. TPU-MLIR 对应模块

≈ TPU-MLIR 的 LMEM/GMEM 地址分配：片上内存极小, 必须精细规划张量驻留与复用。

## 6. Ascend CANN 对应模块

≈ GE 的 `MemoryAssigner` / `BlockMemAssigner`：给计算图做连续内存块分配与复用, 控制 device 内存峰值。

## 7. 性能收益

- 峰值内存直接下降（本例 50%），决定模型能否放进受限内存。
- 静态 arena + 零运行期分配 → 推理延迟更可预测（无 malloc 抖动）。
- 复用提升 cache/局部性（相邻算子复用同地址）。

## 8. Trade-off

- 激进复用 → 引入 WAR 依赖, 限制算子并行/异步执行（与 Module 11 调度耦合）。
- 贪心 first-fit 不一定全局最优；最优是 NP-hard, 工业界普遍用贪心+启发式（by-size / by-breadth）。
- tensor 级规划是估算；真实分配应在 bufferize 后的 memref 上做（含临时/对齐/inplace 信息）。

## 9. 常见 Bug

1. **生命周期算错**：漏算"被 return 使用 → 活到结尾"会导致提前复用、数据被覆盖。
2. **重叠判定边界**：`[birth,death]` 用闭区间, overlap = `!(a.death < b.birth || b.death < a.birth)`;
   把"同一时刻定义又消费"的情况判错会错误共享。
3. **对齐**：不按硬件对齐（如 64B）分配会触发非对齐访问惩罚甚至错误。
4. **原地算子**：relu 等可原地, 若规划器不知道输入可被覆盖, 会浪费一块内存（进阶: 标注 inplace）。

## 10. 调试方法

- 看 `edge-memplan` 报告的 `[birth,death]` 与 offset 表, 人工核对复用是否正确。
- 构造生命周期交错的用例, 验证规划峰值是否等于"最大同时活跃集合的大小之和"。
- 与 bufferize 后实际 `memref.alloc` 数量对比。

## 11. Profiling 方法

- 报告里的 naive vs planned 即内存 profile 的核心指标。
- 配合 Module 12 Profiler 的内存断点, 观察运行期实际峰值是否符合规划。

## 12. 在机器人 / VLA 中的应用

VLA 多相机 + 大主干模型对内存压力极大。静态内存规划让我们在编译期确定 arena 大小、保证推理期零
分配, 既压低峰值（能放进 Jetson/Ascend 边缘内存）又保证控制环延迟可预测。多相机流的临时张量可在
arena 内跨相机复用, 进一步省内存。

> 下一步（Module 11/12）：运行时按规划好的 arena 执行计算图, 并由 Profiler 量化每算子延迟与实际内存。
