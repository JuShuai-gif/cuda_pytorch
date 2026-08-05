# 03_branch —— 分支预测

对应笔记：`note/04_C++语言结构性能分析.md`（7.12）、`note/08_乱序执行与指令级并行.md`
对应 PDF：第 43-45 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `03_baseline` | 数据依赖的 `if (x >= 128)` 分支；可传 `s` 参数排序数据 |
| `03_optimized` | 无分支写法（谓词算术 `x & ~((x-128)>>31)`） |
| `03_benchmark` | 对比：随机数据 vs 排序数据 × 分支 vs 无分支 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 03_baseline 03_optimized 03_benchmark -j

./build/03_branch/03_benchmark
./build/03_branch/03_baseline
./build/03_branch/03_optimized
```

## 用 perf 验证 branch-misses

```bash
# 需要 root（perf_event_paranoid=4）
sudo perf stat -e branch-misses,branches ./build/03_branch/03_baseline
sudo perf stat -e branch-misses,branches ./build/03_branch/03_optimized
```

## 预期结果与解读

- **随机数据**：`branch` 误预测约 50%（PDF 第 44 页），`branchless` 无分支、明显更快。
- **排序数据**：`branch` 几乎总是走同一边，预测极好，可能比 `branchless` 还快（分支预测成功时几乎零成本）。
- 结论：**分支快的条件是"可预测"**；数据随机时分支惩罚（15-25 周期）超过无分支算术的成本。

## 注意事项

- `branchless_sum` 要求 `x` 是 int 且逻辑正确（`x-128` 溢出时语义需注意，本实验数据范围 0-255 安全）。
- **本机实测（g++ 13.3, -O3）**：`objdump` 显示 `03_benchmark` 汇编中有 `cmov`——编译器已把 `if (x >= 128) sum += x` 自动转成条件移动（PDF 第 44 页"compiler can automatically replace a branch by a conditional move"）。因此 branch 与 branchless 版本在本机耗时几乎相同，随机/排序差异也很小。这是"编译器替你做了"的活例：**先看汇编再决定是否手写 branchless**。
- 若想看到明显差异，可改用更复杂、编译器无法转 cmov 的分支（如循环体很大），或用 `perf stat -e branch-misses` 观察仍存在的分支（如内层另有难以消除的分支）。
