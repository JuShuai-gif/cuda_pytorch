# 04_loop —— 循环优化

对应笔记：`note/04_C++语言结构性能分析.md`（7.13）、`note/08_乱序执行与指令级并行.md`
对应 PDF：第 45-48 页、第 113-114 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `04_baseline` | 串行累加（1 个累加器，循环携带依赖链） |
| `04_optimized` | 4 累加器打破依赖链 |
| `04_benchmark` | 1/2/4/8 累加器对比 + 循环不变量外提（除法 vs 乘倒数） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 04_baseline 04_optimized 04_benchmark -j

./build/04_loop/04_benchmark
./build/04_loop/04_baseline
./build/04_loop/04_optimized
```

## 关键：先看编译器做了什么

`-O3` 下 GCC 通常会把 `sum1` 自动改写为多累加器甚至向量化，导致手写版本差异很小。**先关掉 fast-math 看手写效果，再打开看编译器效果**：

```bash
# 手写效果最明显（无 fast-math）
g++ -O2 -std=c++17 src/04_loop/benchmark.cpp -o /tmp/bench_nofast -Isrc
/tmp/bench_nofast

# 编译器自动优化后（fast-math 允许重排浮点累加）
g++ -O3 -ffast-math -std=c++17 src/04_loop/benchmark.cpp -o /tmp/bench_fast -Isrc
/tmp/bench_fast

# 用汇编确认编译器是否展开/向量化
g++ -O3 -ffast-math -std=c++17 -S -masm=intel src/04_loop/benchmark.cpp -o /tmp/l.s
grep -cE "addsd|xmm" /tmp/l.s
```

## 预期结果与解读

- 无 fast-math 时：`sum4`/`sum8` 中位数通常明显低于 `sum1`（多累加器，PDF 第 114 页）。
- `div_hoisted` 比 `div_in_loop` 快：除法 20-45 周期 vs 乘法 3-8 周期（PDF 第 152 页）。
- 累加器数并非越多越好，最优约 3-4 个（取决于加法单元数，PDF 第 114 页）。

## 注意事项

- 本实验用 `-O2`（无 fast-math）时结论最干净；`-O3 -ffast-math` 下编译器可能已替你完成大部分工作。
