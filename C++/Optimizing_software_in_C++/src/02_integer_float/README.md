# 02_integer_float —— 整数与浮点数

对应笔记：`note/04_C++语言结构性能分析.md`（7.2/7.3）、`note/12_专项优化技巧.md`（14.7/14.8）
对应 PDF：第 29-33 页、第 152-154 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `02_baseline` | 混 float/double + 变量除数（每元素有精度转换和除法） |
| `02_optimized` | 统一 float + 乘倒数（无转换、无除法） |
| `02_benchmark` | 对比 int32/int64、float/double、变量/常数除法、int→double 转换 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 02_baseline 02_optimized 02_benchmark -j

./build/02_integer_float/02_baseline
./build/02_integer_float/02_optimized
./build/02_integer_float/02_benchmark
```

## 预期结果与解读

- `sum_int32_div_var`（变量除数）比 `sum_int32_div_const`（常数除数）慢：常数被编译器转成乘法+移位（PDF 第 150 页）。用 `-S` 可确认 `imul` 而非 `idiv`。
- float 与 double 的乘加循环在非向量代码下通常相差无几（PDF 第 32 页）。
- `conv_loop` 每元素做 int→double 转换；大数据量下比不做转换的版本慢（PDF 第 41 页）。

```bash
# 查看常数除法是否真的编译成乘法
g++ -O3 -std=c++17 -S -masm=intel src/02_integer_float/benchmark.cpp -o /tmp/if.s
grep -iE "idiv|imul" /tmp/if.s | head
```

## 验证一致性

`02_baseline` 与 `02_optimized` 因算法不同（除法 vs 乘法）结果略有浮点误差，但量级一致。`02_benchmark` 内各版本打印结果供人工核对。

## 注意事项

- 本机 g++ 13.3 在 `-O3` 下可能自动把部分循环向量化或化简，削弱差异；这正是"编译器可能替你做了"的体现（PDF 第 8 章）。
- 除法优化具体见 `src/16_division_optimization`。
