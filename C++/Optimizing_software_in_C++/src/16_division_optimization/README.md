# 16_division_optimization —— 除法优化

对应笔记：`note/12_专项优化技巧.md`（14.4/14.5/14.6）
对应 PDF：第 149-153 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `16_baseline` | 运行时变量除数（慢 `idiv`） |
| `16_optimized` | 常数除数（乘法+移位）、无符号、2 的幂移位、浮点乘倒数 |
| `16_benchmark` | 全部变体对比 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 16_baseline 16_optimized 16_benchmark -j

./build/16_division_optimization/16_benchmark
./build/16_division_optimization/16_baseline
./build/16_division_optimization/16_optimized
```

## 预期结果与解读

- `int_div_variable` 明显慢于 `int_div_const`：常数除数被编译器转成 `imul`+`shr`（PDF 第 150 页）。
- `int_div_unsigned` 略快于 signed（修正指令更少，PDF 第 150 页）。
- `float_div_recip` 快于 `float_div_variable`（除法 20-45 周期 vs 乘法 3-8 周期，PDF 第 152 页）。

```bash
# 确认常数除法编译成乘法而非除法指令
g++ -O3 -std=c++17 -S -masm=intel src/16_division_optimization/benchmark.cpp -o /tmp/div.s
grep -cE "idiv" /tmp/div.s    # div_const 应为 0 次
grep -cE "imul" /tmp/div.s    # 出现乘法
```

## 注意事项

- `x >> 3` 对负数不等于 `x / 8`（算术右移向负无穷取整）；本实验数据为正，安全。通用代码需用带修正的写法。
- 除法结果一致性：不同变体对相同输入产生相同整数结果（浮点版本因乘倒数有微小误差）。
