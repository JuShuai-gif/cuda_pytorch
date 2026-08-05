# 17_template_metaprogramming —— 模板与编译期优化

对应笔记：`note/13_模板与编译期优化.md`
对应 PDF：第 163-167 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `17_baseline` | 运行时 `std::pow` + 运行时位扫描 |
| `17_optimized` | `if constexpr` 编译期幂 + `constexpr` 位扫描（编译期算好） |
| `17_benchmark` | `std::pow` vs `integerPower<10>` + `constexpr` 阶乘表 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 17_baseline 17_optimized 17_benchmark -j

./build/17_template_metaprogramming/17_benchmark
./build/17_template_metaprogramming/17_baseline
./build/17_template_metaprogramming/17_optimized
```

## 预期结果与解读

- `integerPower<10>` 远快于 `std::pow(x,10)`：编译期展开为 4 次乘法；`pow` 走通用对数/指数路径（PDF 第 163-164 页）。
- `bit_scan_reverse` 在编译期算好，运行期只是常量（PDF 第 167 页）。
- `fact_table` 由 `constexpr` 在编译期生成（PDF 第 167 页"编译期计算表格"）。

```bash
# 确认 optimized 版没有调用 pow
g++ -O3 -std=c++17 -S -masm=intel src/17_template_metaprogramming/optimized.cpp -o /tmp/t.s
grep -cE "call.*pow" /tmp/t.s
```

## 注意事项

- 编译需 `-std=c++17`（`if constexpr`）。
- 编译期计算的收益只针对"参数为编译期常数"的场景；运行时参数按普通函数执行。
