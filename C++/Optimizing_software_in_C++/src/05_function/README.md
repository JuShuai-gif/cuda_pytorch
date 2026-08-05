# 05_function —— 函数调用

对应笔记：`note/04_C++语言结构性能分析.md`（7.14-7.18）、`note/05_编译器优化原理.md`（8.1 内联）
对应 PDF：第 48-52 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `05_baseline` | 热循环内经 `std::function` 调用（类型擦除，难以内联） |
| `05_optimized` | 热循环内经模板 + inline 函数调用（完全内联） |
| `05_benchmark` | 对比普通函数 / static / inline / 函数指针 / std::function / 虚函数 / lambda |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 05_baseline 05_optimized 05_benchmark -j

./build/05_function/05_benchmark
./build/05_function/05_baseline
./build/05_function/05_optimized
```

## 预期结果与解读

- `plain/static/inline/lambda` 在 `-O3` 下通常全部被内联，耗时几乎相同。
- `std_function` 与 `virtual_call` 通常明显更慢：`std::function` 是类型擦除间接调用、虚函数经虚表（PDF 第 55 页）。
- `func_pointer`：若目标地址每次相同，预测良好，开销很小（PDF 第 37 页）。

```bash
# 看汇编确认内联
g++ -O3 -std=c++17 -S -masm=intel src/05_function/benchmark.cpp -o /tmp/fn.s
grep -cE "call" /tmp/fn.s   # 内联版本应没有 call（热循环内）
```

## 注意事项

- 虚函数在**调用目标每次相同时**（本实验）只多几个周期；当目标变化时才出现 10-20 周期误预测（PDF 第 55 页）。
- 编译器可能自动去虚化（PDF 第 76 页），进一步缩小差异。
