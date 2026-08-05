# 06_class_virtual —— 类与虚函数

对应笔记：`note/04_C++语言结构性能分析.md`（7.19-7.25）、`note/05_编译器优化原理.md`（8.1 去虚化）
对应 PDF：第 52-57 页、第 76 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `06_baseline` | 虚函数派发 + `dynamic_cast` + RTTI（200 万元素，含 new/delete） |
| `06_optimized` | 模板编译期多态（无虚表、可内联） |
| `06_benchmark` | 普通成员函数 vs 虚调用 vs RTTI 检查 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 06_baseline 06_optimized 06_benchmark -j

./build/06_class_virtual/06_benchmark
./build/06_class_virtual/06_baseline
./build/06_class_virtual/06_optimized
```

## 预期结果与解读

- `virtual_call` 在调用目标每次相同时只比普通调用多几个周期（PDF 第 55 页）；编译器甚至可能自动去虚化（PDF 第 76 页）。
- `rtti_dyncast`：`dynamic_cast` 做运行时检查，比普通转换慢（PDF 第 42 页）。
- `06_optimized`（模板多态）：编译期解析，热循环内完全内联（PDF 第 59 页）。

## 注意事项

- 虚函数真正的代价是**目标变化时的分支误预测**（10-20 周期），本实验不体现；可自行把两/多个类交替调用观察。
- `06_baseline` 内含 200 万次 new/delete，这本身也是性能开销（PDF 第 95 页）。
