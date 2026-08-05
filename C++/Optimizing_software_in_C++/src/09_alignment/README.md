# 09_alignment —— 内存对齐

对应笔记：`note/06_内存与缓存优化.md`（9.5）、`note/10_Intrinsics编程.md`（12.8）
对应 PDF：第 95 页、第 118 页、第 133 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `09_baseline` | 偏移一个元素（未对齐指针）的 SIMD 循环 |
| `09_optimized` | `alignas(64)` 栈数组 + `std::aligned_alloc(64, ...)` 动态对齐分配 |
| `09_benchmark` | 打印两个缓冲区地址的对齐字节数，并对比对齐/未对齐复制循环 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 09_baseline 09_optimized 09_benchmark -j

./build/09_alignment/09_benchmark
./build/09_alignment/09_baseline
./build/09_alignment/09_optimized
```

## 预期结果与解读

- 现代 x86（SSE/AVX 时代）对**未对齐 load** 的惩罚很小甚至为零（硬件处理行跨界），所以 `aligned_copy` 与 `misaligned_copy` 可能差异不大——这正是 PDF 第 118 页"AVX 后对齐要求放宽"的体现。
- 对齐的价值在以下场景仍明确：
  - 显式向量 `_mm_load_ps`（对齐版本）在部分 CPU 上更快；
  - 对象/数组按缓存行对齐可避免与相邻数据共享缓存行（见 11_false_sharing）。
- 打印出的"对齐字节数"可直接确认 `std::aligned_alloc` 的效果。

## 注意事项

- `std::aligned_alloc` 要求 size 是对齐值的整数倍，且用 `std::free` 释放（C++17）。
- 本机为 i9-14900HX，未对齐惩罚极小属正常；在 Atom 或老 Intel CPU 上差异才明显（PDF 第 124 页）。
