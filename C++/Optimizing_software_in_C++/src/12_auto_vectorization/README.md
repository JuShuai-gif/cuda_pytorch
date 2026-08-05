# 12_auto_vectorization —— 自动向量化

对应笔记：`note/09_SIMD与自动向量化.md`、`note/05_编译器优化原理.md`（8.1 向量化）
对应 PDF：第 118-121 页、第 73 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `12_vectorizable` | `__restrict__` + 规则访问的循环（编译器可向量化） |
| `12_non_vectorizable` | 无别名保证 + 循环内分支（难向量化） |
| `12_benchmark` | 两者对比（`12_benchmark` 单独加 `-mavx2`） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 12_vectorizable 12_non_vectorizable 12_benchmark -j

./build/12_auto_vectorization/12_benchmark
```

## 查看向量化报告

```bash
# GCC：查看哪些循环被向量化/被错过
g++ -O3 -mavx2 -std=c++17 -fopt-info-vec -fopt-info-vec-missed \
    src/12_auto_vectorization/vectorizable.cpp -c -o /tmp/v.o
g++ -O3 -mavx2 -std=c++17 -fopt-info-vec -fopt-info-vec-missed \
    src/12_auto_vectorization/non_vectorizable.cpp -c -o /tmp/nv.o

# Clang 对应选项（本机未装 clang，仅供记录）
# clang++ -O3 -mavx2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize ...

# 确认汇编有 ymm / vaddps
g++ -O3 -mavx2 -std=c++17 -S -masm=intel src/12_auto_vectorization/vectorizable.cpp -o /tmp/v.s
grep -E "vaddps|ymm" /tmp/v.s | head
```

## 预期结果与解读

- `add_two`（restrict）在 `-mavx2` 下被向量化，每次处理 8 个 float。
- `add_conditional` 的分支阻碍干净向量化；在 `-O3 -mavx2 -fno-trapping-math` 下编译器也可能处理（计算两侧再混合，PDF 第 120 页）。
- 报告日志会明确说明哪些循环向量化失败及原因（别名、分支、副作用等，PDF 第 119 页）。

## 注意事项

- 本机 CPU 支持 AVX2、无 AVX-512；`-mavx2` 是本机可用的最大向量宽度。
- 向量化与否及其质量依赖编译器版本；本机 g++ 13.3 能力较强。
