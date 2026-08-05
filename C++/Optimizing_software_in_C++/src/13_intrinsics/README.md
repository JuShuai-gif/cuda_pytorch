# 13_intrinsics —— SIMD Intrinsics 编程

对应笔记：`note/10_Intrinsics编程.md`、`note/09_SIMD与自动向量化.md`
对应 PDF：第 121-124 页、第 133 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `13_baseline` | 标量：点积 / 归约 / min-max |
| `13_optimized` | SSE2 / AVX2 intrinsics（运行时用 cpu_info 选择） |
| `13_benchmark` | 标量 vs SSE2 vs AVX2 对比 + 校验和一致性 |
| `13_avx512_example` | AVX-512 掩码条件示例（编译期开启、运行时检测，无 AVX-512 时跳过） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 13_baseline 13_optimized 13_benchmark 13_avx512_example -j

./build/13_intrinsics/13_benchmark
./build/13_intrinsics/13_avx512_example
./build/13_intrinsics/13_optimized
./build/13_intrinsics/13_baseline
```

## 预期结果与解读

- `avx2_dot`/`avx2_reduce`（每指令 8 个 float）通常明显快于标量；SSE2（4 个）居中（PDF 第 115 页）。
- 点积归约受"归约树"的依赖链影响（PDF 第 129-130 页，12.6），这里用横向加法合并。
- `13_avx512_example`：本机无 AVX-512，程序打印提示并安全退出——**这本身就是"AVX-512 需要运行时保护"的演示**（PDF 第 124 页）。

## 验证一致性

`13_benchmark` 打印三个实现的点积校验和；浮点累加顺序不同会带来微小误差，量级一致即视为正确。

## 注意事项

- AVX-512 文件仅在自身目标上开 `-mavx512f`，不会污染其他目标（总 CMakeLists 默认不强制任何 AVX-512）。
- 当前环境（i9-14900HX 无 AVX-512）下 `13_avx512_example` 的掩码路径未实际执行，属"当前环境未验证，仅编译验证"。
