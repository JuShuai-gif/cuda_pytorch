# 14_cpu_dispatch —— CPU 指令集分发

对应笔记：`note/11_CPU指令集分发.md`
对应 PDF：第 135-141 页（Example 13.1）

## 实验内容

| 文件 | 说明 |
|------|------|
| `dispatch.cpp` | scalar/SSE2 实现 + 首次调用分发器 + `--force` 测试开关 |
| `impl_avx2.cpp` | AVX2 实现（独立 TU，仅此文件 `-mavx2`） |
| `14_cpu_dispatch` | 最终可执行文件 |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 14_cpu_dispatch -j

./build/14_cpu_dispatch/14_cpu_dispatch
./build/14_cpu_dispatch/14_cpu_dispatch --force scalar
./build/14_cpu_dispatch/14_cpu_dispatch --force sse2
./build/14_cpu_dispatch/14_cpu_dispatch --force avx2
```

## 预期结果与解读

- 默认运行：检测到本机指令集级别（i9-14900HX → 8，AVX2），首次调用后函数指针指向 AVX2 版本（PDF 第 140 页，Example 13.1）。
- 三个 `--force` 分支都应与 scalar 参考结果一致（正确性验证，PDF 第 139 页）。
- `--force avx2` 强制 AVX2；若在无 AVX-512 的本机尝试不存在的 AVX-512 分支会崩溃——这正是需要 CPUID 保护的原因（PDF 第 124 页）。

## 注意事项

- 本机无 AVX-512，因此没有 AVX-512 分支；代码结构演示了如何在有 AVX-512 的机器上扩展（`cpu_info` 已支持检测级别 10）。
- 分发器只执行一次（函数指针改写），后续调用零开销（PDF 第 139 页）。
