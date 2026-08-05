# exceptions_vs_error_codes

异常 vs 错误码（成功路径成本）。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 42 页：过去 C++ 异常即使不抛出也有成本，因此性能敏感代码用错误码；
**现代编译器异常只在实际抛出时产生成本**。既然抛出的异常都极端罕见，
可以安全地在性能敏感系统中使用异常，并享受异常优于错误码的表达力。

## 文件

| 文件 | 说明 |
|---|---|
| `baseline.cpp` | 错误码风格：通过输出参数 + 返回值报告错误 |
| `optimized.cpp` | 异常风格：抛 `std::runtime_error` |
| `benchmark.cpp` | 仅测成功路径（除数非 0，不抛出） |
| `tests.cpp` | 两种风格的错误行为等价 |

## 构建与运行

```bash
cmake --build build --target ch01_evc_benchmark ch01_evc_tests
./build/chapter01_zero_cost/ch01_evc_tests
./build/chapter01_zero_cost/ch01_evc_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX）

| 实现（成功路径） | mean |
|---|---|
| 错误码风格 | ~1.2 ns/iter |
| 异常风格 | ~1.2 ns/iter |

两种风格在成功路径上**几乎完全相等**（ratio ~0.96x），验证了原书观点：
现代编译器对"不抛出的异常"生成与普通分支相同的代码
（`if (b == 0) throw ...` 编译为一个 `test` + 条件分支）。

## 适用范围与注意事项

- 结论只在"成功路径"成立；真正抛出的异常有展开栈的成本（数百 ns~µs 量级）；
- 性能热点循环内应避免高频抛出（如用异常做循环控制）；
- 本项目在 `-fno-omit-frame-pointer` 下构建，异常路径成本与开启与否无关，
  但 profile 时保留帧指针更有利于回溯。
