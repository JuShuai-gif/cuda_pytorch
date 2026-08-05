# release_vs_debug

Release 与 Debug 构建的对比。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 97 页（性能测试最佳实践）：用**真实数据**、**有代表性**的输入测试。
本实验用同一份源码分别以 Debug（-O0）和 Release（-O3）构建，展示优化对
性能的影响，也提醒：**不要用 Debug 性能下结论**。

## 构建与运行

```bash
cmake -S src -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --target ch03_release_debug_benchmark -j
./build-debug/chapter03_measurement/ch03_release_debug_benchmark   # Debug
./build/chapter03_measurement/ch03_release_debug_benchmark         # Release
```

## 结果解释（GCC 13.3，i9-14900HX，5 次排序+计数 20 万元素）

| 构建 | 耗时 | 说明 |
|---|---|---|
| Debug（-O0） | ~94 ms | 无优化，保留所有临时对象 |
| Release（-O3） | ~8 ms | 内联、向量化、消除临时 |

Release 约快 **12 倍**。checksum 一致（结果正确）。

## 注意事项

- 本程序内部用 `NDEBUG` 宏区分构建类型并打印；
- 性能报告必须标注构建类型；
- `-fno-omit-frame-pointer` 使 Release 也可被 perf/gprof 可靠回溯栈。
