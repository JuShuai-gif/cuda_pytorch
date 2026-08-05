# 11_false_sharing —— 伪共享

对应笔记：`note/07_多线程优化.md`
对应 PDF：第 112 页（多线程写同一缓存行）

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `11_baseline` | 两个线程自增相邻的 `long long`（同一缓存行） |
| `11_optimized` | `struct alignas(64) Counter`，各占一个缓存行 |
| `11_benchmark` | 同缓存行 vs 独立缓存行的完整对比（各 1 亿次自增） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 11_baseline 11_optimized 11_benchmark -j

./build/11_false_sharing/11_benchmark
./build/11_false_sharing/11_baseline
./build/11_false_sharing/11_optimized
```

## 预期结果与解读

- `same_line` 明显慢于 `padded`：两个线程自增同一缓存行中的不同变量时，缓存行在核间来回传递（cache line ping-pong），每次自增都要等所有权（PDF 第 112 页）。
- 差距大小取决于核间缓存一致性延迟与 CPU 型号；本机（i9-14900HX）通常能看到数倍差异。

```bash
# 用 perf 观察缓存行为差异
sudo perf stat ./build/11_false_sharing/11_baseline
sudo perf stat ./build/11_false_sharing/11_optimized
```

## 注意事项

- 缓存行通常 64 字节；未来 CPU 可能 128/256 字节（PDF 第 112 页）。
- 线程绑定到同一物理核/不同核也会改变结果（可用 `taskset` 尝试）。
- **本机实测（i9-14900HX, -O3）**：差距约 5-15%（`same_line` 中位数约 35 us vs `padded` 约 32 us）。原因是本机缓存一致性极快、且该循环以整数自增为主（单线程本就快）；在缓存一致性较慢或线程较多时差距更大。**伪共享的影响强烈依赖机器与负载**，需要实测而不是假定。
