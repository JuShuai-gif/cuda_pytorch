# 18_benchmark —— 性能测试陷阱

对应笔记：`note/14_性能测试与Benchmark.md`、`note/17_性能优化检查清单.md`
对应 PDF：第 167-171 页、第 85 页

## 实验内容

| 可执行文件 | 演示的陷阱 |
|------------|-----------|
| `18_debug` / `18_release` | Debug（`-O0`）vs Release（`-O3`）同一源码，性能天差地别（PDF 第 85 页） |
| `18_warmup` | 第一次运行（冷缓存）vs 预热后（热缓存）（PDF 第 168 页） |
| `18_size` | 数据规模过小：64 KiB 到 128 MiB 扫描，看到缓存层级（PDF 第 170 页） |
| `18_eliminate` | 编译器消除代码：结果未使用的循环可能被完全删掉 |
| `18_fluctuation` | 结果波动：9 次原始读数 vs 预热+中位数（PDF 第 168 页） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 18_debug 18_release 18_warmup 18_size 18_eliminate 18_fluctuation -j

./build/18_benchmark/18_debug
./build/18_benchmark/18_release
./build/18_benchmark/18_warmup
./build/18_benchmark/18_size
./build/18_benchmark/18_eliminate
./build/18_benchmark/18_fluctuation
```

## 预期结果与解读

- `18_debug` 比 `18_release` 慢一个数量级以上：`-O0` 每行都按源码执行、无优化（PDF 第 85 页）。
- `18_warmup` 第 1 次读数明显偏高（数据未进缓存）。
- `18_size` 超过 L2（32 MiB）/L3（36 MiB）后耗时跳升。
- `18_eliminate` 的第二个循环（结果不使用）在 Release 下可能被完全删除——**这就是为什么 benchmark 必须消费计算结果**。
- `18_fluctuation`：单次读数波动大，中位数稳定。

```bash
# 确认第二个循环被删掉
g++ -O3 -std=c++17 -S -masm=intel src/18_benchmark/eliminate.cpp -o /tmp/e.s
wc -l /tmp/e.s    # 应该很短
```

## 注意事项

- 用 `18_release`（`-O3`）测性能；`18_debug` 只用于调试。
- 本机 CPU 频率动态变化（i9-14900HX，睿频到 5.8 GHz），单次读数噪声较大，务必多轮取中位数。
