# cache_thrashing

缓存抖动：访问顺序决定性能。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 106-107 页：内存按缓存行（64 字节）读取。矩阵填充用 `matrix[i][j]`
（行主序）时连续访问、缓存友好；改成 `matrix[j][i]`（列主序）时每次访问
跨行跳转，产生 L1 缓存缺失。作者机器上从 40ms 恶化到 800ms。

## 构建与运行

```bash
cmake --build build --target ch04_cache_thrashing_benchmark
./build/chapter04_data_structures/ch04_cache_thrashing_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，8192² int 矩阵 = 256 MiB）

| 访问方式 | mean | 相对 |
|---|---|---|
| 行主序 `matrix[i][j]` | ~33 ms | 1.0x |
| 列主序 `matrix[j][i]` | ~180 ms | **~5.4x** |

列主序慢 5.4 倍，与书中规律一致（作者机器 20 倍差异取决于缓存行大小、
预取器与矩阵规模）。

## 重要实现细节

- 矩阵 256 MiB 必须堆分配（`std::vector`），放栈上会栈溢出（本实验
  早期版本即因此崩溃）；
- 必须"读回"矩阵使写入可观察，否则编译器把纯写入循环当死代码消除
  （`-O3` 下测得 0ms 假象）。

## 结论

- 局部性（locality）比单次内存访问的"常数时间"更重要；
- 嵌套循环尽量按内存布局顺序（行主序）访问。
