# 08_memory_cache —— 内存访问与缓存

对应笔记：`note/06_内存与缓存优化.md`（9.1-9.10）
对应 PDF：第 92-108 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `08_cache_stride` | 数组大小从 4 KiB 到 128 MiB 扫描，观察访问时延的缓存层级台阶 |
| `08_cache_random` | 顺序 vs 跨步 vs 随机访问（64 MiB 工作集） |
| `08_aos_soa` | 结构体数组 vs 数组结构体（数据布局） |
| `08_transpose` | 矩阵转置：2 的幂行距的缓存竞争（64/65、512/513） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 08_cache_stride 08_cache_random 08_aos_soa 08_transpose -j

./build/08_memory_cache/08_cache_stride
./build/08_memory_cache/08_cache_random
./build/08_memory_cache/08_aos_soa
./build/08_memory_cache/08_transpose
```

## 预期结果与解读

- `08_cache_stride`：工作集超过 L1（本机每核约 48 KiB L1d）、L2（本机 32 MiB）、L3（36 MiB）时，ns/access 会逐级跳升（PDF 第 21 页：缓存命中 2-4 周期 vs 未命中数百周期）。
- `08_cache_random`：随机访问远慢于顺序访问（PDF 第 106 页）。
- `08_transpose`：本机是 i9-14900HX（L1d 大、L3 36 MiB），512 矩阵可能不触发与 PDF 完全相同的竞争；若差异不明显，说明 512×512×8=2 MiB 完全在 L3 内。**现象随机器缓存几何而变化**（PDF 第 107 页的表格来自 P4 时代的缓存）。

```bash
# 用 perf 验证 cache-misses
sudo perf stat -e cache-misses ./build/08_memory_cache/08_cache_random
```

## 注意事项

- 缓存层级台阶的精确位置依赖本机缓存大小，用 `lscpu` 查看。
- 转置实验在 2 MiB 矩阵上可能看不到 PDF 里的 6 倍差异——那是在旧 CPU 的 L2 竞争下测得的；本实验目的是理解"关键步长"这一机制。
