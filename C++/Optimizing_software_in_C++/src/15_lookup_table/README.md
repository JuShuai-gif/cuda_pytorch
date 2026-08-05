# 15_lookup_table —— 查表优化

对应笔记：`note/12_专项优化技巧.md`（14.1）
对应 PDF：第 144-146 页

## 实验内容

| 可执行文件 | 说明 |
|------------|------|
| `15_baseline` | 循环计算阶乘 + 分支函数 |
| `15_optimized` | `static const` 查表（阶乘表 + 相位表） |
| `15_benchmark` | 循环 vs 查表；小表（L1 命中）vs 大表（被驱逐出缓存） |

## 编译与运行

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target 15_baseline 15_optimized 15_benchmark -j

./build/15_lookup_table/15_benchmark
./build/15_lookup_table/15_baseline
./build/15_lookup_table/15_optimized
```

## 预期结果与解读

- `factorial_table` 明显快于 `factorial_loop`：查表只要几个周期（PDF 第 144 页）。
- `small_table_lookup`（4 KiB 表，L1 内）远快于 `big_table_lookup`（64 MiB 表，随机访问基本全部未命中，PDF 第 145 页"表被驱逐则不适合查表"）。
- 表必须 `const`（编译器知道不变）且小（待在缓存里）才有优势。

## 注意事项

- 查表无法向量化；若查表阻止更快的向量化代码，就别用（PDF 第 145 页）。
- `big_table_lookup` 的 64 MiB 表配合随机索引，实测可能比计算还慢——这正是"查表有适用条件"的演示。
