# Chapter 10: Parallel Algorithms (C++17 并行 STL)

## 文件说明

| 文件 | 内容 |
|------|------|
| `01_parallel_sort.cpp` | 并行排序: `std::sort(par)`, 手动分块归并排序两种实现 |
| `02_parallel_for_each.cpp` | 并行 for_each: 图像处理场景, seq vs par vs par_unseq |
| `03_parallel_transform_reduce.cpp` | 并行 MapReduce: 点积、平方和、归约统计、min/max |
| `04_execution_policies.cpp` | 执行策略: seq/par/par_unseq 对比、安全注意事项、运行时选择 |

## 编译要求

**GCC**: 需要链接 TBB (Threading Building Blocks)

```bash
# 安装 TBB
sudo apt install libtbb-dev

# 编译 (单个文件)
g++ -std=c++20 -O2 -pthread 01_parallel_sort.cpp -ltbb -o parallel_sort

# 或使用 CMake (需取消 CMakeLists.txt 中 TBB 相关注释)
mkdir build && cd build
cmake .. && make -j$(nproc)
```

**MSVC**: 内置支持, 直接编译即可。

**Clang**: 需要 libc++ 和 PSTL 支持。

## 关键技术点

- **执行策略**: `seq` (顺序), `par` (并行), `par_unseq` (并行+向量化)
- **算法**: `sort`, `for_each`, `transform_reduce`, `reduce` 等支持策略参数
- **安全限制**: `par_unseq` 禁止使用锁、内存分配等有副作用的操作
- **TBB 依赖**: GCC 的 `<execution>` 实现依赖 Intel TBB
- **运行时选择**: 可根据数据量和硬件线程数动态选择策略
