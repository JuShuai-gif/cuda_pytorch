# contiguous_vs_pointer

连续对象存储 vs 指针对象存储。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 30-32 页对比 C++ 与 Java 的对象布局：C++ 的 `std::vector<Car>` 把对象
连续放在一块内存；Java 的 `ArrayList<Car>` 只存引用，每个对象单独堆分配。

两个后果：

1. **分配次数**：C++ 一次分配 vs Java 七次分配；
2. **访问模式**：连续内存顺序访问缓存友好；指针跳转造成随机访存。

## 文件

| 文件 | 说明 |
|---|---|
| `baseline.cpp` | `vector<unique_ptr<Particle>>`，每次访问需解引用 |
| `optimized.cpp` | `vector<Particle>` 连续存储 |
| `benchmark.cpp` | 相同数据（固定种子生成），2M 粒子求和 |
| `tests.cpp` | 等价性（相同数据下结果必须一致） |

## 构建与运行

```bash
cmake --build build --target ch01_cvp_benchmark ch01_cvp_tests
./build/chapter01_zero_cost/ch01_cvp_tests
./build/chapter01_zero_cost/ch01_cvp_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，2M 粒子）

| 实现 | mean | 相对 |
|---|---|---|
| `vector<Particle>` 连续 | ~1.4 ms | 1.00x |
| `vector<unique_ptr<Particle>>` | ~2.9 ms | ~2.1x |

指针版本慢约 2 倍。原因（限定本次环境）：

- 每访问一个粒子触发一次指针解引用，进入一个随机 cache line；
- 解引用路径还阻止了编译器对 `mass` 求和做 SIMD 向量化（潜在别名）。

## 适用条件

该收益依赖数据规模：只有遍历/处理对象集本身是热点时，连续存储才有意义。
如果对象偶发访问且数量小，差异可忽略。
