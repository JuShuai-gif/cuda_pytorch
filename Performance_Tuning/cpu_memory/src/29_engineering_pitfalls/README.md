# Experiment 29: Engineering Pitfalls

> 实验 29 是"工程实战中的坑"系列：每个坑一个独立小实验，均可在本机复现。
> 对应的综合讲解见 note/35_工程实战中的坑.md。

## 为什么需要这一组实验

前 28 组实验演示"正确的原理"；这组实验演示"工程中容易踩的坑"——它们
往往来自编译器优化行为、语言语义陷阱、工具误用，而不是缺少硬件知识。

每个坑实验都满足：可编译、可运行、输出 checksum 或可验证的事实、多轮统计、
不编造数据。

## 坑清单

| 可执行文件 | 坑 | 一句话 |
|---|---|---|
| p1_debug_vs_release | 在 Debug 构建里测性能 | -O0 下测得的时间毫无意义 |
| p2_dead_code_elimination | 基准代码被优化掉 | 结果未使用的循环可能被整个删除，测量"看起来飞快" |
| p3_vector_bool | std::vector<bool> 不是普通 vector | 位打包 + proxy，慢且不能取 bool& |
| p4_shared_ptr_contention | 隐式原子计数竞争 | 每线程复制 shared_ptr 触发同一缓存行上的原子 RMW |
| p5_alignment | 以为堆对象天然对齐 | malloc/new 只保证 max_align_t，不保证 64B |
| p6_benchmark_noise | 单次运行下结论 | 冷启动/频率爬升/背景噪声让单次结果误导 |
| p7_atomic_memory_order | 在 x86 上纠结 seq_cst vs relaxed | lock 前缀开销主导，二者差别小；真坑是"不必要的原子操作"（局部归约） |
| p8_volatile_not_atomic | 用 volatile 当线程安全 | volatile 只是不缓存寄存器，不是原子 |
| p9_hugepage_verify | 以为 madvise 一定生效 | THP 是提示，要用 /proc/self/smaps 验证 |

## 编译与运行

```bash
cd src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/p1_debug_vs_release
./build/p8_volatile_not_atomic     # 会显示 volatile 结果错误
./build/p9_hugepage_verify         # 会显示 AnonHugePages 是否真的生效
```

> 注意：p1 用 `__attribute__((optimize("O0")))` 在 Release 二进制里模拟
> Debug 函数，仅用于演示；真实工程不要这样混用。

## 与本项目其它部分的关联

- p2/p6 呼应 benchmark 公共组件（mean/median/min/stddev + checksum + 预热）。
- p5 呼应 note/19（对齐）与图 6.4（未对齐访问代价）。
- p7/p8 呼应 note/24（原子操作与内存序）。
- p9 呼应 note/15（大页）与 note/29（页错误验证）。
