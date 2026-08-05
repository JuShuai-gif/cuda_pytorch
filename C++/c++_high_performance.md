你现在是一名精通现代 C++、编译器优化、性能分析、泛型编程和并发编程的高级 C++ 性能工程师。

当前项目目录中存在一本 PDF：/home/hpc/ghr_code/cuda_pytorch/C++/C++ High Performance - Boost and optimize the performance of your C++ 17 code.pdf


请完整分析这本 PDF，并将其整理成一套可以系统学习和实践现代 C++ 高性能编程的中文项目。

所有学习笔记放入：

note/

所有可编译、可运行的代码放入：

src/

所有构建、运行、Benchmark 和性能分析脚本放入：

scripts/

性能测试结果放入：

benchmark_results/

不要逐句翻译整本书，也不要直接大量复制书中的代码。

你的目标是将书中的知识转化为：

1. 系统化中文笔记；
2. 可以独立运行的 C++17 示例；
3. 未优化版本与优化版本对照实验；
4. 可重复执行的性能 Benchmark；
5. 编译器汇编分析；
6. perf 性能分析；
7. 章节练习和综合实践项目；
8. 从基础 C++ 到高性能现代 C++ 的完整学习路线。

==================================================
一、项目目录结构
==================================================

请在 /home/hpc/ghr_code/cuda_pytorch/C++/c++_high_performance 创建以下目录结构：

├── note/
│   ├── README.md
│   ├── 00_全书导读与学习路线.md
│   ├── 01_C++与零成本抽象.md
│   ├── 02_现代C++核心特性.md
│   ├── 03_性能测量与优化方法论.md
│   ├── 04_数据结构与内存布局.md
│   ├── 05_迭代器原理与自定义迭代器.md
│   ├── 06_STL算法与Ranges.md
│   ├── 07_内存管理与自定义分配器.md
│   ├── 08_模板元编程与编译期计算.md
│   ├── 09_代理对象与惰性求值.md
│   ├── 10_并发与C++内存模型.md
│   ├── 11_Parallel_STL与GPU计算.md
│   ├── 12_Benchmark设计指南.md
│   ├── 13_编译器优化与汇编分析.md
│   ├── 14_C++17到C++20_C++23现代化补充.md
│   ├── 15_高性能C++常见误区.md
│   ├── 16_综合实践项目.md
│   ├── 17_高性能C++检查清单.md
│   ├── 18_术语表.md
│   └── 19_项目完成报告.md
│
├── src/
│   ├── CMakeLists.txt
│   ├── README.md
│   │
│   ├── common/
│   │   ├── benchmark.hpp
│   │   ├── benchmark.cpp
│   │   ├── statistics.hpp
│   │   ├── statistics.cpp
│   │   ├── compiler_barrier.hpp
│   │   ├── test_utils.hpp
│   │   └── system_info.cpp
│   │
│   ├── chapter01_zero_cost/
│   ├── chapter02_modern_cpp/
│   ├── chapter03_measurement/
│   ├── chapter04_data_structures/
│   ├── chapter05_iterators/
│   ├── chapter06_algorithms/
│   ├── chapter07_memory/
│   ├── chapter08_metaprogramming/
│   ├── chapter09_lazy_evaluation/
│   ├── chapter10_concurrency/
│   ├── chapter11_parallel_stl/
│   │
│   └── projects/
│       ├── object_pool/
│       ├── task_system/
│       ├── parallel_pipeline/
│       └── high_performance_container/
│
├── scripts/
│   ├── build.sh
│   ├── clean_build.sh
│   ├── run_all.sh
│   ├── benchmark_all.sh
│   ├── perf_stat.sh
│   ├── perf_record.sh
│   ├── sanitizer_test.sh
│   ├── thread_sanitizer_test.sh
│   ├── assembly.sh
│   └── system_info.sh
│
└── benchmark_results/

==================================================
二、执行方式
==================================================

必须采用分阶段、可恢复的方式执行。

每完成一个阶段：

1. 输出本阶段完成的文件；
2. 输出实际执行的命令；
3. 输出编译和测试结果；
4. 输出未解决问题；
5. 更新 note/README.md 中的进度；
6. 更新根目录中的 progress.md；
7. 停止继续执行；
8. 等待用户输入“继续”后，再开始下一阶段。

不得一次性草率生成整本书的全部内容。

如果 OpenCode 会话中断，下一次执行时必须：

1. 先读取 progress.md；
2. 检查已经存在的 note 和 src；
3. 从上次未完成的位置继续；
4. 不要覆盖已经完成且经过验证的内容；
5. 必要时采用增量修改。

==================================================
三、第一阶段任务
==================================================

第一阶段只完成项目规划，不批量生成代码。

需要完成：

1. 阅读 PDF 的完整目录；
2. 确认全书章节结构；
3. 提取每章起止页码；
4. 提取每章的重要小节；
5. 建立 PDF 章节与 note 文件的对应关系；
6. 建立知识点与 src 实验的对应关系；
7. 创建目录结构；
8. 创建 note/00_全书导读与学习路线.md；
9. 创建 note/README.md；
10. 创建 progress.md；
11. 输出后续阶段计划；
12. 完成后停止，等待用户输入“继续”。

章节映射采用以下格式：

| PDF章节 | PDF页码 | 核心内容 | 对应笔记 | 对应代码 | 状态 |
|---|---:|---|---|---|---|

代码映射采用以下格式：

| 知识点 | 未优化实验 | 优化实验 | Benchmark | 对应章节 |
|---|---|---|---|---|

==================================================
四、内容来源要求
==================================================

所有内容必须区分为以下三类。

### 1. PDF 内容

来自 PDF 的知识必须注明：

> 来源：PDF 第 XX～XX 页  
> 原始章节：Chapter X  
> 原始小节：Performance consideration of std::function

不得逐句翻译，也不得大段复制原文。

应当：

- 提炼作者观点；
- 保留作者的组织逻辑；
- 使用自己的中文重新解释；
- 说明书中示例要证明的结论。

### 2. 原理解释

可以补充理解书中内容所需的 C++、编译器、操作系统和 CPU 原理。

例如：

- 内联；
- 类型擦除；
- Small Buffer Optimization；
- Cache Locality；
- ABI；
- vtable；
- Copy Elision；
- 指令重排；
- Cache Coherence；
- False Sharing。

补充内容必须标记：

> 原理补充：以下内容用于帮助理解，不是 PDF 原文。

### 3. 现代化补充

本书出版于 2018 年，部分内容可能发生变化，例如：

- Ranges 当时尚未正式进入 C++20；
- Parallel STL 的编译器支持；
- std::execution 的实现情况；
- std::is_detected；
- Concepts；
- std::span；
- std::jthread；
- std::stop_token；
- std::pmr；
- std::ranges；
- std::expected；
- C++20 协程；
- Boost Compute；
- OpenCL；
- GPU 编程方案。

现代化补充必须标记：

> 现代补充：以下内容不是原书内容，而是 C++20/C++23 或当前工具链中的对应实现。

不要擅自修改作者的历史观点。

==================================================
五、Markdown 笔记规范
==================================================

每个章节笔记至少包含：

# 标题

## 1. 本章解决什么问题

使用通俗语言说明本章目标。

## 2. 前置知识

列出理解本章需要掌握的知识。

## 3. PDF 章节结构

列出原书章节、小节和页码。

## 4. 核心概念

逐一解释重要术语。

## 5. 工作原理

说明语言、编译器、内存或 CPU 层面发生了什么。

## 6. PDF 核心观点

总结原书观点，并标注页码。

## 7. 简单示例

使用最小代码解释概念。

## 8. 未优化版本

展示容易产生性能问题的代码。

## 9. 优化版本

展示更合理的实现。

## 10. 为什么可能更快

至少从以下维度分析：

- 时间复杂度；
- 空间复杂度；
- 动态内存分配；
- 对象复制和移动；
- Cache Locality；
- 分支；
- 函数调用和内联；
- 类型擦除；
- 虚函数；
- 锁竞争；
- 原子操作；
- 编译期计算；
- SIMD 或并行能力；
- 编译器优化空间。

## 11. 该优化什么时候可能无效

必须说明：

- 数据量是否足够大；
- 编译器是否已经自动优化；
- 是否会增加代码复杂度；
- 是否影响可维护性；
- 是否依赖特定平台；
- 是否产生额外内存开销；
- 是否可能破坏异常安全。

## 12. 如何验证

提供：

- 编译命令；
- 运行命令；
- Benchmark 命令；
- perf 命令；
- Sanitizer 命令；
- 查看汇编的方法。

## 13. 汇编观察点

说明应关注：

- 是否内联；
- 是否存在函数间接调用；
- 是否存在 new/delete；
- 是否出现 memcpy；
- 是否展开循环；
- 是否消除分支；
- 是否进行了常量折叠；
- 是否自动向量化。

## 14. 常见错误

解释容易误用的写法。

## 15. 实践练习

提供 3～5 个由简单到困难的练习。

## 16. 对应代码

列出 src 中对应目录和文件。

## 17. 本章总结

提炼本章最重要的原则。

==================================================
六、代码组织规范
==================================================

每个实验目录建议采用：

experiment_name/
├── baseline.cpp
├── optimized.cpp
├── benchmark.cpp
├── tests.cpp
├── README.md
└── CMakeLists.txt

其中：

- baseline.cpp：直观但可能低效的实现；
- optimized.cpp：优化实现；
- benchmark.cpp：性能测试；
- tests.cpp：正确性测试；
- README.md：原理、运行方法和结果解释。

并非所有概念都必须机械地生成 baseline 和 optimized。

例如 const correctness、自定义迭代器、type traits 等概念，可以采用：

- example.cpp；
- compile_error_example.cpp；
- tests.cpp；
- README.md。

compile_error_example.cpp 不应默认加入正常构建目标，应通过单独脚本演示预期编译失败。

==================================================
七、代码质量要求
==================================================

所有代码必须：

1. 使用 C++17；
2. 在 Linux 下支持 GCC 和 Clang；
3. 使用 CMake；
4. 可以独立编译；
5. 不留下 TODO；
6. 不使用伪代码；
7. 不引用不存在的文件；
8. 不调用未定义函数；
9. 不引入未定义行为；
10. 遵循 RAII；
11. 明确对象所有权；
12. 保证异常安全；
13. 使用固定随机种子；
14. 输出校验结果；
15. 对 baseline 和 optimized 做正确性比较；
16. 对多线程程序避免数据竞争；
17. 对可选依赖进行检测；
18. 缺少依赖时跳过对应示例，而不是导致整个项目失败。

不要直接复制书中大量代码。

应当根据书中的思想重新实现教学示例，并注明：

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

==================================================
八、Benchmark 规范
==================================================

实现统一 Benchmark 工具：

src/common/benchmark.hpp
src/common/benchmark.cpp

至少支持：

struct BenchmarkResult {
    double mean_ns;
    double median_ns;
    double min_ns;
    double max_ns;
    double stddev_ns;
    std::size_t iterations;
};

所有 Benchmark 必须：

1. 使用 Release 或 RelWithDebInfo；
2. 执行预热；
3. 执行多轮；
4. 输出平均值；
5. 输出中位数；
6. 输出最小值；
7. 输出最大值；
8. 输出标准差；
9. 输出迭代次数；
10. 输出 checksum；
11. 防止测试代码被编译器删除；
12. 区分初始化时间和核心计算时间；
13. 保证比较双方输入一致；
14. 使用现实数据规模；
15. 不根据单次结果得出结论；
16. 不编造性能提升比例。

使用以下手段防止代码被删除时，必须解释原因：

- 返回结果；
- checksum；
- volatile；
- std::atomic_signal_fence；
- compiler barrier；
- asm volatile。

不得滥用 volatile 作为线程同步机制。

==================================================
九、各章必须完成的实验
==================================================

--------------------------------------------------
Chapter 1：C++ 与零成本抽象
--------------------------------------------------

实现并解释：

1. C 风格链表搜索与 STL 算法搜索；
2. 直接循环与 std::count；
3. 值语义与引用语义；
4. 连续对象存储与指针对象存储；
5. const correctness；
6. RAII 与手工资源释放；
7. 异常与错误码的资源安全对比；
8. 零成本抽象的汇编验证。

重点不是预设 STL 一定更快，而是验证：

- 生成的汇编是否接近；
- 抽象是否产生额外运行时成本；
- 编译器是否能够内联。

--------------------------------------------------
Chapter 2：现代 C++ 核心特性
--------------------------------------------------

实现：

1. auto 类型推导；
2. const reference；
3. forwarding reference；
4. Lambda 捕获值与捕获引用；
5. Lambda 与仿函数；
6. Lambda 与函数指针；
7. Lambda 与 std::function；
8. std::function 类型擦除开销；
9. 小捕获与大捕获；
10. Copy 与 Move；
11. Rule of Three；
12. Rule of Five；
13. Rule of Zero；
14. noexcept 对容器移动的影响；
15. std::optional；
16. std::any；
17. optional、variant、指针和特殊值方案对比。

必须特别测试：

- std::function 是否发生堆分配；
- 间接调用是否阻止内联；
- move 是否真的减少了资源复制；
- 错误使用 std::move 是否反而更差；
- 空析构函数如何影响隐式 Move。

--------------------------------------------------
Chapter 3：性能测量
--------------------------------------------------

实现：

1. O(1)、O(log n)、O(n)、O(n log n)、O(n²) 实验；
2. 不同数据规模增长曲线；
3. 摊销复杂度；
4. vector 扩容；
5. latency 与 throughput；
6. CPU-bound 与 memory-bound；
7. std::chrono Benchmark；
8. Release 与 Debug 对比；
9. 热身前后对比；
10. perf stat；
11. perf record；
12. perf report；
13. gprof 可选示例；
14. 热点函数识别；
15. 性能回归测试。

优化流程必须遵循：

测量
→ 定位热点
→ 建立假设
→ 修改
→ 再次测量
→ 评估复杂度和维护成本
→ 决定保留或回退

--------------------------------------------------
Chapter 4：数据结构
--------------------------------------------------

比较：

1. std::array；
2. std::vector；
3. std::deque；
4. std::list；
5. std::forward_list；
6. std::map；
7. std::unordered_map；
8. std::set；
9. std::unordered_set；
10. std::priority_queue；
11. std::string；
12. vector<bool> 的特殊行为。

实验至少包括：

- 顺序遍历；
- 随机访问；
- 中间插入；
- 尾部插入；
- 查找；
- 删除；
- 排序；
- 哈希质量；
- reserve；
- rehash；
- 不同 load_factor；
- 连续内存与链式内存；
- AoS 与 Parallel Arrays/SoA。

不得只比较耗时，还要解释复杂度、分配次数和内存布局。

--------------------------------------------------
Chapter 5：迭代器
--------------------------------------------------

实现：

1. Input Iterator；
2. Forward Iterator；
3. Bidirectional Iterator；
4. Random Access Iterator；
5. iterator_traits；
6. 自定义整数迭代器；
7. 浮点范围迭代器；
8. Iterator Pair；
9. Range 对象；
10. make_linear_range；
11. 根据 Iterator Category 选择算法；
12. 无效迭代器使用示例。

必须验证自定义迭代器可以正确用于标准算法。

--------------------------------------------------
Chapter 6：STL 算法与 Ranges
--------------------------------------------------

实现并比较：

1. 手写循环与 std::find；
2. 手写循环与 std::count；
3. 手写循环与 std::transform；
4. 手写循环与 std::copy_if；
5. 手写循环与 std::accumulate；
6. std::sort；
7. std::partial_sort；
8. std::nth_element；
9. 只排序真正需要的数据；
10. 自定义比较器；
11. Predicate；
12. 输出迭代器；
13. Move 与 noexcept 对算法的影响；
14. 多个算法组合；
15. Range-v3 或 C++20 std::ranges 的现代补充。

项目主体使用 C++17。

C++20 Ranges 示例应单独放入：

src/chapter06_algorithms/cpp20_ranges/

并通过 CMake 选项控制：

ENABLE_CPP20_EXAMPLES

--------------------------------------------------
Chapter 7：内存管理
--------------------------------------------------

实现：

1. 栈对象与堆对象；
2. malloc/free；
3. new/delete；
4. Placement New；
5. 对象生命周期；
6. 内存对齐；
7. Padding；
8. sizeof 与成员排列；
9. unique_ptr；
10. shared_ptr；
11. weak_ptr；
12. make_shared 与独立分配；
13. Small String Optimization；
14. Arena；
15. Object Pool；
16. 自定义 STL Allocator；
17. std::pmr 现代补充；
18. 内存分配次数统计。

自定义 Arena 必须正确处理：

- 对齐；
- 容量；
- 越界；
- 生命周期；
- 析构；
- 异常；
- 不可复制或移动规则。

不得写一个只能在理想情况下工作的不安全内存池。

--------------------------------------------------
Chapter 8：模板元编程与编译期计算
--------------------------------------------------

实现：

1. 非类型模板参数；
2. static_assert；
3. type_traits；
4. decltype；
5. enable_if；
6. SFINAE；
7. detection idiom；
8. constexpr；
9. if constexpr；
10. 编译期与运行时计算对比；
11. tuple；
12. structured bindings；
13. variadic templates；
14. parameter pack；
15. any；
16. variant；
17. visitor；
18. 静态异构容器；
19. 动态异构容器；
20. 编译期字符串哈希；
21. 安全 cast；
22. 简化反射示例；
23. Concepts 现代补充。

每个编译期优化需要通过以下方式验证：

- static_assert；
- 编译器输出；
- 汇编；
- 二进制符号；
- 运行时是否仍有相关计算。

不要只说 constexpr 更快，要确认计算是否真正发生在编译期。

--------------------------------------------------
Chapter 9：代理对象与惰性求值
--------------------------------------------------

实现：

1. eager string concatenation；
2. lazy string concatenation proxy；
3. 临时对象和分配次数统计；
4. 二维点距离比较；
5. sqrt 直接计算；
6. 平方距离比较；
7. DistProxy；
8. 延迟求值；
9. rvalue-qualified member function；
10. operator overloading；
11. pipe operator；
12. infix operator；
13. Proxy 生命周期陷阱；
14. 悬空引用示例；
15. 表达式模板基础补充。

必须评估：

- 减少了多少临时对象；
- 是否减少动态分配；
- 编译器是否已经消除临时对象；
- Proxy 是否增加 API 复杂度；
- Proxy 是否容易产生悬空引用。

--------------------------------------------------
Chapter 10：并发
--------------------------------------------------

实现：

1. std::thread；
2. join 与 detach；
3. 数据竞争；
4. mutex；
5. lock_guard；
6. unique_lock；
7. scoped_lock 现代补充；
8. deadlock；
9. 避免死锁；
10. condition_variable；
11. promise；
12. future；
13. async；
14. packaged_task；
15. atomic；
16. shared_ptr 在线程中的行为；
17. C++ 内存模型；
18. instruction reordering；
19. memory_order_relaxed；
20. memory_order_acquire；
21. memory_order_release；
22. memory_order_seq_cst；
23. lock-free queue；
24. 锁竞争；
25. False Sharing；
26. Thread Affinity；
27. 每线程局部数据与 Reduction；
28. ThreadSanitizer。

所有并发代码必须：

- 不得存在未说明的数据竞争；
- 不得依赖 sleep 保证顺序；
- 不得把 volatile 当作原子同步；
- 不得声称 lock-free 而不检查；
- 使用 is_lock_free 或 is_always_lock_free；
- 对 lock-free queue 详细说明适用范围和内存回收问题。

--------------------------------------------------
Chapter 11：Parallel STL 与 GPU
--------------------------------------------------

实现：

1. 串行 std::transform；
2. 手写 naive parallel transform；
3. Divide and Conquer；
4. parallel count_if；
5. parallel copy_if；
6. synchronized write position；
7. 两阶段 copy_if；
8. std::execution::seq；
9. std::execution::par；
10. std::execution::par_unseq；
11. std::accumulate 与 std::reduce；
12. std::transform_reduce；
13. 并行 for_each；
14. 不同输入规模下的并行开销；
15. 不同线程数；
16. 负载均衡；
17. False Sharing；
18. 异常处理；
19. Boost Compute；
20. OpenCL 可选实验。

Parallel STL 和 Boost Compute 必须作为可选模块。

CMake 选项：

ENABLE_PARALLEL_STL
ENABLE_BOOST_COMPUTE
ENABLE_OPENCL

如果工具链或依赖不支持：

- 自动跳过；
- 给出检测结果；
- 不得导致整个项目失败；
- 不得声称实验已经验证。

另外在 note/14 中说明现代替代方案，但不要偏离本书主体。

==================================================
十、编译系统
==================================================

统一使用 CMake。

至少支持：

- Debug；
- Release；
- RelWithDebInfo。

默认：

- C++17；
- GCC 或 Clang；
- 开启警告；
- 保留 Frame Pointer；
- pthread。

推荐参数：

-Wall
-Wextra
-Wpedantic
-Wconversion
-Wshadow

Release：

-O3
-DNDEBUG

RelWithDebInfo：

-O2
-g
-fno-omit-frame-pointer

提供 CMake 选项：

ENABLE_TESTS
ENABLE_BENCHMARKS
ENABLE_SANITIZERS
ENABLE_THREAD_SANITIZER
ENABLE_CPP20_EXAMPLES
ENABLE_PARALLEL_STL
ENABLE_BOOST
ENABLE_BOOST_COMPUTE
ENABLE_OPENCL
ENABLE_NATIVE_OPTIMIZATION

不要默认强制：

-march=native
-mavx2
-mavx512f

==================================================
十一、正确性测试与性能测试
==================================================

正确性测试和性能测试必须分开。

正确性测试验证：

- 输出是否正确；
- 边界条件；
- 异常行为；
- 资源释放；
- 线程安全；
- baseline 与 optimized 等价性。

性能测试验证：

- 时间；
- 吞吐量；
- 内存分配；
- 扩展性；
- 编译器优化；
- 线程数量变化；
- 输入规模变化。

可以使用 Google Test，但必须作为可选依赖。

默认情况下，应保证没有 Google Test 也能构建核心示例。

不得在 CMake 配置过程中强制从网络下载依赖。

==================================================
十二、脚本要求
==================================================

scripts/build.sh：

- 创建 build；
- 使用 Release 配置；
- 编译全部核心示例；
- 遇到编译错误立即退出。

scripts/clean_build.sh：

- 删除旧 build；
- 重新配置；
- 重新编译。

scripts/run_all.sh：

- 运行全部普通示例；
- 显示实验名称；
- 记录失败项目。

scripts/benchmark_all.sh：

- 运行全部 Benchmark；
- 把结果保存到 benchmark_results；
- 文件名包含时间戳；
- 同时保存系统、编译器和构建参数。

scripts/perf_stat.sh：

至少收集：

cycles
instructions
branches
branch-misses
cache-references
cache-misses
context-switches
cpu-migrations

scripts/perf_record.sh：

支持：

./scripts/perf_record.sh ./build/target_name

scripts/assembly.sh：

支持：

./scripts/assembly.sh source.cpp

输出：

- GCC 汇编；
- Clang 汇编；
- 优化版本；
- 未优化版本。

scripts/sanitizer_test.sh：

运行：

- AddressSanitizer；
- UndefinedBehaviorSanitizer；
- LeakSanitizer。

scripts/thread_sanitizer_test.sh：

单独使用 ThreadSanitizer 构建和运行并发示例。

==================================================
十三、性能结论规则
==================================================

禁止直接写：

- vector 永远比 list 快；
- move 永远比 copy 快；
- std::function 一定发生堆分配；
- Lambda 一定没有开销；
- STL 算法一定比循环快；
- constexpr 一定提高运行速度；
- 无锁一定比加锁快；
- 多线程一定更快；
- Parallel STL 一定可以扩展；
- Arena 一定比 malloc 快。

正确表达方式应当是：

- 在本次输入规模和当前环境下；
- 当前编译器生成的汇编显示；
- 当前测试观察到；
- 该实现可能受 Small Buffer Optimization 影响；
- 结果依赖标准库实现；
- 结果依赖硬件、工具链和数据规模；
- 需要通过实际 Benchmark 验证。

==================================================
十四、README 要求
==================================================

note/README.md 必须包含：

1. 项目介绍；
2. PDF 信息；
3. 适合人群；
4. 前置知识；
5. 目录结构；
6. 阅读顺序；
7. 笔记索引；
8. 实验索引；
9. 构建方法；
10. 运行方法；
11. Benchmark 方法；
12. perf 方法；
13. 汇编查看方法；
14. Sanitizer 使用方法；
15. 可选依赖；
16. 当前进度；
17. 已验证环境；
18. 未验证内容。

建立四条学习路线：

### 路线一：现代 C++ 基础

Chapter 1
→ Chapter 2
→ Chapter 5
→ Chapter 6

### 路线二：性能工程

Chapter 3
→ Chapter 4
→ Chapter 7
→ Benchmark
→ perf
→ 汇编分析

### 路线三：高级泛型编程

Chapter 5
→ Chapter 6
→ Chapter 8
→ Chapter 9

### 路线四：并发与并行

Chapter 10
→ False Sharing
→ C++ Memory Model
→ Lock-Free
→ Chapter 11

==================================================
十五、进度文件
==================================================

创建 progress.md，格式如下：

# 当前阶段

阶段一：全书分析与项目规划

# 已完成

- [x] PDF 目录分析
- [ ] Chapter 1 笔记
- [ ] Chapter 1 代码
...

# 当前正在处理

无

# 下一步

等待用户输入“继续”。

# 编译状态

尚未开始。

# 已验证环境

记录：

- 操作系统；
- CPU；
- GCC；
- Clang；
- CMake；
- 标准库；
- Boost；
- OpenCL。

# 未解决问题

记录当前遇到的问题。

==================================================
十六、最终验收
==================================================

完成所有阶段后：

1. 删除旧 build；
2. 从空目录重新配置；
3. 编译全部核心代码；
4. 编译所有当前环境支持的可选模块；
5. 运行所有正确性测试；
6. 运行所有 Benchmark；
7. 运行 Sanitizer；
8. 运行 ThreadSanitizer；
9. 检查所有 Markdown 链接；
10. 检查所有代码路径；
11. 检查空文件；
12. 检查 TODO 和 FIXME；
13. 检查伪代码；
14. 检查未使用源码；
15. 检查未验证却声称通过的内容；
16. 检查书中观点与现代补充是否明确区分；
17. 生成 note/19_项目完成报告.md。

完成报告中必须区分：

- 编译通过；
- 运行通过；
- 测试通过；
- 仅完成代码但未运行；
- 因缺少硬件或依赖而跳过；
- 理论说明；
- 现代补充。

==================================================
十七、禁止事项
==================================================

禁止：

- 一次性草率生成整本书；
- 逐句翻译 PDF；
- 大段复制书中原文；
- 大段复制原书代码；
- 编造 PDF 内容；
- 编造性能数据；
- 编造性能提升比例；
- 未编译就声称编译通过；
- 未运行就声称运行通过；
- 未测试就声称正确；
- 使用单次测试得出结论；
- 使用 Debug 性能得出结论；
- 忽略输入规模；
- 忽略编译器和标准库差异；
- 将 C++20/C++23 内容冒充成原书内容；
- 强制联网下载依赖；
- 因 Boost、OpenCL 或 Parallel STL 不可用导致核心项目失败。

现在开始执行阶段一。

本轮只完成：

1. 阅读完整 PDF 目录；
2. 提取所有章节、小节和页码；
3. 创建目录结构；
4. 创建章节映射表；
5. 创建代码实验规划；
6. 创建 note/00_全书导读与学习路线.md；
7. 创建 note/README.md；
8. 创建 progress.md；
9. 输出实际创建的文件；
10. 停止执行并等待用户输入“继续”。

本轮不要批量生成后续章节笔记，不要开始实现全部代码。



22222222222222222222222222222222222222222222222222222222
读取 progress.md，继续下一阶段。

本轮只处理 Chapter 1：A Brief Introduction to C++。

要求：

1. 完整阅读 Chapter 1；
2. 创建 note/01_C++与零成本抽象.md；
3. 实现 src/chapter01_zero_cost；
4. 包含正确性测试和 Benchmark；
5. 对零成本抽象进行汇编验证；
6. 实际执行 CMake 配置、编译和运行；
7. 不得编造性能数据；
8. 更新 note/README.md 和 progress.md；
9. 完成本章后停止，等待我输入“继续”。

333333333333333333333333333333333333333333333333333333333333333333333
读取 progress.md，继续处理 Chapter 3：Measuring Performance。

重点完成：

1. Big O 与数据规模实验；
2. vector 摊销复杂度实验；
3. Benchmark 公共组件；
4. 热身、多轮测试、统计信息；
5. perf stat；
6. perf record；
7. 热点定位；
8. 性能回归测试；
9. Debug、Release 和 RelWithDebInfo 对比。

必须实际编译和运行。
不得使用单次结果下结论。
完成后更新 progress.md 并停止。



