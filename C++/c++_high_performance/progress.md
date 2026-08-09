# 当前阶段

阶段十五：最终验收 + note/19 完成报告（✅ 项目全部完成）

# 已完成

- [x] 读取 PDF 完整目录（360 条 TOC，全书 362 页）
- [x] 确认全书章节结构（11 章 + 附录）
- [x] 提取每章起止页码（PDF 页码与印刷页码对照）
- [x] 提取每章重要小节与页码
- [x] 建立 PDF 章节与 note 文件对应关系（note/00 §2）
- [x] 建立知识点与 src 实验对应关系（note/00 §4）
- [x] 创建目录结构（note/、src/common、chapter01-11、projects、scripts、benchmark_results）
- [x] 创建 note/00_全书导读与学习路线.md
- [x] 创建 note/README.md
- [x] 创建 note/01~19 占位文件
- [x] 创建 progress.md
- [x] src/common 基础设施（benchmark/statistics/compiler_barrier/test_utils/system_info）
- [x] src/CMakeLists.txt 顶层构建
- [x] note/01 + src/chapter01_zero_cost 全部实验
- [x] note/02 + src/chapter02_modern_cpp 全部实验
- [x] note/03 + src/chapter03_measurement 全部实验
- [x] note/04 + src/chapter04_data_structures 全部实验
- [x] note/05 + src/chapter05_iterators 全部实验
- [x] note/06 + src/chapter06_algorithms 全部实验
- [x] note/07 + src/chapter07_memory 全部实验
- [x] note/08_模板元编程与编译期计算.md
- [x] src/chapter08_metaprogramming 全部 8 个实验
- [x] note/09_代理对象与惰性求值.md
- [x] src/chapter09_lazy_evaluation 全部 5 个实验
- [x] note/10_并发与C++内存模型.md
- [x] src/chapter10_concurrency 全部 8 个实验
- [x] note/11_Parallel_STL与GPU计算.md
- [x] src/chapter11_parallel_stl 全部 6 个实验（含 Boost.Compute GPU）
- [x] 补充笔记 12_Benchmark设计指南.md
- [x] 补充笔记 13_编译器优化与汇编分析.md
- [x] 补充笔记 14_C++17到C++20_C++23现代化补充.md
- [x] 补充笔记 15_高性能C++常见误区.md
- [x] 补充笔记 16_综合实践项目.md
- [x] 补充笔记 17_高性能C++检查清单.md
- [x] 补充笔记 18_术语表.md
- [x] 综合实践项目 projects/object_pool
- [x] 综合实践项目 projects/task_system
- [x] 综合实践项目 projects/parallel_pipeline
- [x] 综合实践项目 projects/high_performance_container
- [x] scripts/build.sh + clean_build.sh
- [x] scripts/run_all.sh + benchmark_all.sh
- [x] scripts/perf_stat.sh + perf_record.sh
- [x] scripts/assembly.sh
- [x] scripts/sanitizer_test.sh + thread_sanitizer_test.sh
- [x] 最终验收：全量构建 0 警告、59 个 tests 全绿、ASan 全过
- [x] 清理 build-asan/build-debug 旧目录，新增 .gitignore
- [x] note/19_项目完成报告.md

# 当前正在处理

无（项目已全部完成）。

# 下一步

- 项目完成。可参考 note/19 与 note/README.md 使用。

# 编译状态

- Release（GCC 13.3，ENABLE_BOOST_COMPUTE + ENABLE_OPENCL=ON）：✅ chapter01-11 全量构建通过，0 编译警告
- Projects（projects/ 独立工程）：✅ 4 个项目 0 警告，全部 tests 通过
- ASan/UBSan/LSan：✅ 59 个 test 二进制全部通过
- TSan：✅ 编译通过；运行受 Ubuntu 24.04 内核 ASLR 限制
- GPU（OpenCL/Boost.Compute）：✅ NVIDIA RTX 4070 实测通过
- 汇编验证：compile_time_hash 编译为 `mov $294`
- perf：⚠️ paranoid=4 需 root（脚本检测提示）

# 编译状态

- Release（GCC 13.3，ENABLE_BOOST_COMPUTE + ENABLE_OPENCL=ON）：✅ chapter01-11 全量构建通过，0 编译警告
- Projects（projects/ 独立工程）：✅ 4 个项目 0 警告，全部 tests 通过
- ASan/UBSan/LSan：✅ chapter08-10 全部测试通过
- TSan：✅ 编译通过；运行受 Ubuntu 24.04 内核 ASLR 限制
- GPU（OpenCL/Boost.Compute）：✅ NVIDIA RTX 4070 实测通过
- 汇编验证：compile_time_hash 编译为 `mov $294`
- perf：⚠️ paranoid=4 需 root（同前）

# Chapter 11 关键验证（GCC 13.3，本机 i7-13700K，24 线程）

- par_transform benchmark：naive ≈ 10.6x、分治 chunk=10000 ≈ 17.2x（书中 8 核 3.8-5.9x）
- par_copy_if benchmark（400 万元素）：is_odd → split 3.9x / sync 0.09x（伪共享灾难）；
  is_prime → split 9.9x / sync 10.1x
- 手写 par_transform_naive 修复书中缺陷：固定任务数在 n 不可整除时漏尾部元素
- execution_policies：seq/par/par_unseq 结果一致；**GCC 13 下 `<execution>` 谓词抛异常
  直接 terminate（书中 GCC 7 会传播），已用普通 transform 演示异常路径并记录**
- parallel_for：LinearRange + for_each 跨章复用，par/par_unseq 均正确
- boost_compute（RTX 4070）：圆面积 GPU/CPU 结果一致；GPU 排序 CPU 验证通过；
  box filter 2D kernel 与 CPU 参考 0 失配

# 综合实践项目验证（GCC 13.3，本机 i7-13700K，24 线程）

- object_pool：pool ≈ 0.5 ns/op vs new/delete ≈ 12 ns/op（约 25x）
- task_system：并行求和 vs 串行 ≈ 1.6x（内存带宽限制）
- parallel_pipeline：并行 vs 串行 ≈ 1.7x（轻量映射带宽受限）
- hash_set：查询与 std::unordered_set 相当（ratio ~1.0x）
- 修复问题：object_pool free list 需块 ≥ 指针大小；hash_set 探测必须 hash % cap
  （曾致段错误）；benchmark 需 asm volatile barrier 防编译器消除测量

# Chapter 8 关键验证（GCC 13.3）

- templates：pow_n float/int 独立实例化；const_pow_n<int,2>/<int,3> 独立
- type_traits：is_same_v/is_floating_point_v/remove_pointer_t 编译期
- is_detected：探测 mess_with_arms/to_string/name_ 均正确
- enable_if + is_detected：print() 按能力分派（Squid/Salmon）
- constexpr：integral_constant<sum(1,2,3)> 编译通过（编译期求值证明）
- if constexpr：generic_mod int 走 %、float 走 fmod，互不干扰
- 汇编：hash_function("abc") 折叠为 mov $294
- heterogeneous：sizeof(any)=16 vs sizeof(variant<int,string,bool>)=40
- reflection：shire==copy 正确、shire<mordor 正确、非反射类型无运算符
- safe_cast：float->int 回转检查、ptr->uintptr_t 正确

# 已发现并修复的重要问题

- CHP_CHECK 宏遇逗号参数（pow_n(2.0F,3)）→ 需额外括号包裹
- std::to_string 不接受 std::string → 改用 if constexpr 分派打印
- const_pow_n 模板参数顺序（T,N）需显式 <int,2>（非 constexpr 时不可用于 static_assert）
- reflection tests 缺 operator!= → 补全
- par_transform_naive 固定任务数在 n 不可整除时漏尾部元素 → 改 while (start < n)
- `<execution>` 谓词抛异常 GCC 13 直接 terminate → example 改用普通 transform 演示
- vector<Account>（含 mutex）不可移动 → 改用 std::array
- future.get() 二次调用抛 future_error → tests 断言该行为
- 综合项目：object_pool free list 需块 ≥ 指针大小；hash_set 探测需 hash % cap
  （曾致段错误）；benchmark 需 asm volatile barrier 防编译器消除测量
- false_sharing benchmark 需 compiler_barrier 防循环提升（否则伪共享不发生）

# 已验证环境

| 项目 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS | Linux 6.17.0-40-generic x86_64 |
| CPU | Intel Core i7-13700K | 24 线程 |
| GPU | NVIDIA GeForce RTX 4070 | OpenCL 运行时实测可用 |
| GCC | 13.3.0 | 可用（C++20 std::ranges 已验证） |
| Clang | 未安装 | 汇编对照脚本需自动跳过 |
| CMake | 4.1.3 | 可用 |
| TBB | oneTBB 2021.x | Parallel STL 后端 |
| 标准库 | libstdc++（GCC 13） | is_detected 在 <experimental/type_traits> |
| Boost | 1.83 | /usr/include/boost/version.hpp |
| OpenCL | CUDA 12.8 头文件 + NVIDIA ICD | 头文件在 /usr/local/cuda-12.8/.../include/CL |
| perf | 可用但受限 | paranoid=4，需 root 才能采样 |
| gprof | 可用 | 无需 root |
| valgrind | 未检出 | Sanitizer 方案不受影响 |

# 未解决问题

- perf 采样受限（paranoid=4）：脚本需检测并友好提示。
- Clang 未安装：assembly.sh 需自动跳过 Clang 分支。
- `<execution>` 谓词抛异常在 GCC 13 直接 terminate（libstdc++ 行为，已记录）。
- 未初始化 git 仓库（用户未要求）。
- `build/`、`build-debug/`、`build-asan/` 目录位于项目根，最终验收前统一清理重建。
