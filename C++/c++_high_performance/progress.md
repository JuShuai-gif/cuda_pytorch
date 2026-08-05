# 当前阶段

阶段九：Chapter 8 笔记与代码（已完成）

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
- [ ] Chapter 9 笔记与代码
- [ ] Chapter 10 笔记与代码
- [ ] Chapter 11 笔记与代码（含可选模块）
- [ ] 补充笔记 12-15、17、18
- [ ] 综合实践项目 projects/*
- [ ] scripts/* 构建与性能脚本
- [ ] 最终验收与 note/19 完成报告

# 当前正在处理

无（等待用户输入"继续"）。

# 下一步

- 等待用户输入"继续"后开始阶段十：Chapter 9 笔记（note/09）+ 全部代码（src/chapter09_lazy_evaluation）。

# 编译状态

- Release（GCC 13.3，ENABLE_CPP20_EXAMPLES=ON）：✅ chapter01-08 全量构建通过，0 编译警告
- ASan/UBSan/LSan：✅ chapter08 全部测试与关键示例通过
- 汇编验证：compile_time_hash 编译为 `mov $294`（编译期计算确认）
- perf：⚠️ paranoid=4 需 root（同前）

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

# 已验证环境

| 项目 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS | Linux 6.17.0-40-generic x86_64 |
| CPU | Intel Core i9-14900HX | 32 线程 |
| GPU | NVIDIA GN21-X11 | 已探测到 VGA 设备 |
| GCC | 13.3.0 | 可用（C++20 std::ranges 已验证） |
| Clang | 未安装 | 汇编对照脚本需自动跳过 |
| CMake | 4.1.3 | 可用 |
| 标准库 | libstdc++（GCC 13） | is_detected 在 <experimental/type_traits> |
| Boost | 1.83 | /usr/include/boost/version.hpp |
| OpenCL | 头文件存在（/usr/include/CL/cl.h） | 运行时（驱动/平台）待探测 |
| perf | 可用但受限 | paranoid=4，需 root 才能采样 |
| gprof | 可用 | 无需 root |
| valgrind | 未检出 | Sanitizer 方案不受影响 |

# 未解决问题

- perf 采样受限（paranoid=4）：脚本需检测并友好提示。
- Clang 未安装：assembly.sh 需自动跳过 Clang 分支。
- OpenCL 运行时可用性未知（阶段十二验证）。
- Parallel STL 的 GCC 13 支持（阶段十二验证）。
- 未初始化 git 仓库（用户未要求）。
- `build/`、`build-debug/`、`build-asan/` 目录位于项目根，最终验收前统一清理重建。
