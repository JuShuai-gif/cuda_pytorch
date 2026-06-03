# 学习笔记总目录

> 从 C++ 模板基础到 CUTLASS 架构 —— 18 章系统笔记

---

## 笔记使用指南

每章笔记遵循统一的五段式结构：

1. **核心问题** —— 这一章要解决什么工程难题
2. **知识体系** —— 系统化的概念图谱
3. **代码示例** —— 可编译、可运行的最小示例
4. **工业关联** —— 本章概念在 CUTLASS / TensorRT 中的位置
5. **常见陷阱** —— 实际编码中最容易踩的坑

### 如何使用这些笔记

- **初学者**：从第1章开始，逐章阅读。每章约 30-45 分钟。
- **有基础者**：直接跳到第6章（SFINAE）或第18章（CUTLASS 全景）。
- **查漏补缺**：使用下方索引表定位薄弱环节。
- **边读边练**：配合 `src/chapter-XX/` 目录下的代码练习。

---

## 18 章快速索引

| 章号 | 标题 | 核心主题 | 工业关联 |
|:----:|------|----------|----------|
| 01 | 函数模板 —— 从零开销抽象到编译期多态 | 函数模板定义、实例化、重载决议、模板实参推导 | CUTLASS `gemm()` 入口、PyTorch `at::` dispatch |
| 02 | 类模板 —— 类型参数化的容器与策略 | 类模板、成员模板、模板模板参数、PIMPL with template | CUTLASS `ThreadblockShape<M,N,K>`、TensorRT `Dims` |
| 03 | 非类型模板参数 —— 编译期常量与性能秘密 | NTTP、编译期数组大小、template<auto> C++17 | CUTLASS Tile 尺寸、Kernel Stages、Warp 数量 |
| 04 | 变参模板 —— 编译期的"任意参数"艺术 | 参数包、递归展开、折叠表达式、index_sequence | CUTLASS Epilogue 链式组合、Operator fusion |
| 05 | 模板基础技巧 —— 你需要知道的编译器约定 | typename vs class、依赖名、template 消歧义、两阶段查找 | 所有模板库的基础规则 |
| 06 | enable_if 与 SFINAE —— 编译期的"减速带" | SFINAE 原理、enable_if、void_t 检测、if constexpr 替代 | CUTLASS Kernel 选择、平台条件编译 |
| 07 | 按值传递与按引用传递 —— 模板中的参数传递哲学 | 转发引用、完美转发、decay、remove_reference | CUTLASS 参数传递、Tensor wrapper |
| 08 | 编译期编程 —— 让编译器为你写代码 | constexpr 函数、模板元编程、编译期计算 | CUTLASS Tile 分块计算、loop unrolling |
| 09 | 模板在实践中的使用 —— 从书本到工程 | 包含模型、显式实例化、模板与虚函数、预编译头 | CUTLASS 头文件组织、.inl 模式 |
| 10 | 模板基本术语 —— 说对术语才能写对代码 | 实例化、特化、偏特化、ODR、名称查找术语表 | 读懂 CUTLASS 文档的前提 |
| 11 | 泛型库 —— CUTLASS 的前身 | Policy 类、Trait 类、Tag dispatch、类型列表 | CUTLASS Traits 系统、Policy 选择器 |
| 12 | 深入模板基础 —— 参数化声明全解析 | 模板声明语法、默认实参、模板模板参数、友元模板 | CUTLASS 模板参数默认值、Collective Builder |
| 13 | 模板中的名称 —— 编译器如何找到你的代码 | 依赖名、非依赖名、ADL、注入类名、current instantiation | 调试 CUTLASS 编译错误的关键 |
| 14 | 实例化 —— 从模板到真实代码的每一步 | 按需实例化、延迟实例化、显式实例化、实例化点 | CUTLASS 编译时间优化、显式实例化策略 |
| 15 | 模板实参推导 —— SFINAE 的魔法世界 | 推导规则、引用折叠、auto 推导、decltype(auto) | CUTLASS GEMM 参数推导链 |
| 16 | 特化与重载 —— 为每种 GPU 架构定制代码 | 全特化、偏特化、函数模板特化 vs 重载 | CUTLASS SM80/SM90 架构特化 |
| 17 | 未来方向 —— Concepts 与现代化模板设计 | C++20 Concepts、requires、consteval、NTTP 浮点 | CUTLASS 3.x 使用 Concepts 约束 Kernel |
| 18 | CUTLASS 模板架构全景 —— 从书本到工业巅峰 | Device→Kernel→Collective→Threadblock→Warp→MMA 全链路 | 完整贯通前 17 章所有知识 |

---

## 推荐阅读顺序

### 路径 A：线性学习（零基础 → 精通）

```
1 → 2 → 3 → 5 → 10 → 4 → 6 → 7 → 8 → 9 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18
```

适合：模板零基础，希望系统建立知识体系。

### 路径 B：快速通道（有基础 → 工业实战）

```
5（基础约定）→ 6（SFINAE）→ 8（编译期编程）→ 11（泛型库）→ 16（特化）→ 18（CUTLASS 全景）
```

适合：已有 1-2 年 C++ 模板经验，想快速切入 CUTLASS。

### 路径 C：查漏补缺

```
按上方索引表，直接跳到目标章节。
```

每章相对独立，可以按需阅读。

---

## 学习建议

### 每章学习 checklist

- [ ] 读完核心问题和知识体系（15 分钟）
- [ ] 手写并运行代码示例（20 分钟）
- [ ] 用自己的话总结本章3个核心概念（5 分钟）
- [ ] 找到 1 个 CUTLASS 源码中的对应位置（10 分钟）
- [ ] 完成配套 `src/chapter-XX/` 练习

### 学习节奏建议

| 周次 | 章节 | 目标 |
|:----:|------|------|
| 第1周 | 1-3 | 建立模板语法基础 |
| 第2周 | 4-6 | 掌握变参和 SFINAE |
| 第3周 | 7-9 | 理解工程实践 |
| 第4周 | 10-12 | 术语和高级声明 |
| 第5周 | 13-15 | 名称查找和推导 |
| 第6周 | 16-18 | 特化、Concepts、CUTLASS 全景 |

**总计 6 周**，每周约 5-8 小时投入。

### 配套练习

- **代码练习**：`multhread/src/chapter-XX/` 每章配备独立可编译示例
- **Mini CUTLASS**：`multhread/cutlass_style/` 自实现的 GEMM 调度器
- **实战项目**：`multhread/project/mini_inference_engine/` 完整推理引擎

### 常见问题

**Q: 需要多深的 C++ 基础？**
A: 熟悉 C++ 基本语法（类、继承、STL 容器）即可开始。项目会在第 5 章讲解模板的编译器约定，这是一个很好的"校准点"。

**Q: 需要 GPU 吗？**
A: 大部分章节不需要。只有第 18 章（CUTLASS 全景）和 `cutlass_style/` 项目需要 CUDA GPU（推荐 SM80+）。前 17 章可以在任意 C++20 编译器上学习。

**Q: 18 章学完能到什么水平？**
A: 能够独立阅读 CUTLASS 源码，理解 TensorRT / PyTorch Backend 中的模板设计，并且能够自己写一个简单的 Kernel 调度器。

**Q: 跟直接读 CUTLASS 官方文档有什么区别？**
A: CUTLASS 文档假设你已经精通模板。本项目先建立模板基础，再用这些基础去"解剖" CUTLASS，是从原理到应用的完整路径。
