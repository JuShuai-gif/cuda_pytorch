# lambda_basics

Lambda 的捕获语义与行为验证。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 48-55 页：

- lambda 本质是**带 `operator()` 的类**，捕获块 = 类的成员变量 + 构造函数（PDF 51 页）；
- 捕获值 vs 捕获引用：值捕获在创建时复制，之后外部变量变化不影响 lambda；
- 每个 lambda 有**唯一类型**；
- `mutable` 允许修改值捕获的副本（lambda 默认 `operator() const`）；
- 捕获初始化 `[c = expr]`、捕获全部 `[=]`/`[&]`、`[this]`/`[*this]`。

## 构建与运行

```bash
cmake --build build --target ch02_lambda_example
./build/chapter02_modern_cpp/ch02_lambda_example
```

## 关键点

- `th` 改为 4 后：值捕获计数 `v>3`（3 个），引用捕获计数 `v>4`（2 个）——行为差异直接可测；
- lambda 与手写类 `IsAbove` 结果一致，验证"lambda 是类的语法糖"；
- `mutable` lambda 修改的是自己的副本，外部 `v` 不变；引用捕获则直接改外部变量；
- `[=]` 只捕获实际用到的变量（`sizeof` 验证）；
- 可用 `[c=0]() mutable { return ++c; }` 实现计数器（书中 Button 例子的核心）。

## 现代补充

> 现代补充：C++20 起无捕获 lambda 可在未求值上下文（`decltype`）使用；
> C++20 支持 `[this, x = 1]` 等混合捕获；C++23 起模板参数列表 `[]<typename T>{}` 更灵活。
