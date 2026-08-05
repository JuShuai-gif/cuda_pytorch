# noexcept_move

`noexcept` 对容器移动的影响，以及空析构函数陷阱。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 72 页：**"Do not forget to mark your move-constructors and move-assignment
operators as noexcept ... Not marking them noexcept prevents STL containers and
algorithms from utilizing them."**

`std::vector` 扩容用 `std::move_if_noexcept`：移动构造 `noexcept` → 移动；
否则为保证异常安全退化为**复制**（复制可能抛异常，但扩容可回滚）。

PDF 第 77-78 页：**空析构函数陷阱**。`struct Point { int x,y; ~Point(){} }`
中手写（空）析构函数使类型**不再 trivially copyable**，`std::copy` 无法用
`memmove` 而退化为逐元素循环；同时它抑制隐式移动构造的生成。

## 构建与运行

```bash
cmake --build build --target ch02_noexcept_example ch02_noexcept_benchmark ch02_noexcept_tests
./build/chapter02_modern_cpp/ch02_noexcept_example
./build/chapter02_modern_cpp/ch02_noexcept_tests
./build/chapter02_modern_cpp/ch02_noexcept_benchmark
```

## 结果解释（GCC 13.3 Release）

示例输出（100 次 `emplace_back` 触发多次扩容）：

```
vector<MoveNoexcept>  growth: copies=0 moves=127    <- 用移动
vector<MoveThrowing>  growth: copies=127 moves=0    <- 退化为复制
```

汇编对比（`-O3 -S`）：

| 类型 | `std::copy` 生成 |
|---|---|
| `PointPlain`（无析构） | `jmp memmove` |
| `PointEmptyDtor`（空析构） | 逐元素 `mov` 循环 |

## 结论

- 移动构造必须 `noexcept`，否则 STL 容器退化为复制（本实验用计数器实测）；
- 手写空析构函数 `~T(){}` 应改为 `~T() = default`（仍是用户声明、抑制移动，
  但保持 trivial copyability，`memmove` 可用）；最佳是**不声明析构函数**（Rule of Zero）。

> 现代补充：C++20 起可用 `std::is_nothrow_move_constructible_v<T>` 在编译期
> 检查；`std::move_if_noexcept` 行为仍是 `vector` 扩容的判定依据。
