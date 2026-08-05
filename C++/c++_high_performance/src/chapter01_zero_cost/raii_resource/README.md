# raii_resource

手工资源释放 vs RAII。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 40-42 页：C++ 对象析构是**确定性**的，可以精确控制资源释放时机。
RAII（Resource Acquisition Is Initialization）把资源的获取放入构造函数、
释放放入析构函数，保证任何退出路径（提前 return、异常）都会释放资源。

对比 PDF 中的 `std::lock_guard` 例子：无论正常返回、提前 return 还是抛异常，
锁都会在作用域退出时自动释放。

## 文件

| 文件 | 说明 |
|---|---|
| `baseline.cpp` | 手工 `new`/`delete`：异常路径下资源泄漏 |
| `optimized.cpp` | `ResourceGuard`：RAII 封装，异常安全 |
| `tests.cpp` | 用构造/析构计数验证两种路径的释放行为 |
| `benchmark.cpp` | 分配+释放大循环：验证 RAII 无额外运行时成本 |

## 构建与运行

```bash
cmake --build build --target ch01_raii_benchmark ch01_raii_tests
./build/chapter01_zero_cost/ch01_raii_tests
./build/chapter01_zero_cost/ch01_raii_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX）

- tests 验证：手工版本在 `value == 0`（抛出）时构造 1 次、析构 0 次（泄漏）；
  RAII 版本构造 1 次、析构 1 次（释放）。
- benchmark 验证：`new`+`delete` 与 RAII 每调用开销比约 0.97x——**RAII 零额外成本**。
  两者底层调用同一套 `new`/`delete`，差异仅来自代码组织，而不是运行时机制。

## 适用条件

- RAII 无法解决所有问题：循环引用（用 `shared_ptr`）仍需注意，见 Chapter 7；
- 极端环境（内核、无堆场景）可能禁用堆与异常，RAII 仍可配合错误码使用
  （析构总是执行），但"释放"可能变成显式 reset。
