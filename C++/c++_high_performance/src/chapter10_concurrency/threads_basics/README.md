# threads_basics

std::thread 生命周期与线程标识。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 289-292 页：

- `std::thread{callable, args...}`：创建线程，参数按 `std::bind` 方式转发；
- `join()`：阻塞等待线程结束；`detach()`：后台继续运行；
- **析构时必须已 join 或 detach**，否则程序调用 `std::terminate()`（abort）；
- `joinable()`：默认构造 / 已 join / 已 detach / 被 move 走的线程为 false；
- `hardware_concurrency()`：硬件并发数（本机 CPU 线程数）。

## 构建与运行

```bash
cmake --build build --target ch10_threads_basics_example ch10_threads_basics_tests -j
./build/chapter10_concurrency/ch10_threads_basics_example
./build/chapter10_concurrency/ch10_threads_basics_tests
```

## 关键点

- 忘记 join/detach 就析构会 abort——生产代码用 RAII 包装（如
  `std::jthread`，C++20）或确保路径全都能到达 join；
- `detach` 后线程在后台运行，main 退出后可能仍执行，慎用。
