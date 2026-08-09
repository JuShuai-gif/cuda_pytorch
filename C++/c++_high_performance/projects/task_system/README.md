# task_system

综合实践：线程池任务系统。

> 综合 Chapter 10（线程、mutex、条件变量、future/packaged_task）的教学实现。

## 功能

- 固定数量工作线程，从共享队列取 `std::function<void()>` 任务执行；
- 队列用 `mutex + condition_variable` 保护；空闲线程休眠（`cv.wait`）不占 CPU；
- `submit(fn, args...)` 返回 `std::future<Result>`：可取值、可捕获异常；
- 析构：置 stop 标志 → `notify_all` → join 全部线程。

## 构建与运行

```bash
cmake -S projects -B build-projects
cmake --build build-projects --target task_system_example task_system_tests \
    task_system_benchmark -j

./build-projects/task_system/task_system_example
./build-projects/task_system/task_system_tests
./build-projects/task_system/task_system_benchmark
```

## 关键验证

- tests：future 返回值/多任务/任务多于线程（排队）/异常传播/参数转发；
- benchmark（本机 24 线程）：并行求和 vs 串行约 1.6x（受内存带宽限制，
  串行求和已被编译器向量化；改用计算密集任务差异更大）；
- example：平方和与公式 `n(n-1)(2n-1)/6` 核对。

## 注意

- 线程数默认 `hardware_concurrency()`；CPU 密集任务线程数 ≈ 核数即可
  （Ch10 性能指南）；
- 任务粒度过小，队列锁争用会吃掉并行收益；
- 用 `packaged_task` 自动建 promise，避免手写 promise/future 配对。
