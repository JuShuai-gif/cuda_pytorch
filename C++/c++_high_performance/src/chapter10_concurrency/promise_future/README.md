# promise_future

用 `std::promise` / `std::future` 跨线程返回数据与错误。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 298-299 页：

- `promise` 是值的**写端**，`future` 是**读端**；
- 工作线程 `p.set_value(v)` 或 `p.set_exception(e)`；
- 调用线程 `f.get()` 未就绪则**阻塞**，就绪后取回值或抛出异常；
- 全程无共享全局变量、无显式锁，错误用常规异常机制传递。

## 构建与运行

```bash
cmake --build build --target ch10_promise_future_example \
    ch10_promise_future_tests -j
./build/chapter10_concurrency/ch10_promise_future_example
./build/chapter10_concurrency/ch10_promise_future_tests
```

## 关键点

- 传引用给线程要用 `std::ref`（`std::thread` 按值转发参数）；
- `set_exception` 用 `std::make_exception_ptr` 包装异常；
- 下一级抽象 `packaged_task` / `async` 自动搭建 promise，见 async_tasks 实验。

## 注意

- `get()` 只能成功调用有限次；重复取同一 future 的共享状态要谨慎；
- 异步任务必须能到达 `set_value`/`set_exception`，否则 `get()` 永久阻塞
  （`std::future` 析构会传播 `broken_promise`）。
