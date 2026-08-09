# async_tasks

任务式并发：`std::packaged_task` 与 `std::async`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 299-300 页：

- `std::packaged_task`：把普通函数包装成可调用对象，**自动搭建 promise**；
  可在任意线程里执行（`std::thread{std::move(task), args}`）；
- `std::async`：一行完成异步调用，由库决定是否开线程；函数保持普通签名，
  返回值/异常通过 future 交给调用方——推荐首选。

## 构建与运行

```bash
cmake --build build --target ch10_async_tasks_example ch10_async_tasks_tests -j
./build/chapter10_concurrency/ch10_async_tasks_example
./build/chapter10_concurrency/ch10_async_tasks_tests
```

## 关键点

- 从"线程式"（手动 thread+锁+全局）切换到"任务式"（async）后，并发代码量最小化；
- `std::launch::deferred`：任务延迟到 `get()` 时才在调用线程执行；
- 用多个 async 并行求和（example 最后一例）验证任务可并行。

## 注意

- `std::async` 默认策略可能同步执行（实现相关）；要确保并行需显式
  `std::launch::async`；
- 详见 Scott Meyers《Effective Modern C++》并发章关于 async 何时优于
  手写线程的讨论。
