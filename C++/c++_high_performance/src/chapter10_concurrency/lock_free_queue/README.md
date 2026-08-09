# lock_free_queue

单读单写无锁队列（环形缓冲）。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 309-311 页：

- 固定容量环形缓冲；**只有 `size_` 是 `std::atomic<size_t>`**；
- `read_pos_` 只归读线程、`write_pos_` 只归写线程，无需原子；
- 算法保证读写线程**永不并发访问同一数组元素** → 无数据竞争；
- 读写双方全程无锁、无分配，适合**实时音频线程**（不能阻塞/加锁/分配）。

## 构建与运行

```bash
cmake --build build --target ch10_lock_free_queue_example \
    ch10_lock_free_queue_tests -j
./build/chapter10_concurrency/ch10_lock_free_queue_example
./build/chapter10_concurrency/ch10_lock_free_queue_tests
```

## 关键点

- `is_lock_free()` 在构造时断言，保证 size_ 真无锁；
- tests 用 10 万元素生产者→消费者压测：总数与总和断言恒成立；
- 双向通信用两个队列（main→audio 与 audio→main）。

## 注意

- 只能**单写单读**；多写或多读需要额外的同步手段（超出本书范围）；
- 满/空时 `push`/`pop` 抛异常，调用方必须处理（实时线程应避免到达该状态）。
