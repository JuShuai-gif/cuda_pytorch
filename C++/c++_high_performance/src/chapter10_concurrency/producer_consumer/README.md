# producer_consumer

用条件变量实现生产者-消费者。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 296-298 页：

- 消费者：`unique_lock` 锁后 `cv.wait(lock)`——**睡眠时释放锁**，
  唤醒返回前重新获锁；用 `while` 而非 `if` 包裹等待（应对 spurious wakeup）；
- 生产者：锁内 push 队列，**锁外** `cv.notify_one()`（通知不必持锁）；
- `done` 哨兵（-1）告诉消费者结束。

## 构建与运行

```bash
cmake --build build --target ch10_producer_consumer_example \
    ch10_producer_consumer_tests -j
./build/chapter10_concurrency/ch10_producer_consumer_example
./build/chapter10_concurrency/ch10_producer_consumer_tests
```

## 关键点

- `wait()` 必须配合 `unique_lock`（不是 `lock_guard`），因为要临时解锁；
- tests 用 1000 个整数 + 哨兵，验证"每项恰好消费一次、总和正确"；
- 消费者在无数据时**休眠不占 CPU**，而非忙等。

## 注意

- 忘记 `while` 用 `if`：spurious wakeup 或竞争消费者会读到空队列；
- 条件变量只负责"唤醒"，条件本身（队列非空）必须由数据保护。
