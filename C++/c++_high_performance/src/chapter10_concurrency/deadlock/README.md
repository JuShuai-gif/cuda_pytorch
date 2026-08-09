# deadlock

用 `std::lock` 同时获取多把锁，避免死锁。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 287、294-295 页：

- 一线程持有一把锁再取另一把锁，两线程按**相反顺序**加锁时互相等待 → 死锁；
- 转账需要同时保护 `from` 和 `to` 两个账户；
- `std::unique_lock{mutex, std::defer_lock}` 构造**未加锁**的 RAII 锁对象；
- `std::lock(lock1, lock2)` **同时**获取两把锁（内部避免顺序竞争），
  返回后统一释放，不存在"持一把等另一把"的死锁窗口。

## 构建与运行

```bash
cmake --build build --target ch10_deadlock_example ch10_deadlock_tests -j
./build/chapter10_concurrency/ch10_deadlock_example
./build/chapter10_concurrency/ch10_deadlock_tests
```

## 关键点

- tests 用 4 账户 + 8 条混合方向转账（含 `{0,1}` 与 `{1,0}` 反向组合），
  金额守恒断言恒成立、程序不会卡死；
- 若改为"先锁 from 再锁 to"，反向转账线程就会死锁（本实验用 `std::lock`
  刻意规避，演示正确写法）。

## 注意

- `std::unique_lock` 比 `lock_guard` 灵活（可 defer、可提前 unlock）；
  不需要 defer 的临界区用 `lock_guard` 更简单。
