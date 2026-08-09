# par_copy_if

手写并行 `std::copy_if()` 的两种方案。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 327-331 页：

- 并行 `copy_if` 难点：多个线程**并发写同一目标位置**是未定义行为；
- **sync（原子写位置）**：全局 `std::atomic<size_t>`，每命中 `fetch_add`
  取唯一下标。正确但多线程写相邻缓存行→**伪共享灾难**；
- **split（拆分合并）**：各 chunk 并行复制到稀疏区间（记录每块起止迭代器），
  再顺序 `std::move` 压实。无共享写，可扩展。

## 构建与运行

```bash
cmake --build build --target ch11_par_copy_if_example \
    ch11_par_copy_if_benchmark ch11_par_copy_if_tests -j

./build/chapter11_parallel_stl/ch11_par_copy_if_example
./build/chapter11_parallel_stl/ch11_par_copy_if_tests
./build/chapter11_parallel_stl/ch11_par_copy_if_benchmark
```

## 关键点

- benchmark 用轻谓词（is_odd）与重谓词（is_prime）对比：轻谓词下 sync 版
  可能比串行慢（伪共享），split 版略微胜出；重谓词下 split 版显著加速；
- 合并用 `std::move` 避免不必要拷贝；
- 本机实测（GCC 13.3 / i7-13700K，400 万元素，chunk=10 万）：
  - is_odd：split 3.9x，sync **0.09x**（伪共享灾难，比串行慢 11 倍）；
  - is_prime：split 9.9x，sync 10.1x；
  - 书中（i7-7700k）：is_odd split 1.1x / sync 0.07x；is_prime split 5.1x。

## 注意

- sync 版是教学反面教材：正确但伪共享使其几乎总是比串行慢；
- 真实并行 `copy_if` 直接用 C++17 `std::copy_if(execution::par, ...)`。
