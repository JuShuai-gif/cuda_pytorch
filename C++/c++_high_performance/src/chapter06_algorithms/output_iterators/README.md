# output_iterators

输出迭代器：算法结果的去向。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 148-149 页：带输出的算法（`std::copy`/`std::transform`）需要**已分配**
的目标空间。三种方案：

1. `resize()` 预分配；
2. 插入迭代器：`std::back_inserter`（vector）、`std::inserter`（set）；
3. 已知大小时 `reserve()` + `back_inserter` 避免重复扩容。

注意：把空容器的 `begin()` 直接当输出目标 → UB/崩溃。

## 构建与运行

```bash
cmake --build build --target ch06_output_iterators_example ch06_output_iterators_tests
./build/chapter06_algorithms/ch06_output_iterators_example
./build/chapter06_algorithms/ch06_output_iterators_tests
```

## 关键点

- 四种方式产生相同结果（tests 验证）；
- `reserve` 后 `capacity()==4`（无重复扩容）。
