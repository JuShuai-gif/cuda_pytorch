# int_iterator

自定义迭代器：生成整数。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 133-136 页：迭代器不一定要指向真实数据，可以**即时生成**值。
`IntIterator` 只存一个 `int`，`operator*` 返回该值、`operator++` 递增。
它满足 forward_iterator（和 input_iterator）类别，因此可用于 range-for
与 `std::copy` 等算法。

`BidirectionalIntIterator` 增加 `operator--` 与 `bidirectional_iterator_tag`，
可反向迭代（PDF 136 页）。

## 构建与运行

```bash
cmake --build build --target ch05_int_iterator_example ch05_int_iterator_tests
./build/chapter05_iterators/ch05_int_iterator_example
./build/chapter05_iterators/ch05_int_iterator_tests
```

## 关键点

- 自定义迭代器只要实现指针式语法（`*`、`++`、`!=`）就可在算法中使用；
- 声明五个关联类型（`difference_type/value_type/reference/pointer/iterator_category`）
  才能让 `std::iterator_traits` 正常工作（PDF 134 页）；
- `std::copy(IntIterator(5), IntIterator(12), back_inserter)` 生成 5..11；
- 反向迭代必须用 `--` 循环（`std::copy` 只能正向，测试早期版本因此死循环）。

## 注意

- 早期 `tests.cpp` 用 `std::copy(BidirectionalIntIterator{5}, {0}, ...)`
  导致无限生成（5,6,7,...）→ `bad_alloc`。正向 copy 无法"向下"到达 0，
  必须用 `--` 循环。
