# iterator_distance

按迭代器类别选择算法。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 135 页：`std::iterator_traits` 暴露迭代器类别。模板函数可用
`if constexpr` 在编译期分派：

- **random_access_iterator**：直接 `b - a`（O(1)）；
- **其他类别**：循环计数（O(n)）。

这解释了 `std::distance` 为何对 vector O(1)、对 list O(n)。

## 构建与运行

```bash
cmake --build build --target ch05_iterator_distance_example
./build/chapter05_iterators/ch05_iterator_distance_example
```

## 输出

```
vector distance: 5
list   distance: 5
pointer distance: 5
```

三种迭代器（vector 迭代器、list 迭代器、裸指针）都能正确求距离，
且编译期选择了各自的实现路径。

## 关键点

- 读取迭代器属性应用 `std::iterator_traits<Iterator>`，而非
  `Iterator::iterator_category`（PDF 135 页明确此约定）；
- `if constexpr` 分支在编译期消除，运行期无分发开销；
- 裸指针天然满足 random_access_iterator 接口。
