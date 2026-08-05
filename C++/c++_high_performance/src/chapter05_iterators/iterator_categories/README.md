# iterator_categories

迭代器类别与 traits。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 129-135 页：迭代器按能力分类：

| 类别 | 能力 |
|---|---|
| `input_iterator` | 读 + 前进（`read_step_fwd`） |
| `output_iterator` | 写 + 前进（`write_step_fwd`） |
| `forward_iterator` | input + 多遍读 |
| `bidirectional_iterator` | forward + 后退 |
| `random_access_iterator` | 任意步前进/后退 O(1)（`it += n`、`it[n]`） |
| `contiguous_iterator`（C++20） | random access + 连续内存 |

`std::iterator_traits` 暴露五个关联类型；裸指针自动满足全部接口。

## 构建与运行

```bash
cmake --build build --target ch05_iterator_categories_example \
      ch05_iterator_categories_tests
./build/chapter05_iterators/ch05_iterator_categories_example
./build/chapter05_iterators/ch05_iterator_categories_tests
```

## 输出

```
int* (raw pointer)          category = random_access
vector<int>::iterator       category = random_access
list<int>::iterator         category = bidirectional
map<int,int>::iterator      category = bidirectional
```

## 关键点

- vector/裸指针 = random access；list/map = bidirectional；
- 类别有继承关系（random_access ⊃ bidirectional ⊃ forward），用
  `std::is_base_of` 判断能力；
- 这决定了 `std::sort`（需 random access）可用于 vector 但不能用于 list。
