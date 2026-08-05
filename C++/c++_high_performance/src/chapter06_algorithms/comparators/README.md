# comparators

自定义比较器与谓词。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 149-151 页：

- 算法默认用 `operator==` 与 `operator<`；
- 需要其他比较时传自定义比较器（`std::sort(names, less_by_size)`）；
- 谓词版本加 `_if` 后缀：`find_if`、`count_if`；
- 建议建一个 `preds` 命名空间放通用谓词，提高可读性。

书中 `equal_case_insensitive` 例子演示"返回 lambda 的 lambda"（谓词工厂）。

## 构建与运行

```bash
cmake --build build --target ch06_comparators_example ch06_comparators_tests
./build/chapter06_algorithms/ch06_comparators_example
./build/chapter06_algorithms/ch06_comparators_tests
```

## 关键点

- 按长度排序、按长度 find、大小写不敏感 count 均正确；
- `equal_by_size(3)` 返回一个谓词（工厂模式）；
- 与 Chapter 2 lambda 捕获知识衔接。
