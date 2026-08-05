# capture_size

捕获大小与 lambda/`std::function` 对象尺寸的关系。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 51-52 页：lambda 的捕获块等价于类成员变量，因此 **lambda 对象的大小
等于其所有捕获对象大小之和**（无捕获 = 1 字节空对象）。这直接决定了存进
`std::function` 时是否触发堆分配（见 `std_function_cost`）。

## 构建与运行

```bash
cmake --build build --target ch02_capture_size_example
./build/chapter02_modern_cpp/ch02_capture_size_example
```

## 关键点

| lambda | `sizeof` | 说明 |
|---|---|---|
| 无捕获 | 1 字节 | 空类型优化（EBO） |
| 捕获 1 个 `int` | 4 字节 | = int |
| 捕获 `int`+`long` | 16 字节 | 对齐后 |
| 捕获 64 字节对象 | 64 字节 | = 捕获对象大小 |
| 引用捕获 | 8 字节 | = 指针大小 |

## 结论

- 捕获对象越大，lambda 越大；
- 超过 `std::function` SBO（libstdc++ 16 字节）即触发堆分配；
- 大捕获对象可用引用捕获（8 字节）避免复制进 lambda，但需注意生命周期。
