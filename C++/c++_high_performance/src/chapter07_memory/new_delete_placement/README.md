# new_delete_placement

`new`/`delete`、placement new、`<memory>` 工具。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 182-185 页：

- `new` = 分配内存 + 构造；`delete` = 析构 + 释放；
- **placement new** 分离分配与构造：在已分配内存上构造对象；
- 没有 placement delete：须显式调用析构函数再释放；
- C++17 `<memory>`：`std::uninitialized_fill_n`/`std::destroy_at` 替代
  placement new 与显式析构。

## 构建与运行

```bash
cmake --build build --target ch07_new_delete_example ch07_new_delete_tests
./build/chapter07_memory/ch07_new_delete_example
./build/chapter07_memory/ch07_new_delete_tests
```

## 关键点

- 显式调用析构函数只在 placement new 之后合法（PDF 183 页明确警告）；
- `new[]`/`delete[]` 必须配对（数组分配带长度前缀）；
- tests 用构造/析构计数验证生命周期平衡。
