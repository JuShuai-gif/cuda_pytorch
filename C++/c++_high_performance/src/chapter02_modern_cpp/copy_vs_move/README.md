# copy_vs_move

何时编译器执行移动而非复制。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 65-75 页：

- 复制 = 分配新资源 + 拷贝数据；移动 = **偷走**资源指针（O(1)）；
- 编译器移动对象当且仅当源是 **r-value**：函数返回的临时对象，或 `std::move()` 包裹的具名变量（PDF 73 页）；
- 具名、非常量的变量是 l-value → 复制；const 变量即使 `std::move` 也复制（const 不可移动）；
- 移动后源对象必须保持"有效但未指定"状态。

## 构建与运行

```bash
cmake --build build --target ch02_copy_move_example ch02_copy_move_benchmark ch02_copy_move_tests
./build/chapter02_modern_cpp/ch02_copy_move_example
./build/chapter02_modern_cpp/ch02_copy_move_tests
./build/chapter02_modern_cpp/ch02_copy_move_benchmark
```

## 结果解释（GCC 13.3 Release，4096 double 的 Buffer）

| 操作 | mean | 相对 |
|---|---|---|
| 复制构造（分配+拷贝） | ~490 ns | 343x |
| 移动构造（偷指针） | ~1.4 ns | 1.0x |

示例输出显示的计数器变化：

```
Buffer b = a;         -> copies 增加（a 是 l-value）
Buffer c = move(b);   -> moves 增加（std::move 使 b 成为 r-value）
Buffer d = make();    -> 无复制（NRVO/移动消除临时）
e = a;                -> 复制
e = move(d);          -> 移动
```

## 结论

- 移动语义使"按值返回 + 值语义"零成本（避免 C 风格指针返回的三大缺点，PDF 71 页）；
- 对含堆资源的对象，移动比复制快几个数量级；
- 无资源对象移动 = 复制，没有额外收益。

> 注意：本环境 `Buffer d = make();` 未产生复制（NRVO 直接构造），
> 若 NRVO 不适用则退化为移动，仍无复制。
