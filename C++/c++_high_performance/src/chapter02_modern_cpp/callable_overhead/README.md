# callable_overhead

Lambda / 仿函数 / `std::function` 的调用开销对比。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 56-59 页：`std::function` 是**类型擦除**容器，可持有任意同签名可调用对象，
代价是：

1. **不能内联**：编译器不知道实际类型，只能间接调用；
2. **堆分配捕获变量**：捕获对象超过 SBO 时分配堆内存；
3. **调用需要更多操作**：解包、跳转。

PDF 第 60 页给出书中的基准：100 万元素循环调用，直接 lambda 约 **四分之一** 的
`std::function` 时间。

## 构建与运行

```bash
cmake --build build --target ch02_callback_benchmark ch02_callback_tests
./build/chapter02_modern_cpp/ch02_callback_tests
./build/chapter02_modern_cpp/ch02_callback_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，100 万元素）

| 实现 | mean | 相对 |
|---|---|---|
| `vector` of lambda（直接） | ~500 µs | 1.0x |
| `vector` of 仿函数 | ~575 µs | 1.1x |
| `vector` of `std::function` | ~1500 µs | **~3.0x** |

- lambda 与仿函数都内联，几乎无差异；
- `std::function` 约慢 3 倍（间接调用阻止内联），与书中"约四分之一"一致。
- 注意：早期版本把 `res = f(res)` 的纯函数循环被编译器常量折叠（~4ns），
  本版加入 `res ^= i` 防止折叠，保证四者做相同的真实工作（公平对比）。

## 结论（限定本环境）

- 类型擦除有真实成本：热路径上避免 `std::function`，优先模板或直接 lambda；
- 需要运行时多态回调时，`std::function` 的灵活性值得这 3 倍开销，但要用 `-O2`+ 实测验证。
