# tuple_variadic

tuple、structured bindings、变参模板。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 227-236 页：

- `std::tuple`：静态异构容器（元素类型不同，大小固定）；
- `std::get<Index>`/`std::get<Type>` 访问元素；
- **structured bindings**（C++17）：`auto [a,b,c] = tuple` 优雅解包；
- **tuple_for_each**：用 `if constexpr` 递归展开元组（编译期循环）；
- **变参模板**：`template<typename... Ts>` 参数包，用 `std::tie(values...)`
  打包成 tuple 再迭代。

## 构建与运行

```bash
cmake --build build --target ch08_tuple_example ch08_tuple_tests
./build/chapter08_metaprogramming/ch08_tuple_example
./build/chapter08_metaprogramming/ch08_tuple_tests
```

## 关键点

- `tuple_for_each`/`tuple_any_of` 是编译期展开（无运行期循环）；
- `make_string(42, "hi", true)` 用参数包 + tuple 实现任意参数；
- tests 验证空 tuple、短路 any_of 等边界。
