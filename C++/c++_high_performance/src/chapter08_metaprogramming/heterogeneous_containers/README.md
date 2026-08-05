# heterogeneous_containers

`vector<any>` vs `vector<variant>`。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 236-242 页：

- **`std::vector<std::any>`**：可存任何类型，但访问时须运行期类型检查，
  编译期类型信息完全丢失；
- **`std::vector<std::variant<...>>`**：固定类型集，元素存栈上（无每元素
  堆分配），可用 `std::visit` + 多态 lambda 优雅访问；
- variant 大小 = 最大备选类型（PDF 240 页）。

## 构建与运行

```bash
cmake --build build --target ch08_heterogeneous_example ch08_heterogeneous_tests
./build/chapter08_metaprogramming/ch08_heterogeneous_example
./build/chapter08_metaprogramming/ch08_heterogeneous_tests
```

## 关键点

- `std::visit` 编译期为每个备选类型生成 lambda 实例；
- `holds_alternative`/`std::get<Type>` 组合做类型+值查找；
- `sizeof(any)=16` vs `sizeof(variant<int,string,bool>)=40`（本环境 libstdc++）。

## 结论

- 需要任意类型 → `any`（代价：运行期检查 + 可能堆分配）；
- 类型集已知 → `variant`（栈存储、编译期安全、visit 优雅）。
