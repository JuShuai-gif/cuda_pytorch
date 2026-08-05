# safe_cast

带检查的泛型类型转换。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 249-251 页：

- 手写 cast 有多种出错方式（精度丢失、负转无符号、指针转错整数、double 溢出等）；
- `safe_cast<Dst>(v)` 用 `if constexpr` 按转换种类分派：
  - 同类型：恒等；
  - 指针↔uintptr_t：`reinterpret_cast`；
  - 指针↔指针：Debug 用 `dynamic_cast` 验证；
  - 浮点↔浮点：回转检查 NaN/Inf；
  - 算术↔算术：回转检查精度；
  - 其他：`static_assert(make_false<T>())` **编译失败**。

## 构建与运行

```bash
cmake --build build --target ch08_safe_cast_example
./build/chapter08_metaprogramming/ch08_safe_cast_example
```

## 关键点

- `make_false<T>()` 延迟断言到实例化时刻（直接 `static_assert(false)`
  会让函数永远无法编译）；
- 未支持转换（如指针→普通 int）编译失败，运行期零检查（Release 下
  assert 被 `NDEBUG` 关闭）；
- `if constexpr` 使未选分支被消除，不会产生非法代码。

## 注意

- 本项目仅保留指针↔uintptr_t、指针↔指针、浮点↔浮点、算术↔算术，
  与书中一致；`dynamic_cast` 分支在本例未触发。
