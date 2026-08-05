# const_correctness

const 正确性。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 34-35 页：const 正确性把"是否修改对象"写进函数签名。

- `int age() const` 承诺不修改对象；
- `const Person& leader() const` 与 `Person& leader()` 构成重载，
  通过 const 对象只能拿到不可变引用；
- 试图在 const 对象上调用非 const 成员 → **编译期错误**，而非运行时 bug。

## 文件

| 文件 | 说明 |
|---|---|
| `example.cpp` | Person/Team 的 const 成员函数与 const/mutable 重载 |
| `tests.cpp` | 运行期行为 + `static_assert` 验证重载返回类型 |
| `compile_error_example.cpp` | **预期编译失败**的文件，不加入构建 |

## 构建与运行

```bash
cmake --build build --target ch01_cc_example ch01_cc_tests
./build/chapter01_zero_cost/ch01_cc_example
./build/chapter01_zero_cost/ch01_cc_tests
```

演示预期编译失败（应报 discards qualifiers）：

```bash
g++ -std=c++17 -fsyntax-only src/chapter01_zero_cost/const_correctness/compile_error_example.cpp
```

## 观察点

- 两个 `leader()` 重载只返回类型不同：`const` 重载返回 `const Person&`，
  非 const 重载返回 `Person&`；
- const 正确性通常是纯编译期概念，`-O2` 下不产生任何运行时代价。
