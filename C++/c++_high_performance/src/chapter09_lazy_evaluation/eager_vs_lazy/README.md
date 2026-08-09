# eager_vs_lazy

AudioLibrary 惰性 vs 急切求值对比。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 259 页：

- `get_eager(id, otherwise)`：`otherwise` 是已构造好的对象，**无论查找是否命中**都先付构造成本；
- `get_lazy(id, fn)`：`otherwise` 是一个函数（工厂），**仅当 id 未命中**时才调用它构造对象。

## 构建与运行

```bash
cmake --build build --target ch09_eager_vs_lazy_example ch09_eager_vs_lazy_tests -j
./build/chapter09_lazy_evaluation/ch09_eager_vs_lazy_example
./build/chapter09_lazy_evaluation/ch09_eager_vs_lazy_tests
```

## 关键点

- `get_lazy` 用模板接收任意工厂（函数对象/lambda），比 `std::function` 更内联友好；
- 命中路径下惰性版本构造次数为 0（tests.cpp 用全局计数验证）；
- 代码写法几乎一样，只是把"值"换成"取值的函数"。

## 注意

- 惰性求值的收益来自"**结果可能用不上**"的场景；如果结果必然要用，两者成本相同。
