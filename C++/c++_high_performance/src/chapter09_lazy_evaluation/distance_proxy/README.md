# distance_proxy

用代理对象延迟 `std::sqrt()`，比较两点距离时只比平方距离。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 265-273 页：

- `Point::distance()` 返回持有**平方距离**的 `DistProxy`，而不是 `float`；
- 比较运算符只比较平方距离：
  - `a.distance(x) < b.distance(x)`：直接比 `dist_sqrd_`；
  - `a.distance(b) < threshold`：比 `dist_sqrd_ < threshold * threshold`；
- 只有真正需要距离值时，才通过 `operator float() const&&` 调用 `std::sqrt`；
- 平方距离不暴露给用户，防止"距离"与"距离平方"混用（PDF 第 268 页）。

## 构建与运行

```bash
cmake --build build --target ch09_distance_proxy_example \
    ch09_distance_proxy_tests ch09_distance_proxy_benchmark -j

./build/chapter09_lazy_evaluation/ch09_distance_proxy_example
./build/chapter09_lazy_evaluation/ch09_distance_proxy_tests
./build/chapter09_lazy_evaluation/ch09_distance_proxy_benchmark
```

## 关键点

- `operator float() const&&` 只允许在**临时对象**上转换：把 `distance()` 结果
  存成变量后再转 float 会**编译失败**（PDF 第 272 页），避免同一代理
  多次触发 sqrt，也避免把代理当普通值传来传去；
- 用户语法不变：`a.distance(bingo) < b.distance(bingo)` 与朴素写法一致；
- benchmark（本机 GCC 13.3 / i7-13700K）：naive/proxy 约 **1.14x**（书中
  2x，i7-7700k）。现代 x86 的 `sqrtss` 是单条指令，省掉它的收益远小于
  2017 年硬件；在更老的 CPU 或软件 `sqrt` 库上收益更大。

## 注意

- `auto dist = a.distance(b);` 得到的是 `DistProxy` 而非 `float`；
  要数值必须 `float dist = a.distance(b);`（临时对象直接转换）；
- 用平方距离比较还顺带提升了浮点精度（`sqrt` 会丢失低位精度）。
