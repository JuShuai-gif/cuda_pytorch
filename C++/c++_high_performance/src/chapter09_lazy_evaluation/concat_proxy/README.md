# concat_proxy

用代理对象比较拼接字符串，免去临时字符串分配。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 261-264 页：

- 朴素写法 `(a + b) == c` 会先拼接出一个临时 `std::string`（堆分配 + 拷贝）再比较；
- `operator+` 不拼接，只返回持有两个 `const std::string&` 的 `ConcatProxy`；
- 全局 `operator==(ConcatProxy&&, const String&)` 用 `is_concat_equal`
  直接两段比对：长度相等 + `std::equal(a, c)` + `std::equal(b, c+len(a))`；
- 语法完全不变：`(a + b) == c`。

## 构建与运行

```bash
cmake --build build --target ch09_concat_proxy_example \
    ch09_concat_proxy_tests ch09_concat_proxy_benchmark -j

./build/chapter09_lazy_evaluation/ch09_concat_proxy_example
./build/chapter09_lazy_evaluation/ch09_concat_proxy_tests
./build/chapter09_lazy_evaluation/ch09_concat_proxy_benchmark
```

## 关键点

- `operator==` 只接收 `ConcatProxy&&`（r-value）：把代理存成变量再比较会
  **编译失败**，防止临时字符串析构后引用悬垂（PDF 第 264 页）；
- 要把拼接结果存成字符串：`String c = a + b;` 走
  `operator String() const&&`；`auto` 会得到 `ConcatProxy` 而非 `String`（PDF 第 265 页）；
- benchmark（本机 GCC 13.3 / i7-13700K）：naive/proxy 约 **5.2x**（书中
  10.7x，i7-7700k；差距因硬件/标准库分配器不同）。

## 注意

- 代理持有的是引用，生命周期必须限定在临时表达式内，不要存变量；
- 短字符串时分配器开销占比小，收益随字符串规模增长。
