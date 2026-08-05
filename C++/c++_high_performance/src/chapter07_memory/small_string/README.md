# small_string

Small String Optimization（SSO）。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 196-198 页：`std::string` 用 union 提供两种布局——短字符串用内联
缓冲（不分配堆），长字符串用堆指针。libc++ 在 24 字节的 string 里塞进
22 字符的内联缓冲；libstdc++（本项目）SSO 边界是 15 字符。

## 构建与运行

```bash
cmake --build build --target ch07_small_string_example
./build/chapter07_memory/ch07_small_string_example
```

## 结果（libstdc++，GCC 13，x86-64）

```
sizeof(std::string) = 32 bytes
len 0..15: capacity=15 allocs=0   <- SSO：无堆分配
len 16+:   capacity>=16 allocs=1  <- 触发堆分配
```

## 关键点

- SSO 避免短字符串（最常见场景）的堆分配；
- 边界因标准库而异（libc++ 22、libstdc++ 15）；
- 通过覆盖全局 `operator new` 计数验证。

## 结论

- 短字符串大量创建时 SSO 显著减少分配；
- 长字符串反复拼接/复制仍需注意分配（见 Chapter 9 惰性求值）。
