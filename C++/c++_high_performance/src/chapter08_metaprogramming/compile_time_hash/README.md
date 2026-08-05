# compile_time_hash

PrehashedString：编译期字符串哈希。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 252-257 页：

- 资源缓存用 `unordered_map<string,...>`，每次查找都运行期计算哈希；
- **PrehashedString**：构造时（编译期）计算哈希，查找零运行期成本；
- 构造函数 `template<size_t N> PrehashedString(const char(&)[N])` 强制只接受
  **编译期字符串字面量**（保证 `strptr_` 生命周期安全）；
- `std::hash<PrehashedString>` 特化让 STL 容器直接使用预计算哈希。

书中验证：`test_prehashed_string()` 编译为 `mov eax, 294; ret`。

## 构建与运行

```bash
cmake --build build --target ch08_compile_time_hash_example
./build/chapter08_metaprogramming/ch08_compile_time_hash_example

# 查看汇编（应看到 mov $294 常量）
g++ -std=c++17 -O3 -S src/chapter08_metaprogramming/compile_time_hash/example.cpp
```

## 汇编验证（本环境 GCC 13.3 -O3）

```
movl $294, %edx     <- hash_function("abc") 在编译期算好
```

## 关键点

- 编译期哈希是"微优化"（书中承认小字符串无感），但演示了把计算从运行期
  移到编译期的方法论，弱硬件场景有意义；
- 求和哈希仅教学用（真实应用用 `boost::hash_combine`，见 Chapter 4）；
- `PrehashedString` 不拥有字符串 → 必须字面量（编译器保证）。

## 注意

- 哈希计算只对**编译期字面量**生效；运行期字符串仍走运行期哈希。
