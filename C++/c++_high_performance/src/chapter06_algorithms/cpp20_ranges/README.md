# cpp20_ranges

C++20 `std::ranges`：书中 ranges 库的现代标准实现。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

> 现代补充：本模块使用 C++20 `std::ranges`（书中 PDF 163-173 页描述的是
> 当时尚未入标准的 range-v3 库，现已以 `std::ranges` 进入 C++20）。
> 概念一一对应：`std::views`（view）、range 算法、`|` 管道运算符。

书中要点（PDF 163-173 页）：
- **STL 迭代器缺乏组合性**：如"找出弓手最高等级"需 copy_if 到新容器再
  max_element（浪费复制）；
- **views**：惰性求值，不复制数据，访问时才计算；
- **管道**：`numbers | transform(...) | filter(...)` 从左到右读；
- **actions vs views**：action 修改容器，view 只提供视图。

## 构建与运行

需要 C++20（GCC 10+/Clang 10+）与 `-DENABLE_CPP20_EXAMPLES=ON`：

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CPP20_EXAMPLES=ON
cmake --build build -j
./build/chapter06_algorithms/ch06_cpp20_ranges_example
./build/chapter06_algorithms/ch06_cpp20_ranges_tests
```

## 输出

```
max archer level (ranges): 10        <- 书中 Warrior 例子
odd squares: 1 9 25 49               <- 书中 transform|filter 例子
ranges::count(7): 1
joined: 1 2 3 4 5 5 4 3 2 1          <- 书中 join 例子
```

## 关键点

- views 惰性：`archer_levels` 不创建中间 vector；
- 管道从左到右可读；
- `std::ranges::max_element` 直接作用在 view 上；
- 与 C++17 主项目分离（CMake 选项控制）。

## 注意

- view 引用源数据：源析构后 view 悬垂；
- 惰性求值意味着每次访问都重算变换（热循环中注意）。
