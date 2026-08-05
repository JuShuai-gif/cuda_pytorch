# value_semantics

值语义 vs 引用/共享语义。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 32-33 页的 Bagel 示例：C++ 默认**值语义**——构造时复制对象，
实例与来源完全隔离；而 Java 等语言默认共享引用，修改透过所有引用可见。

C++ 通过 `std::shared_ptr`/`std::weak_ptr` 显式表达"共享所有权"，
把意图写进类型系统；Java 无法区分"独占""共享""临时持有"。

## 文件

| 文件 | 说明 |
|---|---|
| `example.cpp` | Bagel 值语义 vs 共享语义演示 + 值/引用传参对比 |
| `tests.cpp` | 隔离性与共享性断言 |

## 构建与运行

```bash
cmake --build build --target ch01_vs_example ch01_vs_tests
./build/chapter01_zero_cost/ch01_vs_example
./build/chapter01_zero_cost/ch01_vs_tests
```

## 输出解读

- 值语义 bagel 'a' 只有 `salt`；即使之后源集合加入 pepper/oregano 也不受影响。
- 共享 bagel 'c'、'd' 都看到全部调料——这正是 Java 的默认行为（容易产生隐蔽 bug）。
- `change_value`（传值）不影响原集合；`change_reference`（传引用）会修改。

## 性能关联

值语义需要复制：对象大或复制昂贵时应避免无谓拷贝（用引用/移动，见 Chapter 2）。
值语义的隔离性使数据本地化（见 `contiguous_vs_pointer`），但也增加拷贝成本——权衡见
Chapter 4/7。
