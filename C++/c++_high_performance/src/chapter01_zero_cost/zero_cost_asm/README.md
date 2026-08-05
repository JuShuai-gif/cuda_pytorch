# zero_cost_asm

零成本抽象的汇编验证。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 25 页："零成本抽象"意味着抽象不增加运行时成本——即生成与手写版本
相同的机器码。判断标准不是"STL 一定更快"，而是**汇编是否接近、抽象是否
产生额外运行时成本、编译器是否能够内联**。

## 文件

| 文件 | 说明 |
|---|---|
| `example.cpp` | 4 个 `noinline` 函数供汇编分析 |
| `tests.cpp` | 对应行为等价性 |

## 汇编生成

```bash
g++ -std=c++17 -O3 -S src/chapter01_zero_cost/zero_cost_asm/example.cpp
clang++ -std=c++17 -O3 -S src/chapter01_zero_cost/zero_cost_asm/example.cpp
```

## 观察结果（GCC 13.3 -O3，本环境实测）

`count_loop`（手写）与 `count_algo`（`std::count`）的内层循环**几乎相同**：

```
.L39:                                    .L20:
  movdqu (%rax), %xmm0                     movdqu (%rax), %xmm0
  pcmpeqd %xmm6, %xmm0                     pcmpeqd %xmm5, %xmm0
  ...punpck/psubq 累加计数...
  paddq   %xmm1, %xmm2                     paddq   %xmm1, %xmm2
```

两者都：

- **内联**：`std::count` 无函数调用指令（`call`），直接内联；
- **自动向量化**：SSE2 `movdqu` 一次处理 4 个 `int`，`pcmpeqd` 比较、`paddq` 累加；
- 汇编结构差异仅来自循环边界处理细节，核心计算一致。

## 结论（限定本环境）

- 对 `count` 这类简单算法，`std::count` 是**零成本抽象**：汇编与手写循环等价；
- 若把函数放在头文件并标记内联，调用点也会被完全消除。

## 检查点

- 是否内联（无 `call` 指令）；
- 是否函数间接调用（无 `call *reg`）；
- 是否出现 `new`/`delete`；
- 是否出现 `memcpy`；
- 是否展开循环 / 自动向量化（`pcmpeqd`、`movdqu`、`add $0x10` 等）。
