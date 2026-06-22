# 第 10 章：并行算法

## 章节概述

C++17 以前，标准库算法（`std::sort`、`std::for_each` 等）都是老老实实一个线程从头跑到尾。你的 CPU 有 16 个核？不好意思，标准库只用 1 个。

C++17 引入**并行算法**——给标准库算法加一个参数，告诉它可以并行执行。你不需要自己切数据、创建线程、同步结果，编译器/库帮你搞定。第 10 章就是讲这套机制的设计理念、使用方法和注意事项。

---

## 10.1 为什么要并行化标准库算法？

**生活类比**：你有一万封信要折好装进信封。一个人折要一整天。你叫来四个朋友，每人分两千五百封，同时开始折。理论上时间变成四分之一。这就是并行化的核心价值。

在编程里，`std::for_each` 就是对每个元素执行某个操作。如果这些操作之间没有依赖关系（折第三封信不需要等第二封信折完），那完全可以**分给多个线程同时做**。

C++ 的想法很务实：大多数程序员已经有 `std::sort`、`std::transform` 的使用习惯，与其让大家重新学一套并行 API，不如**在原有算法上加一个"并行开关"**——加一个参数，行为就变了。学习成本近乎为零。

---

## 10.2 执行策略（Execution Policy）

这是本章最重要的概念。执行策略是一个**标签类型**，告诉标准库"你可以用什么方式执行这个算法"。

### 10.2.1 四大执行策略

```cpp
#include <execution>
#include <algorithm>
#include <vector>

std::vector<int> data(1'000'000);

// 策略一：顺序执行（就是 C++17 之前的默认行为）
std::sort(std::execution::seq, data.begin(), data.end());

// 策略二：并行执行（多线程）
std::sort(std::execution::par, data.begin(), data.end());

// 策略三：并行 + 向量化
std::sort(std::execution::par_unseq, data.begin(), data.end());

// 策略四（C++20）：单线程但向量化
std::sort(std::execution::unseq, data.begin(), data.end());
```

| 策略 | 含义 | 多线程？ | SIMD 向量化？ | C++ 版本 |
|------|------|----------|---------------|----------|
| `seq` | 顺序执行 | ✗ | ✗ | C++17 |
| `par` | 并行执行 | ✓ | ✗ | C++17 |
| `par_unseq` | 并行+向量化 | ✓ | ✓ | C++17 |
| `unseq` | 单线程向量化 | ✗ | ✓ | C++20 |

### 10.2.2 seq —— 顺序执行

`std::execution::seq`：禁止并行，保证按照迭代器顺序逐个元素处理，和 C++14 以前的普通算法行为一致。

**生活类比**：一个人按顺序把全部折完，第一封折完才折第二封。

### 10.2.3 par —— 并行执行

`std::execution::par`：允许多线程同时执行。线程间的执行顺序**不保证**，元素之间的操作可以被调度到不同线程。

**约束**：
- 不能抛出异常（如果抛了，会调用 `std::terminate`）
- 不能有数据竞争（对共享数据的访问必须同步）
- 不能死锁（你写的 lambda 如果等另一个线程，可能永远等不到）
- 迭代器至少是**前向迭代器**（`ForwardIterator`）

**生活类比**：四个人同时折信，每个人拿一叠，互不干扰。

### 10.2.4 par_unseq —— 并行 + 向量化

`std::execution::par_unseq`：在 `par` 的基础上，再加一层：**SIMD 向量化**。

SIMD（Single Instruction Multiple Data）是指在现代 CPU 上，一条指令可以同时对 4/8/16 个数据执行相同操作（如 AVX-512 寄存器一次处理 16 个 int）。

**关键差异**：`par_unseq` 不仅不保证不同线程间的顺序，甚至**不保证同一线程内的元素顺序**。CPU 可能一次同时处理 vec[0]～vec[7]，里面的 8 个元素同时开始、同时结束，不存在先后。

**生活类比**：不只是四个人，而且每个人同时双手操作——左手折一封信的同时右手也在折另一封。

**重要约束**：lambda 中禁止使用 `std::mutex`、`std::condition_variable` 等同步原语。因为 SIMD 操作可能让同一个线程"同时"执行多份 lambda，互斥锁在这种场景下没有意义且会死锁。

### 10.2.5 unseq（C++20）

`std::execution::unseq`：单线程内用 SIMD 向量化，不跨线程。适用场景：操作很轻量但数据量大，不希望承担多线程创建和同步的开销。

### 10.2.6 策略是"建议"，不是"命令"

**这是最容易踩的坑**。标准规定，执行策略是**对实现的建议**——实现如果觉得并行不划算（数据量太小、线程池耗尽等），**可以退化为顺序执行**。

```cpp
// 你写了 par，但实际可能走 seq
std::sort(std::execution::par, data.begin(), data.end());
// 不能假设一定并行！
```

后果：
- 你不能依赖并行行为做正确性假设
- 不能用 thread_local 变量来跨调用传递状态（因为不知道哪些调用在同一线程）
- 测试时并行化发生了，生产环境可能没有

---

## 10.3 支持的并行算法

C++17 给 69 个标准库算法加入了执行策略重载（C++20 扩展到更多）。下面按类别介绍重点。

### 10.3.1 排序类

```cpp
std::sort(std::execution::par, v.begin(), v.end());
std::stable_sort(std::execution::par, v.begin(), v.end());
std::partial_sort(std::execution::par, v.begin(), v.begin() + 10, v.end());
```

对于大数据量排序，并行版本的实现通常采用**并行归并排序**或**并行快速排序**的思路：分治——把数据切分成多个块，各线程分别排序，最后合并。

**限制**：要求**随机访问迭代器**（`RandomAccessIterator`）。

### 10.3.2 for_each

```cpp
std::for_each(std::execution::par, v.begin(), v.end(),
    [](int& x) { x *= 2; });
```

最简单的并行化场景。每个元素的处理独立，天然并行友好。

### 10.3.3 transform 和 reduce

```cpp
// transform：对每个元素做变换，输出到另一个容器
std::vector<int> out(data.size());
std::transform(std::execution::par,
    data.begin(), data.end(), out.begin(),
    [](int x) { return x * x; });

// reduce：归约操作（类似 accumulate 但可以并行）
int sum = std::reduce(std::execution::par,
    data.begin(), data.end(), 0);  // 初始值 0

// transform_reduce：先 transform 再 reduce（一步到位，不存中间结果）
int sum_of_squares = std::transform_reduce(std::execution::par,
    data.begin(), data.end(), 0,
    std::plus<>{},          // reduce 用的 combine
    [](int x) { return x * x; });  // transform 操作
```

**`reduce` 和 `accumulate` 的区别**：
- `std::accumulate`：严格从左到右（顺序保证，不能并行）
- `std::reduce`：不保证顺序（可并行，要求操作可结合、可交换）

```cpp
// accumulate: ((1+2)+3)+4 = 10  严格顺序
// reduce:     (1+2)+(3+4) = 10  也可能 (1+4)+(2+3)=10  只要是结合可交换的
```

### 10.3.4 查找类

```cpp
auto it = std::find(std::execution::par, v.begin(), v.end(), 42);
auto it2 = std::find_if(std::execution::par, v.begin(), v.end(),
    [](int x) { return x > 100; });
```

并行 find 的行为：多个线程从不同位置开始搜索，**谁先找到算谁的**。这意味着即使有多个匹配元素，返回的也是"某个"而不是"第一个"。

如果需要第一个，用 `std::find`（不带 execution policy）或 `seq`。

### 10.3.5 复制和移动

```cpp
std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
std::copy_if(std::execution::par, src.begin(), src.end(), dst.begin(), pred);
```

注意 `copy_if` 的目标迭代器不能和源迭代器指向同一容器。

### 10.3.6 其他支持的算法（部分列表）

- `std::count_if`、`std::count`
- `std::any_of`、`std::all_of`、`std::none_of`
- `std::equal`
- `std::replace`、`std::replace_if`
- `std::reverse`、`std::rotate`
- `std::unique`（`par_unseq` 不支持）
- C++20 新增：`std::lexicographical_compare`、`std::shift_left/right`

---

## 10.4 使用限制和注意事项

### 10.4.1 异常处理

**用 `par` 或 `par_unseq` 时，lambda 抛异常 → `std::terminate()` → 程序直接死。**

背后逻辑：多个线程同时跑，如果线程 A 在元素 3 上抛异常，线程 B 正在处理元素 4，怎么协调？抛出多个异常怎么办？标准委员会选择了简单粗暴的方案——别抛。

**解决方案**：lambda 内部用 `try-catch` 兜住，把异常存起来，等算法结束后统一处理。

### 10.4.2 避免死锁

并行算法的迭代函数**不应该**去获取全局互斥锁，尤其在 `par_unseq` 下。SIMD 执行可能导致同一线程试图多次获取同一个锁，那必死锁。

### 10.4.3 数据竞争

这是"老问题但在新场景"：现在你的 lambda 在多个线程同时跑，如果你在 lambda 里写入共享变量（哪怕是 `int`），那就是数据竞争 → UB。

```cpp
int shared = 0;
// 危险！
std::for_each(std::execution::par, v.begin(), v.end(),
    [&shared](int x) { shared += x; });  // 数据竞争！

// 正确做法
int sum = std::reduce(std::execution::par, v.begin(), v.end(), 0);
```

### 10.4.4 迭代器要求

多数并行算法要求至少**前向迭代器**（`ForwardIterator`）。输入迭代器（`InputIterator`，如 `std::istream_iterator`）不能并行——你不能让多个线程同时从一个网络流里读数据。

### 10.4.5 元素的相对顺序

并行算法对**非稳定**操作不保证元素间的顺序。如果你需要保留等价元素的相对顺序，用 `stable_*` 版本，但要接受性能损失。

---

## 工业场景

| 场景 | 对应技术 |
|------|----------|
| **大规模数据排序** | `std::sort(par)` 对 TB 级日志数据按时间戳排序，比串行快 4～8 倍 |
| **并行变换** | 图像处理中对每个像素做颜色校正，`std::transform(par)` 处理百万像素 |
| **并行归约** | 金融风控中对千万条交易记录求和、求均值，`std::reduce(par, …)` |
| **数据清洗管道** | `std::copy_if(par)` + `std::replace(par)` 组合处理 ETL 任务 |

---

## 常见坑点

1. **以为 `par` 一定并行 —— 错了**。小数据量实现会 fallback 到 `seq`。基准测试时要注意数据规模。
2. **`par_unseq` 里用了互斥锁** → 死锁或 UB。SIMD 执行模型下锁毫无意义。
3. **`std::find(par)` 不保证返回第一个匹配**。因为多线程同时搜索，谁先找到取决于调度。如果你要第一个，别并行。
4. **`reduce` 使用不可结合的运算**。`reduce` 要求操作满足结合律 `(a+b)+c == a+(b+c)`。浮点加法严格来说不满足（因为 rounding），所以 `reduce` 的结果和 `accumulate` 可能**位数不同**。
5. **忘记链接 TBB 库**。GCC 的并行算法实现依赖 Intel TBB（Threading Building Blocks）库。如果编译时报 `-ltbb` 找不到，需要装 `libtbb-dev`（Ubuntu）或等效包。某些编译器（MSVC）不需要额外库。

---

## 面试常问

**Q：C++17 支持哪些 execution policy？**
- `seq`（顺序）、`par`（并行，多线程）、`par_unseq`（并行+SIMD 向量化）、C++20 还有 `unseq`（单线程 SIMD）

**Q：`std::reduce` 和 `std::accumulate` 的区别？**
- `accumulate` 保证严格从左到右累加，无法并行
- `reduce` 不保证顺序，可并行化，要求运算满足结合律和交换律
- 浮点运算下两者结果可能有微小差异

**Q：`par_unseq` 下有什么特殊限制？**
- 不能用任何同步原语（mutex、condition_variable、atomic 的 memory_order 等）
- 不能抛异常
- 元素间完全无顺序保证（同一线程内也无顺序）

**Q：并行 sort 内部原理是什么？**
- 一般采用并行样本排序（Parallel Sample Sort）或并行归并排序
- 大致流程：采样→分桶→各线程独立排序桶内元素→合并桶边界
- 关键是让每个线程处理的数据量大致相等且桶间已有序

---

## 我应该掌握什么

- [ ] 四种 execution policy 的名称和含义
- [ ] `par` 和 `par_unseq` 的核心区别
- [ ] execution policy 是"建议"这个事实及其影响
- [ ] 哪些算法支持并行，如何加 execution policy 参数
- [ ] `reduce` vs `accumulate` 的区别
- [ ] `find(par)` 为什么不保证返回第一个匹配
- [ ] `par_unseq` 下禁止使用互斥锁的原因
- [ ] 并行算法中 lambda 抛异常会发生什么
- [ ] 常见的编译器/TBB 依赖问题
- [ ] 能写出 `transform_reduce` 的实用示例
