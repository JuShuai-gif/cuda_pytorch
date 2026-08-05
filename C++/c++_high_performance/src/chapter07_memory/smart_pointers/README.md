# smart_pointers

智能指针与 `make_shared` 分配差异。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 190-196 页：

- **`unique_ptr`**：独占所有权，可转移（move）不可复制，**零开销**（sizeof == 指针）；
- **`shared_ptr`**：引用计数共享所有权，计数原子更新（线程安全）；控制块有内存开销；
- **`weak_ptr`**：不保持对象存活，`lock()` 尝试提升为 shared_ptr（防悬垂）；
- **`make_shared` 一次分配 vs `shared_ptr(new T)` 两次分配**（对象+控制块分别分配）。

## 构建与运行

```bash
cmake --build build --target ch07_smart_pointers_example ch07_smart_pointers_tests
./build/chapter07_memory/ch07_smart_pointers_example
./build/chapter07_memory/ch07_smart_pointers_tests
```

## 结果（libstdc++）

```
unique_ptr: owner=null new_owner=42
make_shared: allocs=1 frees=0          <- 一次分配
shared_ptr(new T): allocs=2 frees=0    <- 两次分配
weak_ptr while alive: 10
weak_ptr expired after owner destroyed
```

## 关键点

- `make_shared` 更高效（一次分配 + 空间局部性），且异常安全
  （`shared_ptr(new T)` 在构造 control block 时抛异常会泄漏 T）；
- `weak_ptr.lock()` 避免悬垂指针；
- 测试用 `sizeof`/`use_count`/`expired` 验证语义。

## 注意

- 引用计数无法处理循环引用 → 用 `weak_ptr` 打破环。
