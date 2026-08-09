# object_pool

综合实践：固定大小对象池。

> 综合 Chapter 1（RAII）+ Chapter 7（自定义内存管理）的教学实现，非原书代码。

## 功能

- 构造时一次性分配 `capacity` 个 `block_size` 字节块，对齐到 `max_align_t`；
- 空闲链表：`allocate()` / `deallocate()` 均 O(1)、不触碰系统分配器；
- `Pooled<T>` RAII 包装：构造时在池内 placement-new，析构时返回块；
- 池耗尽：`allocate()` 返回 `nullptr`，`Pooled` 抛 `std::bad_alloc`。

## 构建与运行

```bash
cmake -S projects -B build-projects
cmake --build build-projects --target object_pool_example \
    object_pool_tests object_pool_benchmark -j

./build-projects/object_pool/object_pool_example
./build-projects/object_pool/object_pool_tests
./build-projects/object_pool/object_pool_benchmark
```

## 关键验证

- tests：容量/耗尽/LIFO 复用/对齐/RAII 构造析构计数/bad_alloc；
- benchmark（本机 GCC 13.3 / i7-13700K）：pool ≈ 0.5 ns/op vs `new/delete`
  ≈ 12 ns/op（约 **25x**；`asm volatile` barrier 防编译器消除测量）；
- 典型应用：粒子系统、高频小对象（分配次数可统计，见 Ch7）。

## 注意

- 只能分配**同尺寸**对象；异尺寸需多池或改用 arena；
- 块内首字节在空闲时存下一块指针，正在使用时无此约束；
- 线程安全需加锁或每线程池（此处未做，属 Ch10 范畴）。
