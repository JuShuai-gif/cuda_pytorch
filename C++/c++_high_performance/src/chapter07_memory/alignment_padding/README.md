# alignment_padding

内存对齐与 padding。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 186-189 页：

- 每个类型有对齐要求（`alignof`），对象须位于对齐地址的倍数；
- 编译器在成员间插入 padding 以满足对齐；
- **重排成员（大的在前）可缩小结构体**：`DocumentV1`（bool,double,int）
  24 字节 → `DocumentV2`（double,int,bool）16 字节；
- `new`/`malloc` 返回的内存满足 `alignof(std::max_align_t)` 对齐；
- `alignas` 可提高对齐（如对齐到缓存行 64 字节）。

## 构建与运行

```bash
cmake --build build --target ch07_alignment_example ch07_alignment_tests
./build/chapter07_memory/ch07_alignment_example
./build/chapter07_memory/ch07_alignment_tests
```

## 结果（x86-64）

```
alignof(max_align_t) = 16
sizeof(DocumentV1) = 24   (bool first)
sizeof(DocumentV2) = 16   (double first)
alignof(CacheAligned) = 64
```

## 关键点

- 成员重排是"免费"的内存优化（尤其对象数组）；
- 对齐到缓存行可减少对象跨行（性能收益需实测）；
- tests 用 `offsetof`/`% alignof` 验证对齐不变量。
