# boost_compute

用 Boost.Compute + OpenCL 把 STL 算法搬到 GPU。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。
> 仅在有 OpenCL 运行时 + Boost.Compute 的环境编译（可选模块）。

## 原理

PDF 第 343-352 页：

- `device`（GPU）/ `context` / `queue` 初始化；
- 数据必须进 `bc::vector`（GPU 内存），算完 `bc::copy` 回 CPU；
- 自定义 struct 用 `BOOST_COMPUTE_ADAPT_STRUCT` 适配（成员无 padding）；
- 函数用 `BOOST_COMPUTE_FUNCTION` 宏（OpenCL C99 语法，运行时编译）；
- `bc::transform` / `bc::reduce` / `bc::sort` / `bc::iota` / `bc::fill`
  与 STL 同名 API；
- 自定义 kernel：`program::create_with_source` + `build` + `kernel` +
  `enqueue_nd_range_kernel`（二维并行，box filter）。

## 构建与运行

```bash
cmake -DENABLE_BOOST_COMPUTE=ON -DENABLE_OPENCL=ON -S src -B build
cmake --build build --target ch11_boost_compute_example -j
./build/chapter11_parallel_stl/ch11_boost_compute_example
```

## 关键点

- 本机已验证：OpenCL 平台 + NVIDIA GeForce RTX 4070 可用；
- 圆面积 transform-reduce：GPU 结果与 CPU 相比差 <0.1%；
- GPU 排序结果用 CPU 谓词 `std::is_sorted` 验证；
- box filter 结果用 epsilon 容差与 CPU 参考比对（浮点，勿用 `==`）。

## 注意

- GPU 常受**数据往返拷贝**瓶颈，数据量小无优势；
- 自定义 struct 成员必须对齐（无 padding），否则 GPU 读取错位；
- OpenCL 头文件在本机位于 CUDA 目录，CMake 已自动探测。
