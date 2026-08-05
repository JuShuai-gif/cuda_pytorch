# latency_throughput

延迟 vs 吞吐量。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 96 页定义：

- **Latency/response time**：一次请求到响应的耗时（如转换一张图片的时间）；
- **Throughput**：单位时间处理的交易数（如每秒转换的图片数）。

本实验对比两种典型负载：
- **串行依赖链**：每步依赖前一步结果（LCG 伪随机），CPU 无法重叠，
  受限于单次操作的延迟；
- **独立元素批量**：元素互相独立，可流水线/向量化，受限于吞吐量。

## 构建与运行

```bash
cmake --build build --target ch03_latency_throughput_benchmark
./build/chapter03_measurement/ch03_latency_throughput_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，10M 步）

- 串行链：~0.76 ns/步（延迟受限，每一步等待上一步完成）；
- 独立批量：~0.57 ns/元素，吞吐 ~1.7 ops/ns。

链式依赖无法隐藏延迟；独立批量可利用 CPU 的多执行单元与向量化。

## 含义

- 需要低延迟的交互/实时应用关注延迟；批量转换等关注吞吐；
- 优化目标不同：延迟优化减少依赖链，吞吐优化提升并行度/向量化。
