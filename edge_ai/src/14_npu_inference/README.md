# NPU 推理优化实战

RK3588 NPU 推理性能优化案例研究：从 13fps 到 29fps。

## 背景

基于真实案例，演示边缘端 NPU 推理管线的三大关键优化：

1. **IO 持久化** — rknn_set_io_mem() 预注册 DMA-BUF，消除每次推理的分配开销 (27ms→0.1ms)
2. **双 Letterbox 修复** — 纠正硬编码预处理尺寸，消除 SDK 隐式 CPU resize (22ms→0.3ms)
3. **NEON LUT FP16 转换** — 256 元素查表法加速模型输出格式转换 (36-55ms→0.3-3.6ms)

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 运行

```bash
./npu_inference_demo
```

## 输出文件

- `npu_inference_metrics.json` — 优化前后指标对比

## 演示内容

| 演示 | 说明 |
|------|------|
| 1. IO 持久化 | 对比 rknn_input_set vs rknn_set_io_mem 的分配开销 |
| 2. 双 Letterbox | 展示硬编码 640×640 导致的 SDK CPU resize 浪费 |
| 3. 完整管线 | 运行 20 帧管线，对比错误配置 vs 正确配置的 FPS 差距 |

## 文件结构

```
14_npu_inference/
├── CMakeLists.txt          # 构建系统
├── README.md               # 本文件
├── main.cpp                # 入口 (3 个演示 + JSON 输出)
├── io_persistence.h/.cpp   # IO 持久化模拟 + NEON LUT
├── double_letterbox.h/.cpp # 双 Letterbox 演示
└── pipeline_sim.h/.cpp     # 完整 5 阶段管线模拟
```

## 关键教训

- 「外围比核心慢」: 管线开销 (IO + 预处理 + 后处理) 远大于 NPU 推理本身
- 「默认 ≠ 最优」: SDK 的自动行为往往是性能陷阱，必须用 perf 数据验证
- 「先测再做」: 不要在没测量之前就凭直觉优化模型
- 「利用率是诊断指标」: NPU 利用率 27%→85%，低利用率说明有阻塞，排查管线而非模型
