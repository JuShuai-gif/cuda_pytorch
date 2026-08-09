# Chapter 18：边缘端 VLA 推理流水线

| 示例 | 工业能力 |
|---|---|
| `01_edge_vla_pipeline.cpp` | latest-frame 队列、背压、`jthread/stop_token`、deadline、P50/P95/P99、优雅停机、自检 |

```bash
./ch18_01_edge_vla_pipeline
./ch18_01_edge_vla_pipeline --self-test
```

示例以 CPU sleep 模拟设备执行。接入 TensorRT、LibTorch 或 ONNX Runtime 时，保留队列、停止协议和指标结构，将推理函数替换为真实后端，并用 CUDA event 测量设备时间。
