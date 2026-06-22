# 实验 5 常见错误

## 1. ONNX 导出模拟错误

### 错误：使用不支持的算子
**问题描述：** 模型中使用了 ONNX 不支持的 PyTorch 操作。
**正确做法：** 在实际 ONNX 导出中，需要确保所有操作都是 ONNX 标准算子。避免使用控制流、动态形状等复杂操作。
```python
# 可能不支持的操作
# - torch.where 的动态条件
# - 动态循环
# - 非标准激活函数
# 使用 torch.onnx.export 时添加 opset_version 参数
```

### 错误：输入形状不是固定的
**问题描述：** 未指定固定的输入形状，导致 ONNX 导出失败或产生动态图。
**正确做法：** 使用固定的 `dummy_input` 进行 tracing。
```python
# 正确做法
dummy_input = torch.randint(0, vocab_size, (1, 32))  # 固定形状
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## 2. INT8 量化错误

### 错误：量化 Embedding 层
**问题描述：** 对 Embedding 层也进行了 INT8 量化。
**正确做法：** 通常不对 Embedding 层进行量化，因为：
1. Embedding 层本质是查表操作，不是数学运算
2. 量化 Embedding 会导致精度显著下降
3. 许多推理框架自动排除 Embedding 层

### 错误：未正确恢复量化权重
**问题描述：** INT8 量化后未进行反量化，直接用整数进行推理。
**正确做法：** 在模拟实验中，量化后需要反量化为 FP32 才能继续用 FP32 数学库推理。实际部署中，推理框架会使用 INT8 数学指令。
```python
# 模拟量化
scale = weight.abs().max() / 127.0
q_weight = torch.clamp(torch.round(weight / scale), -128, 127)
deq_weight = q_weight.float() * scale  # 反量化用于 FP32 推理
```

### 错误：使用逐张量量化而非逐通道量化
**问题描述：** 对整个权重张量使用一个 scale。
**正确做法：** 至少使用逐通道量化（每个输出通道一个 scale），以减少量化误差。
```python
# 逐通道量化
for out_ch in range(weight.shape[0]):
    ch_weight = weight[out_ch]
    scale = ch_weight.abs().max() / 127.0
    q_weight[out_ch] = torch.clamp(torch.round(ch_weight / scale), -128, 127)
    deq_weight[out_ch] = q_weight[out_ch].float() * scale
```

---

## 3. CPU 基准测试错误

### 错误：未进行预热
**问题描述：** 第一次推理包含初始化开销（如内存分配、CPU 缓存加载），导致延迟数据不准确。
**正确做法：** 先预热 10-20 次后再计时。
```python
# 预热
for _ in range(warmup):
    _ = model(input_tensor)

# 计时
start = time.time()
for _ in range(num_runs):
    _ = model(input_tensor)
end = time.time()
avg_latency = (end - start) / num_runs
```

### 错误：未考虑批次大小对延迟的影响
**问题描述：** 只用 batch_size=1 测试延迟。
**正确做法：** 测试多种批次大小（1, 4, 8, 16），因为边缘设备上的批处理策略会影响整体性能。延迟并非随 batch_size 线性增长。

### 错误：使用 GPU 进行边缘设备基准测试
**问题描述：** 在 GPU 上进行基准测试来模拟边缘设备。
**正确做法：** 使用 CPU 进行测试，可以进一步设置线程数模拟低功耗 CPU。
```python
# 模拟边缘设备的 CPU 限制
torch.set_num_threads(2)  # 限制为 2 线程
```

---

## 4. 部署报告错误

### 错误：忽略内存占用
**问题描述：** 只关注模型大小和延迟，忽略了运行时内存占用。
**正确做法：** 运行时内存 = 模型参数 + 中间激活 + KV Cache（对于生成任务）。需要综合考虑所有内存来源。
```python
# 估算峰值内存
param_memory = total_params * bytes_per_param
activation_memory = batch_size * seq_len * d_model * num_layers * bytes_per_float
kv_cache_memory = 2 * batch_size * seq_len * d_model * num_layers * bytes_per_float
peak_memory = param_memory + activation_memory + kv_cache_memory
```

### 错误：忽略预热和初始化开销
**问题描述：** 报告中只报告稳定状态的推理延迟。
**正确做法：** 部署报告应包含冷启动延迟（首次推理）和稳定状态延迟。边缘设备上的冷启动可能对用户体验有显著影响。

---

## 5. 概念性错误

### 错误：认为量化总是 4× 压缩
**问题描述：** 认为 INT8 量化一定使模型大小减少为原来的 1/4。
**正确做法：** 实际上并非所有参数都能量化（如 Embedding、LayerNorm），且需要额外存储 scale 和 zero_point 参数。实际压缩率通常略低于理论值。

### 错误：忽视量化对 BatchNorm 的影响
**问题描述：** 量化模型时直接保留 BatchNorm 层。
**正确做法：** 在实际 INT8 部署中，通常将 Conv-BN 或 Linear-BN 融合为一个操作，因为量化推理更高效。
```python
# 融合 Conv + BN + ReLU 为单一量化操作
# 这在 ONNX Runtime 和 TensorRT 中自动完成
```

### 错误：混淆模型压缩和推理加速
**问题描述：** 认为模型越小推理一定越快。
**正确做法：** 虽然 INT8 量化减小了模型，但实际加速依赖于：
1. 硬件是否支持 INT8 指令（如 ARM NEON、Intel VNNI）
2. 内存带宽是否成为瓶颈
3. 推理框架是否针对 INT8 优化
在纯 CPU 模拟中，INT8 量化的加速可能不明显（因为模拟使用 FP32 数学），但在实际硬件上有显著加速。
