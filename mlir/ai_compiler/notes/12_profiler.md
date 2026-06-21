# 12 · Profiler：延迟分解 / 内存分解 / 时间线 / 执行 Trace

> 对应代码：`include/Edge/Runtime.h`（`Profiler`）、`src/Runtime/Runtime.cpp`、`tools/edge-run`
> 验证：`ninja -C build check-edge`（run 测试输出含 per-op 延迟分解）

---

## 1. 中文原理讲解

Profiler 回答"时间和内存花在哪了"。本模块的 Profiler 在运行时给每个算子计时（`std::chrono`），
记录 `(算子名, 延迟ms, 输出字节)`，输出 Markdown 报告：

- **延迟分解（latency breakdown）**：每算子耗时与占总延迟的百分比（验证输出：matmul 77.7% / relu 22.3%）。
- **内存分解（memory breakdown）**：每算子输出张量字节数（接 Module 09 的峰值内存）。
- **时间线 / trace（扩展）**：把 `(start, end, op)` 序列导出为 Chrome Trace（`chrome://tracing` / Perfetto）
  的 JSON，可视化算子重叠与气泡。这是 `LLVM_DEBUG`/`TimeTraceScope` 的运行期对应物。

定位瓶颈的标准方法：先看延迟分解找 top-N 热点算子，再针对性优化（融合/量化/换 kernel）。

## 2. 工业背景

"先测量，再优化"。所有部署框架都内建 profiler：没有 per-op 延迟分解, 优化就是盲猜。Profiler 还用于
回归监控（性能不退化）、容量规划（是否满足实时预算）。

## 3. TensorRT 对应模块

≈ TensorRT 的 `IProfiler`（回调每层耗时）+ `trtexec --dumpProfile` + Nsight Systems 时间线。

## 4. TVM 对应模块

≈ TVM 的 `debug_executor` / `time_evaluator`（per-op 耗时）+ VM profiler。

## 5. TPU-MLIR 对应模块

≈ TPU-MLIR/SOPHON 的 `bmprofile`：算子级耗时、片上/片外带宽、指令分布。

## 6. Ascend CANN 对应模块

≈ CANN 的 `msprof` / Profiling 工具链：算子耗时、AI Core 利用率、内存带宽、PMU 计数。

## 7. 性能收益

- Profiler 不直接提速, 但**指引优化方向**：80/20 法则下, 优化 top 热点收益最大。
- 延迟分解能验证融合/量化是否真的生效（融合后该段延迟应下降）。

## 8. Trade-off

- 计时本身有开销（`chrono` 调用、缓存扰动）；细粒度 profiling 会扰动被测对象（observer effect）。
- 同步计时简单准确, 但异步/多流下需用事件（CUDA event 式）才能正确归因延迟。
- 输出字节是"逻辑内存", 与真实分配（含临时/对齐/复用）有差, 精确内存看 Module 09。

## 9. 常见 Bug

1. **计时把分发开销算进 kernel**：应只包住 kernel 计算段；首个算子常含冷启动/分配开销, 需 warmup 多次取均值。
2. **异步未同步就计时**：GPU/NPU 上必须等 kernel 完成（event/sync）才停表, 否则测的是 launch 时间。
3. **百分比除零**：总延迟为 0（极小图）时百分比要保护（本模块已判 `total>0`）。

## 10. 调试方法

- `edge-run` 直接打印延迟分解；多跑几次看抖动（首跑偏高属正常冷启动）。
- 异常热点：单独跑该算子、放大输入规模, 确认是否计算密集还是访存密集。

## 11. Profiling 方法（本模块即工具）

- 延迟分解：`edge-run` 内置。
- 内存分解：per-op 输出字节 + Module 09 的规划峰值。
- 时间线：可扩展导出 Chrome Trace JSON（`{"name","ph":"X","ts","dur"}` 序列）。

## 12. 在机器人 / VLA 中的应用

机器人控制环有硬延迟预算（如 20 ms @ 50 Hz）。Profiler 用于：①验证整条 VLA 推理是否落在预算内；
②定位是视觉编码、attention 还是动作头最耗时, 据此决定融合/量化/降分辨率；③监控多相机管线各路延迟,
做调度均衡。延迟分解 + 时间线是把"能不能上机器人"量化成数据的关键工具。

> 后续（Module 16）：把 shape-inference → fusion → lowering → run/profiling 串成端到端驱动, 一键产出
> fusion/compilation/latency/memory 报告。
