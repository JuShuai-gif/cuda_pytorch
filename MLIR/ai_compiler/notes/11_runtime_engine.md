# 11 · 运行时引擎：ExecutionContext / GraphExecutor / Scheduler

> 对应代码：`include/Edge/Runtime.h`、`src/Runtime/Runtime.cpp`、`tools/edge-run`
> 验证：`ninja -C build check-edge`（run 测试通过，matmul×relu 校验和=8.0）

---

## 1. 中文原理讲解

运行时负责"拿到编译产物 → 真正算出结果"。本模块实现一个最小但真实的图执行运行时：

- **ExecutionContext**：`SSA Value → Tensor` 的存储（`llvm::DenseMap<Value, Tensor>`），相当于
  运行期的"寄存器堆/张量池"。输入（函数入参）由调用方预置。
- **OperatorScheduler**：决定算子执行顺序。当前返回**程序顺序的拓扑序**（SSA 已保证 def-before-use），
  并预留异步/并行扩展点（多流、算子级并行）。
- **GraphExecutor**：按调度顺序逐个 dispatch 算子到 kernel（`constant` / `relu` / `matmul`），把结果写回
  context；每个算子用 `std::chrono` 计时并交给 Profiler。
- 验证：`matmul(全1[2x2])×(全1[2x2])` → 每元素 2.0；`relu` 不变；输出校验和 8.0，正确。

这是"解释执行型"运行时（像 TFLite interpreter / ONNXRuntime 的 sequential executor）。与之相对的是
"编译执行型"（把图 lower 到 LLVM IR → JIT/AOT 成机器码，见 Module 10）。两条路本项目都有。

## 2. 工业背景

推理运行时是部署的最后一公里：它管理张量内存、调度算子、对接硬件 backend。设计点包括：同步/异步、
单流/多流、静态/动态形状、内存池复用（接 Module 09）、算子分发（查表 vs JIT）。

## 3. TensorRT 对应模块

- ExecutionContext ≈ TensorRT `IExecutionContext`（绑定输入输出、持有 workspace）。
- GraphExecutor/Scheduler ≈ TensorRT engine 的 `enqueueV3`（在 CUDA stream 上调度 fused layer kernel）。
- 我们的"解释执行"对应 TensorRT 的"engine 执行"，只是后者是预编译好的 kernel 序列。

## 4. TVM 对应模块

- ExecutionContext + Executor ≈ TVM `GraphExecutor` / `VirtualMachine`（持有 storage pool, 顺序/VM 执行）。
- Scheduler ≈ VM 的指令序列；kernel ≈ TVM 编译出的 `PackedFunc`。

## 5. TPU-MLIR 对应模块

≈ TPU-MLIR 配套的 `model_runner` / bmodel 运行时：加载编译产物, 在 TPU 上按算子序调度执行。

## 6. Ascend CANN 对应模块

≈ ACL (AscendCL) 运行时：`aclmdlExecute` 在 device 上执行编译好的 om 模型, 管理 stream 与内存。

## 7. 性能收益

- 运行时本身的开销应尽量小：算子分发用查表 + 预解析（避免每次执行都遍历 IR）。
- 异步/多流调度可重叠计算与拷贝（H2D/D2H），是吞吐优化的关键（本模块预留扩展点）。
- 配合 Module 09 的静态 arena, 推理期零分配, 延迟可预测。

## 8. Trade-off

- 解释执行：实现简单、易调试, 但每算子有分发开销, 不如 AOT/JIT 极致 → 适合调试/参考实现。
- 编译执行（lower 到 LLVM）：极致性能, 但编译期长、调试难 → 适合生产。
- 同步执行简单可预测；异步吞吐高但难调试、难保证实时确定性（机器人场景需权衡）。

## 9. 常见 Bug（本模块真实注意点）

1. **输入未预置**：执行 relu/matmul 前其操作数必须已在 context, 否则 `ctx.has()` 失败 → 必须先填入参。
2. **splat 常量**：`dense<1.0>` 只存 1 个值, 解析时要广播填满（本模块已处理）。
3. **行主序假设**：matmul kernel 假设行主序 + 2D; 与 lowering/ODS 的 layout 约定要一致。
4. **形状不匹配**：matmul 要求 `lhs.K == rhs.K0`, 否则 kernel 返回 failure（执行前最好先跑 shape-inference）。

## 10. 调试方法

- `edge-run --edge-fill=1.0`：用全 1 输入跑, 校验和可手算核对（如本例 8.0）。
- 对每个 kernel 单测：构造单算子 IR, 比对输出。
- 不支持的算子会打印 warning 并跳过, 便于定位缺失 kernel。

## 11. Profiling 方法

- 内置 Profiler 直接给 per-op 延迟与占比（见 Module 12）。
- 对比"解释执行"与"lower 到 LLVM 后执行"的延迟, 量化编译执行的收益。

## 12. 在机器人 / VLA 中的应用

机器人推理运行时必须**确定性、低抖动**：固定算子序、静态内存、可选单流同步执行以保证控制环周期稳定。
本模块的 Scheduler 预留异步扩展点, 未来可把多相机预处理与策略主干分流并行, 在保证实时性的前提下提吞吐。

> 下一步（Module 12）：把执行时的 per-op 延迟/内存做成时间线与 trace, 形成完整 Profiling 报告。
