# 40 工程落地·pass 编排与依赖（深度版）

> 本类问题的共性：编译器是 pass 流水线，pass 之间有**顺序依赖**（前面的产出是后面的输入）。难点在于"**顺序必须正确、依赖必须声明、可重入性必须保证**"。

## 机制 1：pass 流水线的顺序约束

### 约束
- pass 之间有严格先后（如 LayoutInference 必须在 LowerTileOp 前）。
- 顺序错了 → 崩溃或错误产物。

### 实现（`cuda/pipeline.py:145-258`）
```python
mod = tilelang.transform.MaterializeKernelLaunch()(mod)
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
mod = tilelang.transform.LayoutInference()(mod)   # 必须在 LowerTileOp 前
mod = tilelang.transform.LowerTileOp()(mod)       # 展开 tileop
...
```

> **不变量 PO1**：pass 流水线的顺序满足每个 pass 的"输入前提"。

### 工程判断
- **顺序 = 依赖的显式化**：把依赖关系写死在流水线里，比每个 pass 自查可靠。
- 违反顺序的后果（如先 LowerTileOp 后 LayoutInference）会是：tileop 已被展开，布局推断失去语义锚点 → 错误。

## 机制 2：pass 注册与命名

### 约束
- pass 需要全局唯一名（`tl.transform.Xxx`）供 FFI 调用。
- 注册与 Python 绑定必须对应。

### 实现（`layout_inference.cc:1266-1281`）
```cpp
tvm::transform::Pass LayoutInference() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) { ... };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}
TVM_FFI_STATIC_INIT_BLOCK("tl.transform") {
  refl::GlobalDef().def("tl.transform.LayoutInference", LayoutInference);
}
```

> **不变量 PO2**：pass 的 C++ 注册名与 Python 绑定名必须一致。

### 工程判断
- **命名即契约**：`tl.transform.Xxx` 是跨语言的稳定接口。
- `CreatePrimFuncPass(func, 0, "tl.Xxx", {})` 的第 4 参 `required_passes` 可声明依赖——**pass 依赖显式声明**。

## 机制 3：pass 可重入性（同一 IRModule 多次跑）

### 约束
- 同一个 pass 可能被跑多次（不同 target 或调试）。
- pass 必须是纯函数：输入 IRModule → 输出新 IRModule，不改输入。

### 实现
- `CreatePrimFuncPass` 的 pass_func 是 `PrimFunc → PrimFunc`，TIR 对象不可变（immutable）。
- 改动通过构造新对象完成。

> **不变量 PO3**：pass 不修改输入 IRModule，返回新值。

### 工程判断
- **不可变性 = 可重入性**：TIR 的 immutable 设计保证 pass 可安全复用、可并行。
- 这与函数式编程的纯函数原则一致，是编译器正确性的基石。

## 机制 4：host/device IR 分离的编排

### 约束
- 最终产物分 host（launcher）与 device（kernel）。
- 分离时机必须在所有 device 优化之后。

### 实现（`lower.py:259` lower_to_host_device_ir）
- `SplitHostDevice` pass 在 device pass 链之后执行。
- host 部分（参数打包/launch 逻辑）与 device 部分（kernel 本体）分离。

> **不变量 PO4**：device 优化完成后才分离，host/device 边界清晰。

### 工程判断
- **分离是流水线的最后阶段**：如果提前分离，device pass 无法作用到 kernel 内部。
- 合并进一个 `.so` 管理生命周期（见 `34`）。

## 机制 5：配置驱动的 pass 行为（pass_configs）

### 约束
- 同一 pass 在不同配置下行为不同（如 fast_math 是否开）。

### 实现
```python
with tvm.transform.PassContext(config={"tl.enable_fast_math": True}):
    mod = transform.Simplify()(mod)
```
- pass 用 `PassContext::Current()` 读配置。

> **不变量 PO5**：pass 行为只由输入 IR + 当前 PassContext 决定（确定性）。

### 工程判断
- **配置外置**：pass 不读全局可变状态，只读 PassContext——保证确定性与可重放。
- 这使"同一流水线多配置"（调优）成为可能。

## 本类工程判断总结

1. **顺序 = 依赖显式化**：写死在流水线里。
2. **命名即契约**：跨语言稳定接口。
3. **不可变性 = 可重入性**：TIR immutable 保证 pass 安全复用。
4. **分离时机正确**：device 优化完成后再 split。
5. **配置外置保证确定性**：pass 只读 PassContext。

## 深入自测

1. 为什么 LayoutInference 必须在 LowerTileOp 前？
2. pass 注册名与 Python 绑定如何对应？
3. 不可变性如何保证 pass 可重入？
4. host/device 分离为什么在链尾？
5. pass_configs 如何保证确定性？

## 下一步

回顾 `32_工程落地设计·总纲` 与 `31_架构设计与工程权衡.md`，形成工程维度全景。
