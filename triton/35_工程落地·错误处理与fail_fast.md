# 35 工程落地·错误处理与 fail-fast（深度版）

> 本类问题的共性：编译器错误分两类——**输入错误**（DSL 写错，应友好提示）与**内部错误**（pass 断言失败，应 fail-fast）。难点在于"**哪些错该报、报多细、是静默还是当场拒绝**"。

## 机制 1：CompilationError 包装

### 约束
- DSL 语法/语义错误要转成用户可读的编译错误，而非裸 Python 异常。

### 实现（`compiler/code_generator.py` + `errors.py`）
- `call_Function`（:1496）把 builtin 调用的异常包装成 `CompilationError`。
- `TRITON_FRONT_END_DEBUGGING=1` 关闭包装，透传原始异常（knobs.py:367）。

> **不变量 E1**：用户 DSL 错误默认以 CompilationError 形式出现，且可选透传原始异常调试。

### 工程判断
- **异常包装 = 给用户的友好层**：内部堆栈对用户无意义，包装成"哪个函数哪行"。
- **调试开关**：`TRITON_FRONT_END_DEBUGGING` 让开发者看原始异常——**用户友好与开发者可调试可切换**。

## 机制 2：PTXASError 错误分类

### 约束
- ptxas 编译失败有多种原因（内部错误、SIGSEGV、其他）。
- 失败要区分并给出原因。

### 实现（`nvidia/backend/compiler.py:570-575`）
```python
# returncode 解析
if returncode == 255: error = "ptxas internal error"
elif returncode == SIGSEGV: error = "ptxas segfault"
else: error = "unknown"
raise PTXASError(f"{error}: {cmd}\n{output}")
```

> **不变量 E2**：ptxas 失败必须被识别并归因（内部 vs 崩溃 vs 其他），且带完整命令与输出。

### 工程判断
- **错误归因**：按 returncode 区分错误类型，便于定位是编译器 bug 还是环境问题。
- **带上下文**：命令 + 输出，否则无法复现。

## 机制 3：编译器崩溃的 reproducer

### 约束
- 编译器自身崩溃（C++ bug）需要可复现的最小输入。

### 实现（AGENTS.md + knobs）
```
编译器崩溃时打印 mlir_reproducer（含 {-# ... #-} metadata）
→ 保存到 /tmp/x.mlir
→ triton-opt /tmp/x.mlir --run-reproducer 复现
→ triton-reduce 最小化
```

> **不变量 E3**：任何编译器崩溃都必须能产出一个可离线复现的最小 IR 输入。

### 工程判断（深度）
- **崩溃即交付物**：不要求崩溃前自己定位，但必须让开发者能复现——**把"可复现性"设计进崩溃路径**。
- `triton-reduce` 自动最小化，把"缩小问题"从手工变自动。
- 这是 MLIR 生态最强的工程资产，也是 Triton 相对 TileLang 的调试优势（见 `18`）。

## 机制 4：fail-fast vs 静默——mma 不支持组合

### 约束
- `tl.dot` 的 dtype/shape 可能不满足任何 mma 版本。
- 静默用 FMA 可能产生错误（如 fp8 不支持）。

### 实现（`getMMAVersionSafe` + `supportMMA`）
```cpp
if (!supportMMA(op, baseVersion)) continue;   // 尝试下一个版本
return 0;   // 都不支持 → 用 FMA（显式降级）
```

> **不变量 E4**：mma 版本选择必须显式——要么支持，要么明确降级 FMA，绝不静默用错指令。

### 工程判断
- **显式降级 ≠ 静默错误**：FMA 兜底是"支持但没有 tensor core"的明确选择，不是"没想到"。边界：fp8 输入走 FMA 才是错误（`supportMMA` 拒绝 fp8 在 FMA 下无意义）。
- 这是"该拒绝的拒绝、该降级的明确降级"的边界。

## 本类工程判断总结

1. **用户错误友好包装，开发者可透传**（CompilationError + TRITON_FRONT_END_DEBUGGING）。
2. **错误归因**：按 returncode 分类，带命令与输出。
3. **崩溃可复现性设计进崩溃路径**（reproducer + triton-reduce）。
4. **显式降级 ≠ 静默错误**：FMA 兜底是明确选择。
5. **AI 最弱环节：静默实现不可证明的组合**——Triton 用 fail-fast + 显式降级防御。

## 深入自测

1. CompilationError 包装的作用？调试开关是什么？
2. PTXASError 如何按 returncode 分类？
3. reproducer 机制的价值？triton-reduce 做什么？
4. FMA 兜底是静默错误还是显式降级？边界在哪？
5. 对比 TileLang 的错误处理（fail-fast 布局冲突）有何异同？

## 下一步

进入 `36_工程落地·同步与内存序.md`。
