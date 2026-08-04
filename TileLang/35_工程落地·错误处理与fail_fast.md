# 35 工程落地·错误处理与 fail-fast（深度版）

> 本类问题的共性：编译器错误分两类——**输入错误**（DSL 写错，应友好提示）与**内部错误**（pass 断言失败，应 fail-fast）。难点在于"**哪些错该报、报多细、是静默还是当场拒绝**"。这是 AI vibe coding 最常犯错的领域：AI 倾向静默处理不可证明正确的组合。

## 机制 1：布局冲突——当场拒绝而非静默选择

### 约束
- 同一 buffer 被多个 tileop 使用，每个 op 提出布局要求。
- 布局错了 = 数据错位 = **静默错误**（最可怕）。

### 实现（`layout_inference.cc:246-264`）
```cpp
LOG(FATAL) << "Get different layout for " << buffer
           << "\n current layout: " << layout->DebugOutput()
           << "\n previous layout: " << layout_map[buffer]->DebugOutput();
```

> **不变量 E1**：布局冲突时，必须立即中止编译，绝不静默选一个。

### 为什么 fail-fast
- 静默选一个是"把不可证明正确的选择藏起来"，最终以数据错位暴露——比报错危险得多。
- 这正是面试题后记抨击的第一类 AI 通病。

### 例外：什么时候可以合并
- 两个 swizzle 布局可 `MergeSwizzleLayouts` 取较小粒度（:249-257）。
- **这是有明确数学理由的合并**（bank 优化非语义），与"瞎猜"不同——判断标准是"是否有可证明的正确性依据"。

## 机制 2：错误定位——span 溯源

### 约束
- 报错要能定位到用户源码行，否则无法排查。

### 实现（`errors.py` + `span_utils.h`）
```
Check failed: (layout_map.count(buffer) != 0) is false:
The layout for fragment C_local can not be inferred correctly.
  --> /path/to/kernel.py:21:1
   |
21 |     C_local = T.alloc_fragment(...)
   |     ^
```

> **不变量 E2**：每个编译器错误都携带用户源码位置（`SpanHintSuffix`）。

### 工程判断
- 编译错误是"给人类看的"，定位信息是用户体验核心。
- `enrich_error` 渲染带源码行和脱字符的片段——**把内部错误翻译成可行动的提示**。

## 机制 3：语义检查（输入错误的友好拦截）

### 约束
- DSL 参数/类型错误应在 lower 前拦截，给出友好信息。

### 实现（`engine/semantic_check.py`）
- 通过 `TL_DISABLE_PRELOWER_SEMANTIC_CHECK` 控制开关（默认开）。
- 检查参数类型、dtype 合法性。

> **不变量 E3**：可静态判定的输入错误，必须在编译早期、以友好方式报出。

### 工程判断
- **分层错误策略**：输入错误（可恢复，友好提示）vs 内部错误（fail-fast，崩溃报告）。二者不得混淆。
- 这区别于把一切当崩溃：语义检查是"给用户的机会"，`LOG(FATAL)` 是"给开发者的 bug"。

## 机制 4：编译回调的错误传播

### 约束
- nvcc 编译失败要透传原始错误（用户需要看编译器输出）。

### 实现（`lower.py:101-175`）
```python
# nvcc 失败 → RuntimeError 带完整命令 + 输出
raise RuntimeError(f"nvcc failed: {cmd}\n{output}")
```

### 工程判断
- **错误要带上下文**：命令、输出、源码——否则无法复现。
- 与 Triton 的 `PTXASError`（解析 returncode：255=内部错误、SIGSEGV 等）同构。

## 本类工程判断总结

1. **不可证明正确就当场拒绝**：布局冲突 fail-fast，绝不静默。
2. **区分输入错误与内部错误**：前者友好提示、后者崩溃报告，不得混淆。
3. **错误带定位和上下文**：span 溯源 + 完整命令输出。
4. **合并不等于瞎猜**：只有有数学理由的合并才合法（swizzle merge）。
5. **这正是 AI 编码最弱的环节**：AI 倾向静默实现不可证明的组合。

## 深入自测

1. 布局冲突为什么 fail-fast？什么例外可以合并？
2. E2 不变量如何实现？
3. 输入错误与内部错误的处理有何不同？
4. nvcc 失败如何透传？
5. 面试题后记抨击的 AI 通病，编译器工程如何防御？

## 下一步

进入 `36_工程落地·同步与内存序.md`。
