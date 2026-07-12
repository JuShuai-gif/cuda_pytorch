# torch.export 模型导出与部署源码分析

> 源码路径: `/home/ghr/code/pytorch/torch/export/` (1812 行 `exported_program.py`)
> 入口: `torch/export/__init__.py`, 核心函数 `export()` (line 59)
> 动态形状: `torch/export/dynamic_shapes.py`

## 0. 一句话总览

`torch.export` 是 PyTorch 2.x 的**标准化模型导出方案**，将 `nn.Module` 编译为 `ExportedProgram`（一个包含 FX Graph + state_dict + 元数据的自包含部署包）。核心：**全程走 tracing（非 scripting），保证图是 static single assignment (SSA) 形式，无 Python control flow**。

---

## 一、核心 API: `torch.export.export()`

```python
# __init__.py:59
def export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: Any = None,           # ★ 动态维度声明
    strict: bool = False,                  # True: 图必须包含所有 op
    preserve_module_call_signature: tuple[str, ...] = (),
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
```

### 参数解读:
- `args/kwargs`: 示例输入，用于 tracer 捕获计算图
- `dynamic_shapes`: 声明哪些维度是动态的，支持范围约束 (`min/max`)
- `strict`: `True` 时不自动填充缺失的 op → 保证图的完整性

---

## 二、`ExportedProgram` — 导出的核心数据结构

```python
# exported_program.py:1062
class ExportedProgram:
    _graph_module: torch.fx.GraphModule          # FX 计算图
    _graph_signature: ExportGraphSignature        # 输入/输出签名（名称/类型/位置）
    _state_dict: dict[str, Any]                   # 参数 + buffer 的数值
    _range_constraints: dict[sympy.Symbol, ValueRanges]  # 符号维度约束
    _module_call_graph: list[ModuleCallEntry]     # 子模块调用层级
    _example_inputs: tuple[...] | None            # 示例输入
    _constants: dict[str, ...]                     # 常量（如 padding, stride）
    _verifiers: list[type[Verifier]]              # 验证器
    _guards_code: list[str]                        # guard 条件代码
```

### `ExportedProgram` 是可调用的:

```python
ep = export(model, (x,))
output = ep(*new_args)  # 直接用新输入运行
```

### 序列化/反序列化:

```python
torch.export.save(ep, "model.pt2")        # 保存
ep = torch.export.load("model.pt2")       # 加载
```

---

## 三、动态形状 (`dynamic_shapes.py:109`)

### `Dim` 类

```python
_dim = Dim("batch_size", min=1, max=1024)
# 支持: max, min 范围约束
```

### 三种声明方式

**旧版 (字典):**

```python
dynamic_shapes = {
    "x": {0: Dim("batch"), 1: Dim("seq", max=512)},
    "y": {0: Dim("batch")},
}
```

**新版 (`ShapesCollection`):**

```python
batch = Dim("batch", max=128)
dynamic_shapes = ShapesCollection(
    x=(batch, 512),   # 第一个维度动态，第二个静态
    y=(batch, 10),
)
```

**`Constraint` — 维度间关系:**

```python
dynamic_shapes = ShapesCollection(
    x=(batch, seq),
    y=(batch, seq),
    constraints=[Constraint(x_seq=seq, y_seq=seq)]  # x[:,1].size == y[:,1].size
)
```

---

## 四、导出流程源码深度分析

`_export_to_torch_ir` 是核心入口，位于 `torch/export/_trace.py:884`。

### 4.1 函数签名

```python
# _trace.py:884
def _export_to_torch_ir(
    f: Callable,                              # nn.Module.forward 或任意 callable
    args: tuple[Any, ...],                    # 位置参数
    kwargs: dict[str, Any] | None = None,      # 关键字参数
    dynamic_shapes: _DynamicShapesInput = None, # Dim 或 ShapesSpec
    *,
    preserve_module_call_signature: tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    ...
) -> torch.fx.GraphModule:
```

### 4.2 内部执行流程 (:884-983)

```
Step 1: [参数处理 :915-937]
  args, kwargs = tree_map_only(int, _IntWrapper, (args, kwargs))
  → 将普通 int 包装为 _IntWrapper，以便后续标记为 dynamic/static
  → dynamic_shapes 中的 Dim 约束被收集到 constraints 列表

Step 2: [Dynamo trace :956-965]
  with torch._dynamo.config.patch(dynamo_cfg):
      with _wrap_submodules(f, ...):
          dynamo_graph_capture = dynamo_graph_capture_for_export(f)
          → 用 Dynamo 捕获计算图（走 AOT export 路径, 不同于 torch.compile）

Step 3: [decompose + functionalize] 在 Dynamo 之后自动进行:
  - decompose: composite op → core ATen op (如 F.gelu → aten::gelu)
  - functionalize: in-place op → functional op (add_ → add)

Step 4: [unlift] 将 parameter/buffer 提升为图输入:
  - 原图: param 和 buffer 作为 constant 嵌入图中
  - 提升后: param 和 buffer 变成 placeholder 节点 → graph signature 记录

Step 5: [包装为 ExportedProgram :exported_program.py:1062]
  GraphModule + graph_signature + state_dict + range_constraints
    → ExportedProgram
```

### 4.3 Dynamic shapes 解析 (:1459 及后续)

```python
# _trace.py:1459
def _process_export_inputs(args, kwargs, dynamic_shapes):
    # 支持三种格式:
    # {name: {dim: Dim(...)}}        — 旧版 dict
    # ShapesCollection(x=(Dim, 8))   — 新版 API
    # ShapesSpec(...)                — ParamsSpec 格式
    ...
```

`Dim` 实例会被转换为符号变量 (`sympy.Symbol`)，约束 (`min`, `max`) 转为 `ValueRangeAnalysis`，维度间关系 (`Constraint`) 转为等式。

---

## 五、关键源码位置速查

| 机制 | 文件 | 行号 |
|------|------|------|
| `export()` 入口 | `torch/export/__init__.py` | 59 |
| `ExportedProgram` 类 | `torch/export/exported_program.py` | 1062 |
| `Dim` 类 | `torch/export/dynamic_shapes.py` | 109 |
| `ShapesCollection` | `torch/export/dynamic_shapes.py` | — |
| `ExportGraphSignature` | `torch/export/graph_signature.py` | — |
| `_export_to_torch_ir` | `torch/export/_trace.py` | — |
| `save` / `load` | `torch/export/__init__.py` | — |
| `unflatten` | `torch/export/unflatten.py` | — |
| 图 passes | `torch/export/passes/` | — |

---

## 六、可借鉴的工程技巧

1. **SSA 保证**: `torch.export` 强制输出图为 static single assignment 形式 → 无 python control flow → 可在任何后端（C++/Rust/TVM）直接执行。

2. **graph signature**: 将参数/buffer 的名称、类型、输入/输出位置统一签名 → 后端知道每个 tensor 是参数还是 buffer 还是用户输入。

3. **符号维度约束**: `Dim(min=1, max=1024)` 让 runtime 在 guard 失败时知道合法范围 → 可以 fallback 或 re-export。

4. **functionalize**: 在导出前将 in-place op（`add_`, `mul_`）转为 functional（`add`, `mul`），消除副作用 → 图变成纯函数。

5. **保存/加载**: `save` 使用 zip archive 格式，包含 graph JSON + state_dict.pt + constants → 自包含，可跨语言加载。

---

## 七、实战常见坑点

### 1. Dynamic shapes 编译不通过
**现象**: 声明了 `Dim("batch", max=128)`，但 export 时仍报 "Guards for batch_dim"。
**原因**: runtime 输入 shape 超出了 Dim 声明的范围，或者维度被其他约束（如 padding）改变了。
**排查**:
```python
ep = export(model, (x,), dynamic_shapes={"x": {0: Dim("batch", min=1, max=256)}})
print(ep._guards_code)  # 查看生成的 guard 条件
```
**解决**: 放宽 `max` 范围或确保输入严格在此范围内。

### 2. 导出的图与原始模型输出不一致
**现象**: `ep.module()(x)` 的结果与 `model(x)` 有微小差异。
**原因**: export 过程中 op 被 decompose 了（如 `dropout` → `identity`）或精度变了。
**排查**:
```python
# 对比每个 op 的语义是否一致
ep.set_verifier(None)  # 关闭默认 verifier
# 检查 graph 中是否有多余/缺失的 op
for n in ep.graph.nodes:
    if n.op == "call_function":
        print(n.target)
```
**解决**: 确保 `model.eval()`，export 前关闭 dropout、BN 用 `track_running_stats=False` 或设置 running stats。

### 3. 跨版本不兼容
**现象**: PyTorch 2.3 导出的 `.pt2` 文件，在 2.4 上 load 报错 "Unsupported version"。
**原因**: `.pt2` 格式是 zip archive，内含 `exported_program.json` + `state_dict.pt`。ATen op schema 在不同版本间可能不兼容。
**解决**: 用相同或相近的 PyTorch minor 版本导出和加载；或者用 `torch.export.save` 保存额外元数据，`torch.export.load` 会做 version check。

### 4. 非严格模式丢 ops
**现象**: `strict=False` 时某些 op 不见了。
**原因**: `strict=False` 允许自动填充缺失 op → 图可能不完整。
**解决**: 生产环境始终用 `strict=True`，看到报错就知道哪些 op 不支持。

### 5. 参数/buffer 没序列化进 .pt2
**现象**: load 后模型缺少某些参数。
**原因**: persistent=False 的 buffer 不会进 state_dict；某些 constant（如 padding 值）存在 `_constants` 中而非 `state_dict`。
**排查**:
```python
print("state_dict keys:", list(ep.state_dict.keys()))
print("constants:", list(ep._constants.keys()))
```

