# PyTorch torch.compile 源码深扒

> 目标:理解 `torch.compile` 从「Python 函数」到「编译后 kernel」的完整链路,以及帧拦截 / 图捕获 / guard 缓存 / 后端 codegen 四大机制。
> 源码版本:torch 2.10.0,路径 `torch/_dynamo/`、`torch/_functorch/`、`torch/_inductor/`。

## 0. 一句话总览

`torch.compile` 不是一个编译器,而是**三层流水线 + 一套缓存系统**:

```
torch.compile(model)
        │  (① 安装 CPython 帧求值钩子 set_eval_frame)
        ▼
[Dynamo]   逐条字节码符号执行 → 捕获出 FX 图 + 一组 guards      (torch/_dynamo)
        │  (遇到不支持的操作就 graph break,把图切开,中间回 eager)
        ▼   call_user_compiler(gm, example_inputs)   ← backend 在这里被调
[AOTAutograd] trace 出 joint(前向+反向)图 → functionalize → 切成 fw/bw 两张图   (torch/_functorch)
        ▼
[Inductor]  FX 图 → Inductor IR(lowering) → Scheduler 融合 → 生成 Triton/C++ kernel → 编译加载   (torch/_inductor)
        │
        ▼
生成 GuardedCode,挂到该 code object 的 CacheEntry 链表
下次调用:guard 全过 → 直接跑编译产物(零 trace 开销);guard 失败 → recompile
```

核心心智模型:**Dynamo 负责"抓图",AOTAutograd 负责"把反向也纳入图并前后向分家",Inductor 负责"把图变成 kernel"。guard 是连接「编译期假设」与「运行期输入」的契约。**

---

## 1. 三层架构与核心抽象

| 层               | 包                  | 职责                                       | 关键产物                          |
| ---------------- | ------------------- | ------------------------------------------ | --------------------------------- |
| **入口/帧拦截**  | `_dynamo/eval_frame`  | 安装 CPython 帧钩子,接管每个函数帧的执行   | `OptimizeContext` / `OptimizedModule` |
| **图捕获**       | `_dynamo/convert_frame` `symbolic_convert` `output_graph` | 字节码符号执行 → FX 图 + guards            | `GuardedCode(code, guard_manager)`  |
| **守卫/缓存**    | `_dynamo/guards`      | 生成/检查 guard,决定缓存命中还是 recompile | `CacheEntry` 链表                   |
| **前后向分图**   | `_functorch/_aot_autograd` | trace joint 图,functionalize,切 fw/bw     | `fw_module` / `bw_module`             |
| **后端 codegen** | `_inductor`           | FX → IR → 融合 → Triton/C++ kernel         | 可调用的 `CompiledModule`           |

| 概念               | 含义                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| **帧拦截**         | 替换 CPython 的 `_PyFrameEvalFunction`,每个 Python 帧执行前先问 Dynamo |
| **符号执行**       | 不真跑代码,而是用 `VariableTracker` 抽象追踪每条字节码对栈/变量的影响 |
| **graph break**    | 遇到不支持的操作,把图切断,中间片段回退 eager,之后再续编译            |
| **guard**          | 编译期记录的输入假设(dtype/shape/id/类型...),运行期逐条校验          |
| **joint graph**    | 前向 + 反向合在一起的计算图(AOTAutograd 用 `autograd.grad` trace 出)   |
| **lowering**       | 把 `aten` 算子翻译成 Inductor IR(`TensorBox`/`Buffer`/`Operation`)        |

---

## 2. 入口层:从 `torch.compile` 到帧钩子

### 2.1 `torch.compile` 只是 `_dynamo.optimize` 的薄封装

`torch/__init__.py:2551` 是真正实现(2526/2539 是 `@overload` 签名)。它做两件事:**把 mode/options 折叠成 backend wrapper**,然后转交给 `_dynamo.optimize`:

```python
# torch/__init__.py:2742-2756
if backend == "inductor":
    backend = _TorchCompileInductorWrapper(mode, options, dynamic)  # 默认后端
else:
    backend = _TorchCompileWrapper(backend, mode, options, dynamic)

return torch._dynamo.optimize(
    backend=backend,
    nopython=fullgraph,      # fullgraph=True → nopython,不允许 graph break
    dynamic=dynamic,
    disable=disable,
)(model)                     # 注意末尾的 (model):optimize 返回装饰器,立刻作用到 model
```

> `_TorchCompileInductorWrapper.__call__(gm, inputs)` 最终调 `torch._inductor.compile_fx.compile_fx(...)`;`mode`("default"/"reduce-overhead"/"max-autotune")通过 `list_mode_options` 展开成一组 inductor config。

### 2.2 `optimize` → `OptimizeContext`:把 backend 包成"帧回调"

```python
# _dynamo/eval_frame.py:1407 _optimize (精简)
backend = get_compiler_fn(backend)                       # 字符串/注册表 → 编译函数
return _optimize_catch_errors(
    convert_frame.convert_frame(backend, hooks),         # ← backend 被包进帧回调 ConvertFrame
    hooks, backend_ctx_ctor, ...)                        # → 返回 OptimizeContext
```

`backend` 在这里发生了**身份转换**:从「FX 图编译函数」被层层包装成「每个 Python 帧的回调」:
`compile_fx` → `ConvertFrame` → `CatchErrorsWrapper` → `OptimizeContext.callback`。

### 2.3 `_TorchDynamoContext`:用 `set_eval_frame` 安装 CPython 帧钩子

这是整个机制的"开关"。`OptimizeContext` 继承自 `_TorchDynamoContext`,既能当装饰器(`__call__`),也能当上下文管理器(`__enter__/__exit__`):

```python
# _dynamo/eval_frame.py:753-781
def __enter__(self):
    self.prior = set_eval_frame(None)                       # ① 保存并清空旧钩子
    ...
    _maybe_set_eval_frame(_callback_from_stance(self.callback))  # ② 安装 Dynamo 回调 (:765)

def __exit__(self, *exc):
    set_eval_frame(None)                                    # ③ 关闭
    _maybe_set_eval_frame(_callback_from_stance(self.prior))     # ④ 恢复旧钩子 (:779)
```

装饰函数时走 `compile_wrapper`(`:904`),同样在进入时 `_maybe_set_eval_frame(callback)`、`finally` 里恢复——保证 Dynamo **只在受控区域活跃**。

### 2.4 帧拦截原理(C 层)

`set_eval_frame` 是 C 扩展(`torch._C._dynamo.eval_frame`),它替换 CPython 全局的帧求值函数指针。此后**每个 Python 帧执行前**,C 层会:

1. 取该帧 `f_code` 上挂载的 `ExtraState`(通过 `_PyCode_GetExtra`),里面有一条 `cache_entry_list`(编译结果链表);
2. 遍历每个 `CacheEntry`,用其 `guard_manager.check(f_locals)` 校验当前输入;
3. **命中**(guard 全过)→ 直接返回缓存的编译 `code`,完全跳过 Dynamo;
4. **未命中** → 回调 Python 侧 `callback(frame, ...)` → 触发一次完整编译 → 把新的 `GuardedCode` 插入链表。

```c
// CacheEntry(C/Python 共享)概念结构
struct CacheEntry {
    PyCodeObject* code;            // 编译后的字节码(里面会去调 inductor 产物)
    GuardManager guard_manager;    // C++ guard 树
    CacheEntry* next;              // 同一函数的多份编译(不同输入特征)串成链表
};
```

> 这解释了"为什么第一次慢、之后快":第一次走 callback 全量编译,之后只做 guard 校验。

### 2.5 `OptimizedModule`:`nn.Module` 的编译包装

`torch.compile(module)` 返回 `OptimizedModule`(`eval_frame.py:383`),它代理原 module,只把 `forward` 换成被 Dynamo 接管的版本:

```python
# _dynamo/eval_frame.py:429 _initialize (精简)
self.forward = self.dynamo_ctx(self._orig_mod.__call__)  # 用 __call__ 保证经过 hooks
# __getattr__/__setattr__ 把非白名单属性透传/写回 self._orig_mod
```

---

## 3. Dynamo 图捕获:字节码 → FX 图

帧未命中缓存时,callback 进入 `convert_frame.py`,主链:
`CatchErrorsWrapper`(`:2208`,跳过判断/DDP 劫持)→ `ConvertFrame`(`:2053`,异常软失败回退)→ `ConvertFrameAssert`(`:736`)→ **`_compile`**(`:1390`,核心)。

### 3.1 `_compile`:一个 frame → `GuardedCode`

```python
# _dynamo/convert_frame.py:1441 _compile_inner (精简)
dynamo_output = compile_frame(code, globals, locals, ..., compiler_fn, one_graph)  # ① 符号执行出图
check_fn      = dynamo_output.build_guards(code, hooks=hooks)                       # ② 生成 guards
guarded_code  = GuardedCode(out_code, check_fn.guard_manager, compile_id, ...)      # ③ 组装
return wrap_guarded_code(guarded_code)                                              # ④ 交回 C 层
```

它还负责 **recompile 限额检查**(`:1668` `exceeds_recompile_limit`,默认 `cache_size_limit=64`)和异常分类。

### 3.2 `InstructionTranslator`:逐条字节码符号执行

`compile_frame` → `trace_frame` 创建 `InstructionTranslator`,核心是一个 `while step()` 循环:

```python
# _dynamo/symbolic_convert.py:1647 run / :1291 step (精简)
def run(self):
    while self.step():      # 逐条指令推进
        pass

def step(self):
    inst = self.instructions[self.instruction_pointer]
    self.instruction_pointer += 1
    try:
        self.dispatch_table[inst.opcode](self, inst)   # 每个 opcode 一个 handler
        return not self.output.should_exit
    except (Unsupported, StepUnsupported):             # ← graph break 入口
        if self.one_graph or self.current_speculation is None:
            raise                                      # fullgraph / 无 checkpoint → 直接失败
        self.current_speculation.fail_and_restart_analysis(...)  # 否则触发 RestartAnalysis
```

每条字节码(`LOAD_FAST`/`CALL_FUNCTION`/...)不真跑,而是操作 `VariableTracker` 抽象值,并把 tensor 运算记到 FX 图里。

### 3.3 graph break:切图 + resume 函数

遇到不支持的操作(数据依赖控制流、未注册的库调用等)抛 `Unsupported`。Dynamo 的回退用的是 **"从头重跑、这次别走那条路"** 的策略:

```python
# _dynamo/symbolic_convert.py:236 SpeculationEntry
def fail_and_restart_analysis(self, ...):
    self._failed = True
    raise exc.SpeculationRestartAnalysis()   # 外层 compile_frame 捕获 → transform_code_object 重跑
def failed(self, tx):
    return self._failed                      # 第二次到达此点 → 直接 graph break
```

graph break 的结果:**把断点前的子图编译掉 + 生成一个 resume 函数**(从断点后的字节码继续),断点处回到 Python 解释器 eager 执行,之后 resume 函数再次被 Dynamo 拦截、续编译。`fullgraph=True` 时禁止 graph break,直接报错。

### 3.4 `OutputGraph`:建 `GraphModule` 并调用 backend

符号执行累积的节点最终在 `compile_subgraph`(`output_graph.py:1432`)→ `compile_and_call_fx_graph`(`:2076`)里组装成 `fx.GraphModule`,然后调用用户 backend:

```python
# _dynamo/output_graph.py:2147-2225 (精简)
gm = _make_graph_module(root, self.graph)                  # FX 图 → GraphModule
compiled_fn = self.call_user_compiler(gm, self.example_inputs())   # → 进入 _call_user_compiler

# _dynamo/output_graph.py:2412  ★ backend 在这一行真正被调用 ★
compiled_fn = compiler_fn(gm, example_inputs)
# inductor 时 compiler_fn == torch._inductor.compile_fx.compile_fx
```

> `output_graph.py:2412` 是「Dynamo 世界」与「后端世界」的分界线。backend 失败抛 `BackendCompilerFailed`,某些情况降级为 graph break。

---

## 4. Guards 与缓存 / 重编译

### 4.1 guard 的生成

`build_guards` → `CheckFunctionManager`(`guards.py:3563`)遍历 `output_graph.guards`,每个 `Guard` 调 `guard.create(builder)`,由 `GuardBuilder` 往 C++ guard 树添加节点:

| guard 类型     | 校验内容                              |
| -------------- | ------------------------------------- |
| `TENSOR_MATCH`   | tensor 的 dtype/device/shape/stride 等 |
| `TYPE_MATCH`     | 对象类型                              |
| `ID_MATCH`       | 对象 id 不变(并注册 weakref 用于失效) |
| `EQUALS_MATCH`   | 标量/值相等                           |
| `SHAPE_ENV`      | 动态 shape 的符号约束                 |
| `GLOBAL_STATE`   | `grad_enabled` 等全局状态               |

构建后会**立即自检**:`guard_manager.check(local_scope)` 必须为真,否则 "Guard failed on the same frame it was created"。

### 4.2 运行时检查与 recompile

每次调用编译函数:C 层遍历 `CacheEntry` 链表 → `guard_manager(f_locals)` 逐个校验:

- 命中 → 跑该 entry 的编译 code;
- 全部失败 → recompile,记录失败原因(`guards.py:4463` `get_and_maybe_log_recompilation_reasons`),新编译结果**追加**到链表(同一函数可有多份特化版本);
- `ID_MATCH` 对象被 GC → `weakref.finalize` 触发 `invalidate`,该 entry 失效。

> 典型 recompile 诱因:输入 shape 变了(且非 dynamic)、dtype 变了、传了不同的 Python 对象。链表长度超过 `cache_size_limit` 会告警并回退 eager。

---

## 5. AOTAutograd:把反向也纳入图

backend(inductor)拿到 Dynamo 的 `GraphModule` 后,**先过 AOTAutograd**。

### 5.1 为什么需要

- Dynamo 只抓**前向**;反向默认仍由 autograd engine 在 eager 跑,有大量 Python 调度开销。
- AOTAutograd 提前 trace 出 **joint(前向+反向)图**,让反向也能被 Inductor 编译。
- 再用 partitioner 把 joint 图**切成 fw/bw 两张图**,智能决定哪些激活值前向保存、哪些反向重算(rematerialization),权衡显存与算力。

### 5.2 两阶段流程

```
aot_module_simplified (aot_autograd.py:1030)
 ├─ Stage1 graph capture (_aot_autograd/graph_compile.py:172)
 │    └─ aot_dispatch_autograd_graph (graph_capture.py:363)
 │         ├─ create_joint (graph_capture_wrappers.py:274)   # 用 autograd.grad 包出反向
 │         ├─ create_functionalized_fn                       # 消除 inplace,变纯函数式
 │         └─ make_fx(...) + FunctionalTensorMode             # trace 出 joint FX 图
 └─ Stage2 compile (graph_compile.py:1968 aot_stage2_autograd)
      ├─ _aot_stage2a_partition  → partition_fn 切 joint → fw_module / bw_module
      ├─ _aot_stage2b_fw_compile → fw_compiler(fw_module)   # == compile_fx_forward
      ├─ _aot_stage2b_bw_compile → bw_compiler(bw_module)   # == compile_fx_backward
      └─ _aot_stage2c_make_autograd_function                # 组回一个 autograd.Function
```

`create_joint` 的关键是用 `torch.autograd.grad` 把反向算进来,并给前向节点打 `partitioner_tag="is_forward"`:

```python
# _functorch/_aot_autograd/graph_capture_wrappers.py:274 create_joint (精简)
def inner_fn(primals, tangents):
    outs = fn(*primals)                                  # 前向
    for node in mode.tracer.graph.nodes:
        node.meta["partitioner_tag"] = "is_forward"
    grad_primals = [p for p in primals if p.requires_grad]
    backward_out = torch.autograd.grad(needed_outs, grad_primals, needed_tangents)  # 反向
    return outs, backward_out
```

---

## 6. Inductor:FX 图 → Triton/C++ kernel

`compile_fx`(`_inductor/compile_fx.py:2457`)是 inductor backend 入口,内部编排 → 调 AOTAutograd → 对 fw/bw 各跑一遍 `compile_fx_inner`。

### 6.1 调 AOTAutograd

```python
# _inductor/compile_fx.py:2809 (精简)
return aot_autograd(
    fw_compiler=fw_compiler,        # = compile_fx_forward  (:2251)
    bw_compiler=bw_compiler,        # = compile_fx_backward (:2361)
    partition_fn=partition_fn,      # min_cut_rematerialization_partition
    decompositions=decompositions,
)(model_, example_inputs_)
```

### 6.2 单图编译:lowering → 调度 → codegen

`compile_fx_inner`(`:767`)→ `fx_codegen_and_compile`(`:1716`)→ `_InProcessFxCompile.codegen_and_compile`(`:1184`):

```
fake_tensor_prop                         (:1290)  推断 fake 元数据
_recursive_post_grad_passes              (:551)   pattern matching 等图优化
GraphLowering(gm)                        (:1418)  建 IR 容器
graph.run()                              (:1452)  ★ 解释执行 FX → 生成 Inductor IR (lowering)
graph.compile_to_module()                (:1537)  进入 codegen
```

**lowering**:`GraphLowering` 继承 `fx.Interpreter`,每个 FX 节点经 `call_function`(`graph.py:1259`)在 `lowerings` 表里查到对应 lowering 函数,把 `aten` 算子翻译成 IR:

```python
# _inductor/graph.py:1259 call_function (精简)
out = lowerings[target](*args, **kwargs)   # register_lowering 注册的翻译规则
```

### 6.3 Scheduler:算子融合

```python
# _inductor/scheduler.py:2763 _init (精简)
self.nodes = [self.create_scheduler_node(n) for n in nodes]
self.compute_dependencies()                    # 建依赖 DAG
self.nodes = self.topological_sort_schedule(self.nodes)
self.nodes = self.fuse_nodes(self.nodes)       # ★ 最多 10 轮迭代融合
```

`fuse_nodes`(`:3552`)把可水平/垂直融合的 `SchedulerNode` 合并成 `FusedSchedulerNode`——**一个融合节点 = 一个 Triton/C++ kernel**,这是 Inductor 性能的主要来源(减少 kernel 启动与显存往返)。

### 6.4 codegen + 编译加载

```python
# _inductor/graph.py:2350 codegen (精简)
self.init_wrapper_code()
self._update_scheduler()              # 创建 Scheduler(触发融合)
self.scheduler.codegen()              # 每个融合节点 → TritonScheduling/CPPScheduling
result = self.wrapper_code.generate() # 组装成完整 Python wrapper
```

- `TritonKernel.codegen_body`(`codegen/triton.py:4752`):拼 `tl.load`/compute/`tl.store` 成 kernel 体;
- `PythonWrapperCodegen.generate`(`codegen/wrapper.py:1714`):组装含 `@triton.jit` kernel 的 Python 文件;
- 最后写盘 + import,import 时 `@triton.jit` 触发 Triton JIT 编出 PTX/cubin:

```python
# _inductor/graph.py:2449 _compile_to_module_lines (精简)
key, path = PyCodeCache.write(wrapper_code.value)      # 写 .py 到 cache
mod = PyCodeCache.load_by_key_path(key, path, ...)     # import → 触发 Triton JIT
return mod                                              # 可直接调用的 CompiledModule
```

---

## 7. 端到端调用链总表(带行号)

```
torch.compile(model)                                   torch/__init__.py:2551
 └─ torch._dynamo.optimize(backend)(model)             torch/__init__.py:2756
     └─ _optimize → OptimizeContext                    _dynamo/eval_frame.py:1407 / 1046
         └─ set_eval_frame(callback)  [装 CPython 帧钩子] _dynamo/eval_frame.py:765
─────────────────────────  (运行时每个帧)  ─────────────────────────
 C 层: 遍历 CacheEntry → guard_manager.check(f_locals)
   命中 → 跑编译产物 ;  未命中 ↓ callback
 CatchErrorsWrapper                                    _dynamo/convert_frame.py:2208
  └─ ConvertFrame → ConvertFrameAssert                _dynamo/convert_frame.py:2053 / 736
      └─ _compile                                      _dynamo/convert_frame.py:1390
          ├─ compile_frame → InstructionTranslator.run _dynamo/symbolic_convert.py:1647
          │    └─ while step(): dispatch[opcode]       _dynamo/symbolic_convert.py:1291
          │         └─ Unsupported → graph break       _dynamo/symbolic_convert.py:236
          ├─ OutputGraph.compile_and_call_fx_graph     _dynamo/output_graph.py:2076
          │    └─ ★ compiler_fn(gm, example_inputs)    _dynamo/output_graph.py:2412
          │         └─ inductor: compile_fx            _inductor/compile_fx.py:2457
          │             └─ aot_autograd(...)           _inductor/compile_fx.py:2809
          │                 ├─ create_joint            _functorch/.../graph_capture_wrappers.py:274
          │                 ├─ partition → fw/bw        _functorch/.../graph_compile.py:1567
          │                 └─ compile_fx_inner(fw/bw)  _inductor/compile_fx.py:767
          │                     ├─ GraphLowering.run    _inductor/graph.py:1452
          │                     ├─ Scheduler.fuse_nodes _inductor/scheduler.py:3552
          │                     └─ codegen + PyCodeCache _inductor/graph.py:2350 / 2449
          └─ build_guards → GuardedCode                _dynamo/convert_frame.py:1566
 → C 层把 GuardedCode 插入该 code 的 CacheEntry 链表
```

---

## 8. 运行时快路径(第二次及以后)

```
调用 compiled_fn(*args)
  → C 帧钩子拦截
  → 遍历 CacheEntry 链表
       guard_manager.check(args):
         ├─ shape/dtype/type/id 全部匹配 → 命中
         │     → 直接执行编译字节码(内部调 Inductor kernel) → 返回   ★ 零 trace 开销
         └─ 某条 guard 失败
               → 继续下一个 entry;都失败 → recompile,新增一份特化版本
```

**一句话**:`torch.compile` 第一次把"这次输入长这样"的假设记成 guard 并编译;之后只要输入"长得一样"就直接复用,变了就再特化一份。理解 torch.compile 性能与 recompile 问题,本质就是理解 **guard 的粒度**。

---

## 9. 调试速查

| 想看什么            | 手段                                                              |
| ------------------- | ----------------------------------------------------------------- |
| 为什么 recompile    | `TORCH_LOGS=recompiles python x.py`                                 |
| graph break 在哪    | `TORCH_LOGS=graph_breaks` 或 `torch._dynamo.explain(fn)(*args)`        |
| 生成的 Triton 代码  | `TORCH_LOGS=output_code`(或 `TORCH_COMPILE_DEBUG=1`,产物在 `torch_compile_debug/`) |
| 捕获的 FX 图        | `TORCH_LOGS=graph_code`                                             |
| 禁用编译对比        | `torch.compiler.disable` / 环境变量 `TORCHDYNAMO_DISABLE=1`           |
| 强制不许 graph break| `torch.compile(fn, fullgraph=True)`                                 |

---

## 十一、实战常见坑点

### 1. graph break 静默降级
**现象**: 代码里加了 `torch.compile`，跑起来不报错但没加速。
**原因**: Dynamo 遇到不支持的操作时会 graph break —— 图切成多段，每段之间回退到 eager 执行。大量 graph break → compile 实际上只编译了少数 op。
**排查**:
```bash
TORCH_LOGS=graph_breaks python your_script.py
```
**解决**: 用 `torch.compile(fullgraph=True)` 强制要求零 graph break → 报错直接告诉你哪行不兼容。

### 2. recompile 风暴
**现象**: 训练每个 batch 都触发重编译，显存和速度都崩。
**原因**: 没开 `dynamic=True`，但每次 forward 的输入 shape 不同（如 padding 后 bucket 对齐）。
```bash
TORCH_LOGS=recompiles python your_script.py  # 看到 guard failure 原因
```
**解决**: `torch.compile(model, dynamic=True)` — shape 变为符号变量，一份 kernel 适配多个 size。

### 3. 编译后反向传播 NaN
**现象**: eager 模式下梯度正常，compile 后出现 NaN。
**原因**: Inductor 对浮点运算做重排（reassociation），改变了累加顺序 → 精度变化。常见于 bf16 + 大 reduction。
**解决**:
```python
# 方案 A：关闭重排 (torch >= 2.4)
torch._inductor.config.reorder_for_compute_comm_overlap = False
# 方案 B：对敏感 op 不做融合
torch._dynamo.config.cache_size_limit = 128  # 保守
```

### 4. CUDA Graph 与 DataLoader 多 worker 冲突
**现象**: compile + num_workers>0 时随机死锁。
**原因**: multiprocessing + CUDA context 初始化顺序问题。PyTorch 2.0-2.2 早期版本的已知问题。
**解决**:
```python
# 升级到 torch >= 2.3 或：
torch.multiprocessing.set_start_method("spawn")
# 或者 DataLoader 用 num_workers=0 先验证是否是此问题
```

### 5. 内存不降反升
**现象**: compile 后期望显存下降，实际 peak memory 更高。
**原因**: 编译后的 kernel 可能需要更多寄存器/spilling；融合后的临时 buffer 在 liveness 分析不精确时可能过早分配。
**排查**:
```python
# 看 compilation 的显存开销
torch._dynamo.config.accumulated_cache_size_limit = 64
TORCH_LOGS=+dynamo python your_script.py
```

### 6. 分布式 + compile 的调用顺序
**现象**: `DistributedDataParallel` + `torch.compile` 组合不 work。
**正确顺序**:
```python
model = MyModel().cuda()
model = torch.compile(model)  # compile 在 DDP 外面
model = DDP(model)
# 而不是 DDP(compile(model)) 或 compile(DDP(model))
```
DDP 需要 visibility 到未编译的梯度 → compile 必须在 DDP 外层。
