"""
06_architecture.py — torch.compile 全链路源码级架构

以 fn(x) = sigmoid(relu(linear(x))) + 1 为例，
展示每个阶段的算法原理、可运行代码和 PyTorch 源码位置。

运行:
    python 06_architecture.py
"""

import torch
import torch.nn as nn
import torch.fx
import dis


# ═══════════════════════════════════════════════════════════════
# 辅助工具
# ═══════════════════════════════════════════════════════════════


def section(title):
    print(f"\n{'█' * 65}\n█  {title}\n{'█' * 65}")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 16)

    def forward(self, x):
        return torch.sigmoid(torch.relu(self.fc(x))) + 1.0


# ═══════════════════════════════════════════════════════════════
# ① Dynamo: 字节码 → FX Graph
# ═══════════════════════════════════════════════════════════════

section("① Dynamo: 字节码拦截 → FX Graph")

# ── 1.1 看字节码 ──
print("\nDynamo 看到的第一手材料: CPython 字节码（不是 Python 源码）")
print("\n  forward() 的字节码:")
dis.dis(Model().forward)

print("""
  原理: PEP 523 (Python 3.6+) 允许替换帧评估函数:
    _PyInterpreterState_SetEvalFrameFunc(dynamo_eval_frame)

  此后每次调用被编译函数的 forward 时，
  CPython 不再执行默认的 ceval.c 主循环，
  而是进入 Dynamo 的 eval_frame:
    → 创建 InstructionTranslator
    → 逐条读取字节码: dispatch_table[opcode](self, inst)
    → 符号执行 → 构建 FX Graph

  源码:
    torch/_dynamo/eval_frame.py → optimize():L1392, _optimize_catch_errors():L1210
    torch/_dynamo/convert_frame.py → _compile():L1390, compile_inner():L1423
    torch/_dynamo/symbolic_convert.py → InstructionTranslator (核心类)
    torch/_dynamo/bytecode_transformation.py → 字节码分析辅助
""")

# ── 1.2 VariableTracker ──
print("""
  符号执行的关键: VariableTracker

  Python 值              → VariableTracker (存符号信息，不存真实值)
  ─────────────────────────────────────────────────────────────
  torch.Tensor           → TensorVariable
    - .shape = (4, 8)
    - .dtype = float32
    - .device = cuda:0
    - 没有 .data, 没有真实数值！

  torch.relu             → UserFunctionVariable
    - .target = torch.relu
    - .source = <built-in method relu>

  42, 3.14, "hello"      → ConstantVariable

  nn.Linear(8, 16)       → UnspecializedNNModuleVariable
    - .parameters = {weight: (16,8), bias: (16,)}

  源码:
    /home/ghr/code/pytorch/torch/_dynamo/variables/base.py
      VariableTracker     L320   ← 所有 VT 的基类, call_function() L677
    /home/ghr/code/pytorch/torch/_dynamo/variables/tensor.py
      TensorVariable      L180   ← Tensor 的符号表示 (fake tensor)
    /home/ghr/code/pytorch/torch/_dynamo/variables/functions.py
      UserFunctionVariable L552  ← torch.relu 等函数
    /home/ghr/code/pytorch/torch/_dynamo/variables/nn_module.py
      NNModuleVariable            ← nn.Linear 等模块
    /home/ghr/code/pytorch/torch/_dynamo/variables/builder.py
      wrap_fx_proxy()     L3593  ← 在 Graph 创建 Node + 返回新 VT

  关键代码路径:
    TensorVariable 调用 torch.relu 时:
      1. UserFunctionVariable.call_function(tx, [TensorVariable], {})
      2. → 检查 target 是否 PyTorch 函数
      3. → tx.output.create_proxy("call_function", target, args)
      4. → 在 FX Graph 创建: Node(name="relu", op="call_function", target=relu)
      5. → 返回新 TensorVariable，指向这个 Node
""")

# ── 1.3 实际运行看 FX Graph ──
torch._dynamo.reset()


def show_fx(gm, inputs):
    print("\n  Dynamo 输出的 FX Graph:")
    for n in gm.graph.nodes:
        t = n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
        inps = [a.name if isinstance(a, torch.fx.Node) else repr(a) for a in n.args]
        print(f"    [{n.op:14}] {n.name:<8} target={t:<10}  inputs={inps}")
    return gm.forward


c = torch.compile(Model(), backend=show_fx)
c(torch.randn(4, 8))

# ── 1.4 Guards ──
print("""
  Guards: 编译后附带的假设条件，失效则重编译

  类型           检查内容                         示例
  ─────────────────────────────────────────────────────────
  TENSOR_MATCH   shape/dtype/device               x.shape == (4, 8)
  TYPE_MATCH     isinstance check                 type(x) == Tensor
  ID_MATCH       id(obj) 没变                     id(torch.relu) == 0x7f...
  GLOBAL_STATE   grad_mode/default_device         torch.is_grad_enabled()
  MODULE_MATCH   模块类引用没变                    type(model.fc) == Linear

  每次调用 compiled_fn(x):
    1. 跑 guards check
    2. 通过 → 直接执行编译后的 kernel
    3. 不通过 → recompile

  源码:
    /home/ghr/code/pytorch/torch/_dynamo/guards.py
      GuardBuilder        L1219  ← 五种 guard 的生成逻辑
      TENSOR_MATCH()      L3502  ← shape/dtype/device guard
      compile_check_fn()  L4827  ← 生成最终的 check 函数
    /home/ghr/code/pytorch/torch/_dynamo/convert_frame.py
      _compile()          L1633  ← 连接 Dynamo 抓图 + backend 编译
""")

# ── 1.5 Graph Break ──
print("""
  Graph Break: 遇到不能符号执行的代码

  触发条件:
    ✗ 数据依赖 if (if y.sum() > 0)    → Dynamo 不知道条件真假
    ✗ .item() (val = x.sum().item())  → Tensor 逃逸为 Python 标量
    ✗ print(), 文件 IO                → 不是 PyTorch 操作
    ✗ numpy/scipy 调用                → 逃逸出 PyTorch
    ✗ Python list/dict 操作           → 符号追踪无法处理

  处理方式:
    结束当前子图 → eager 执行 break 点代码 → Dynamo 重新启动抓下一张

  详细诊断: 见 05_graph_break.py
""")


# ═══════════════════════════════════════════════════════════════
# ② AOTAutograd: autograd 提前追踪
# ═══════════════════════════════════════════════════════════════

section("② AOTAutograd: 提前追踪 autograd 反向")

print("""
  Dynamo 输出的 FX Graph 只有前向计算。
  训练时需要反向传播梯度，AOTAutograd 负责提前 (AOT) 追踪出来。

  三个子步骤:

  1. Functionalization — in-place → 纯函数
     x.relu_() → y = torch.relu(x)
     x.add_(1) → y = x + 1
     原因: 纯函数图更容易分析、融合、优化

  2. 构建 Joint Graph — 前向 + 反向合一
     用 torch.autograd.grad 或 functorch.vjp 追踪反向
     sin(cos(x)) 的反向:
       前向: y = sin(cos(x))
       反向: grad = d(sin)/d(cos) * d(cos)/dx = cos(cos(x)) * (-sin(x))

  3. Partitioner — min-cut 切分
     把 Joint Graph 切分成:
       forward graph:  input → output + 反向需要的中间值
       backward graph: grad_output + 中间值 → grad_input
     目标: 最小化需要保存给反向的 tensor 总显存

  源码:
    /home/ghr/code/pytorch/torch/_functorch/aot_autograd.py
      aot_module_simplified()          L1132  ← AOTAutograd 入口
    /home/ghr/code/pytorch/torch/_functorch/_aot_autograd/graph_capture.py
      aot_dispatch_autograd_graph()    L471   ← autograd 图 + dispatch
    /home/ghr/code/pytorch/torch/_subclasses/functional_tensor.py
      FunctionalTensor                 L120   ← 函数化 Tensor 包装
    /home/ghr/code/pytorch/torch/_functorch/partitioners.py
      min_cut_rematerialization_partition() L3725 ← min-cut 切分

  推理时 (with torch.no_grad()) 跳过此阶段，直接进入 Inductor。
""")

# ── 演示: AOTAutograd 输出 ──
torch._dynamo.reset()


def show_aot(gm, inputs):
    print("\n  AOTAutograd 处理后的图 (Inductor 之前):")
    for n in gm.graph.nodes:
        t = n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
        print(f"    {n.op:<14} {n.name:<8} → {t}")
    return gm.forward


c_aot = torch.compile(Model(), backend=show_aot)
c_aot(torch.randn(4, 8))
print("  → 经过了 functionalization，还没有 Inductor 的融合")


# ═══════════════════════════════════════════════════════════════
# ③ Inductor: FX Graph → Triton/C++ kernel
# ═══════════════════════════════════════════════════════════════

section("③ Inductor: FX Graph → 编译后的 kernel")

print("""
  三个子步骤:

  1. Lowering: aten ops → Inductor IR
     对 FX Graph 中每个 aten Node:
       aten.linear → ExternKernel(addmm)  调用 cuBLAS
       aten.relu   → Pointwise
       aten.sigmoid → Pointwise
       aten.add    → Pointwise

     源码:
       /home/ghr/code/pytorch/torch/_inductor/lowering.py
         register_lowering()   L535    ← 注册 aten → IR 映射表
       /home/ghr/code/pytorch/torch/_inductor/ir.py
         Pointwise    L1173   ← 逐元素操作 (relu, sigmoid, add)
         Reduction    L1337   ← 归约操作 (sum, mean)
         Buffer       L5087   ← 代表一个 tensor buffer
         ExternKernel L6878   ← 外部库操作 (cuBLAS matmul)

  2. Scheduler + Fusion:
     BFS 遍历 IR 依赖图:
       连续 pointwise → 融合成一个 kernel
       matmul + pointwise → epilogue fusion (Triton)
       matmul + matmul → 不融合 (两个 cuBLAS 调用)

     本例: relu + sigmoid + add → fused kernel
           linear(matmul) → 单独一个 kernel (extern)

     源码:
       /home/ghr/code/pytorch/torch/_inductor/scheduler.py
         Scheduler  L4129     ← 调度器主类
         fuse()     L2674     ← 融合算法 (BFS + 依赖分析)

  3. Codegen: Inductor IR → Triton/C++ 源码

     GPU (Triton):
       @triton.jit
       def fused_add_relu_sigmoid_kernel(in_ptr, out_ptr, ...):
           x = tl.load(in_ptr + offsets)
           x = tl.where(x > 0, x, 0)       # relu
           x = 1.0 / (1.0 + tl.exp(-x))    # sigmoid
           x = x + 1.0                     # add
           tl.store(out_ptr + offsets, x)

     CPU (C++):
       auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr);
       auto tmp1 = at::vec::clamp_min(tmp0, 0);           // relu
       auto tmp2 = 1/(1 + tmp1.neg().exp());              // sigmoid
       tmp2 + 1.0;                                        // add

     源码:
       /home/ghr/code/pytorch/torch/_inductor/codegen/triton.py
         TritonKernel   L3131   ← GPU kernel 生成主类
         codegen_body()  L5903   ← 生成 kernel 函数体 (tl.load / tl.store)
       /home/ghr/code/pytorch/torch/_inductor/codegen/cpp.py
         CppKernel      L2081   ← CPU kernel 生成主类
       /home/ghr/code/pytorch/torch/_inductor/codecache.py
         FxGraphCache   ← 编译缓存管理 (hash key → .cubin/.so)
""")

# ── 演示: 查看 Inductor 输出 ──
print("\n  查看 Inductor 生成的 kernel 源码:")
print("    TORCH_LOGS='output_code' python your_script.py")
print("    TORCH_COMPILE_DEBUG=1 python your_script.py  # dump 到 disk")
print("  或运行: python 04_compile.py  (自动生成到 _compile_artifacts/)")
print("  源码中查看: torch/_inductor/codegen/triton.py → codegen_body()")


# ═══════════════════════════════════════════════════════════════
# ④ Triton 编译 + ⑤ 执行
# ═══════════════════════════════════════════════════════════════

section("④⑤ Triton 编译 + 执行")

print("""
  ④ Triton 编译管线:
    @triton.jit Python 代码
      → Triton IR (TTIR)        # 通用 MLIR-like IR
      → Triton GPU IR (TTGIR)   # GPU 特定优化 (swizzling, pipelining)
      → LLVM IR
      → PTX (*.ptx)             # NVIDIA 虚拟指令集
      → .cubin                  # GPU 可执行二进制

    缓存: ~/.triton/cache/

  ⑤ 执行 compiled_fn(x):

    def compiled_fn(x):
        # 1) guard check
        if not self.guards.check(x.shape, x.dtype, x.device, ...):
            return self.recompile_and_call(x)

        # 2) (可选) CUDA Graph 包裹
        if self.use_cuda_graph:
            return cuda_graph_exec(x)

        # 3) 直接启动
        kernel_1.launch(grid, block, stream, x, buf0, buf1)
        return buf1

    guard 通过 → 直接执行 → 后续调用都不再编译
    guard 不通过 → recompile → 新 shape/dtype 产生新的编译结果

    mode="reduce-overhead":
      把整个执行序列 capture 成 CUDA Graph
      后续每步只 launch 1 次 cudaGraphLaunch
      和 CUDA/Graph/cuda_graph_demo.cu 原理一样
""")


# ═══════════════════════════════════════════════════════════════
# 完整调用栈
# ═══════════════════════════════════════════════════════════════

section("完整调用栈: compiled_fn(x) 从入口到 GPU")

print("""
  compiled_fn(x_tensor)
    │
    ├─ torch/_dynamo/eval_frame.py        [guard check]
    │   _optimize_catch_errors → check_guards()
    │
    ├─ [首次 / recompile]
    │   │
    │   ├─ torch/_dynamo/symbolic_convert.py    [Dynamo 抓图]
    │   │   InstructionTranslator.step()
    │   │   → 逐条字节码 → VariableTracker.call_function()
    │   │
    │   ├─ torch/_dynamo/variables/builder.py    [构建 FX Node]
    │   │   wrap_fx_proxy() → create_proxy()
    │   │
    │   ├─ torch/_functorch/_aot_autograd/       [AOTAutograd]
    │   │   aot_module_simplified() → joint_graph → partition
    │   │
    │   └─ torch/_inductor/compile_fx.py         [Inductor]
    │       compile_fx():
    │         → lowering.py  (aten → IR)
    │         → scheduler.py (fusion)
    │         → codegen/     (Triton/C++)
    │         → triton.compile → .cubin
    │
    └─ [执行]
        kernel.launch(grid, block, stream, args)
""")


# ═══════════════════════════════════════════════════════════════
# 源码速查
# ═══════════════════════════════════════════════════════════════

section("源码速查表（带精确行号）")

SRC = "/home/ghr/code/pytorch/torch"

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ 源码根目录: {SRC}/
│ PyTorch 版本: 源码库 (git log -1 可查)
└─────────────────────────────────────────────────────────────────────┘

── 阶段 ① Dynamo ──

  入口与帧评估钩子:
    {SRC}/_dynamo/eval_frame.py
      optimize()                        L1672    ← torch.compile 的入口
      _optimize_catch_errors()          L1479    ← 帧钩子 + 异常包装

  字节码拦截与符号执行:
    {SRC}/_dynamo/symbolic_convert.py
      InstructionTranslator             L5336    ← 核心类，逐条字节码
      step() 方法                       类内部    ← dispatch_table[opcode]() 分发

    {SRC}/_dynamo/convert_frame.py
      _compile()                        L1633    ← 编译入口，连接 Dynamo+backend

  VariableTracker (符号世界的变量):
    {SRC}/_dynamo/variables/base.py
      VariableTracker                   L320     ← 所有 VT 的基类
      call_function()                   L677     ← 符号执行函数调用

    {SRC}/_dynamo/variables/tensor.py
      TensorVariable                    L180     ← Tensor 的符号表示 (fake tensor)

    {SRC}/_dynamo/variables/functions.py
      UserFunctionVariable              L552     ← torch.relu 等函数

    {SRC}/_dynamo/variables/builder.py
      wrap_fx_proxy()                   L3593    ← 在 Graph 创建 Node + 返回 VT

  FX Graph 构建:
    {SRC}/_dynamo/output_graph.py
      OutputGraph                       L734     ← 管理 FX Graph 构建
      create_proxy()                    L1424    ← 创建 Node 的核心调用

  Guards (守卫条件):
    {SRC}/_dynamo/guards.py
      GuardBuilder                      L1219    ← 生成 guard 检查代码
      TENSOR_MATCH()                    L3502    ← shape/dtype/device guard
      compile_check_fn()                L4827    ← 生成最终的 check 函数

  FX 图的基础数据结构:
    {SRC}/fx/graph.py
      Graph                             L1384    ← 图的容器 (双向链表)
      create_node()                     L1572    ← 创建节点的公共 API
      python_code()                     L2448    ← 图 → Python 代码

    {SRC}/fx/proxy.py
      Proxy                             L600     ← 假 Tensor，重载运算符


── 阶段 ② AOTAutograd ──

  入口:
    {SRC}/_functorch/aot_autograd.py
      aot_module_simplified()           L1132    ← AOTAutograd 入口

  Autograd 追踪:
    {SRC}/_functorch/_aot_autograd/graph_capture.py
      aot_dispatch_base_graph()         L283     ← 基础图调度
      aot_dispatch_autograd_graph()     L471     ← autograd 图追踪 + dispatch

  Functionalization (in-place → 纯函数):
    {SRC}/_subclasses/functional_tensor.py
      FunctionalTensor                  L120     ← 函数化 Tensor 包装

  Partitioner (前向/反向切分):
    {SRC}/_functorch/partitioners.py
      MinCutOptions                     L158     ← min-cut 配置
      min_cut_rematerialization_partition() L3725  ← min-cut 算法


── 阶段 ③ Inductor ──

  入口:
    {SRC}/_inductor/compile_fx.py
      compile_fx_inner()                L794     ← Inductor 核心编译
      compile_fx()                      L2798    ← Inductor backend 入口

  Lowering (aten → Inductor IR):
    {SRC}/_inductor/lowering.py
      register_lowering()               L535     ← 注册 aten → IR 映射

  IR 数据结构:
    {SRC}/_inductor/ir.py
      Pointwise                         L1173    ← 逐元素操作 (relu, sigmoid, add)
      Reduction                         L1337    ← 归约操作 (sum, mean)
      Buffer                            L5087    ← 代表一个 tensor buffer
      ExternKernel                      L6878    ← 外部库操作 (cuBLAS, cuDNN)

  Scheduler (算子融合):
    {SRC}/_inductor/scheduler.py
      Scheduler                         L4129    ← 调度器主类
      fuse()                            L2674    ← 融合算法

  Codegen (代码生成):
    {SRC}/_inductor/codegen/triton.py
      TritonKernel                      L3131    ← GPU Triton kernel 生成
      codegen_body()                    L5903    ← 生成 kernel 函数体

    {SRC}/_inductor/codegen/cpp.py
      CppKernel                         L2081    ← CPU C++ kernel 生成


── 环境变量 ──

  TORCH_LOGS="graph_breaks,recompiles"      看 break 位置 + 重编译原因
  TORCH_LOGS="+dynamo,inductor,output_code" 看全过程 + 生成的 kernel 源码
  TORCH_COMPILE_DEBUG=1                     dump 所有中间产物到 disk
  TORCHINDUCTOR_CACHE_DIR=./cache           编译缓存目录
""")


if __name__ == "__main__":
    pass  # 所有演示已在上方 print 和实际编译中完成
