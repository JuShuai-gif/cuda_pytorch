"""
00_overview.py — torch.compile 全景图

三个核心角色:
  FX      = 图的格式（Graph + Node，纯 Python，可读写）
  Dynamo  = 抓图的人（PEP 523 字节码拦截 → 生成 FX Graph）
  Inductor = 吃图的人（FX Graph → Triton/C++ kernel）

torch.compile = Dynamo(前) + AOTAutograd(中) + Inductor(后)

四层"图"概念:
  Computation Graph  → 数学运算的抽象描述
  CUDA Graph         → GPU 操作序列化，减少 launch 开销
  FX Graph           → PyTorch 算子级中间表示，编译器吃的 IR
  Inductor IR        → 编译器内部表示，lowering → fusion → codegen
"""


# ═══════════════════════════════════════════════════════════════
# 1. 三种图的对比
# ═══════════════════════════════════════════════════════════════


def demo_graph_types():
    print("=" * 60)
    print("三种「图」的关系")
    print("=" * 60)
    print("""
  Computation Graph:
    节点 = 数学运算 (matmul, relu, ...)
    作用层 = 逻辑/概念
    优化目标 = 减少显存读写

  CUDA Graph:
    节点 = GPU 操作 (kernel, memcpy, ...)
    作用层 = GPU 驱动
    优化目标 = 减少 CPU launch 开销

  FX Graph:
    节点 = PyTorch 算子 (aten.linear, aten.relu, ...)
    作用层 = Python 编译器 IR
    优化目标 = 算子融合 + 代码生成

  三者关系:
    torch.compile 把粗粒度的 Python 代码转成 FX Graph
    → Inductor 把 FX Graph 融合成细粒度的 Triton kernel
    → (可选) 用 CUDA Graph 包裹以减少 launch 开销
""")


# ═══════════════════════════════════════════════════════════════
# 2. 演进史：Eager → TorchScript → FX → Dynamo → compile
# ═══════════════════════════════════════════════════════════════


def demo_evolution():
    import torch
    import torch.nn as nn
    import torch.fx

    print("=" * 60)
    print("PyTorch 计算图演进")
    print("=" * 60)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            return torch.relu(self.fc(x))

    model = Tiny()
    x = torch.randn(2, 4)

    # 阶段 0: Eager
    print("\n阶段 0: Eager 模式 (PyTorch 1.0+ 一直存在)")
    print("  写一行跑一行，调试方便，无图")

    # 阶段 1: TorchScript
    scripted = torch.jit.script(model)
    print("\n阶段 1: TorchScript (PyTorch 1.0, 2018)")
    print("  静态图，C++ 可部署，语法受限")

    # 阶段 2: FX
    traced = torch.fx.symbolic_trace(Tiny())
    print("\n阶段 2: FX (PyTorch 1.8, 2021)")
    print(f"  图是 Python 对象，{len(list(traced.graph.nodes))} 个节点，可读可改写")
    traced.graph.print_tabular()
    # FX 图 4 种 op 类型:
    #   placeholder    = 函数输入 (x)
    #   call_module    = self.xxx(...), 有 nn.Module 包装, 图里保存模块引用
    #   call_function  = torch.relu(...), 纯函数, 无参数无状态
    #   output         = 返回值
    # self.linear = nn.Linear(...) → call_module, 有 weight/bias 可学习参数
    # torch.relu 没有 nn.Module 包装 → call_function
    # 若写成 self.relu = nn.ReLU(); y = self.relu(y), relu 也是 call_module
    #
    # symbolic_trace 抓图方式: 静态分析 Python 源代码 (AST 级别)
    #   局限: 不支持动态控制流 (如 if x.shape[0] > 2: ...)
    #   Dynamo 通过字节码级 tracing (PEP 523 frame 拦截) 解决了这个问题,
    #   遇到动态分支会用 guard 回退到 eager, 覆盖面更广.
    #   两者输出的都是 torch.fx.GraphModule, FX 是共同的 IR.

    # 阶段 3: torch.compile
    compiled = torch.compile(Tiny(), backend="inductor")
    compiled(x)
    print("\n阶段 3: torch.compile = Dynamo + Inductor (PyTorch 2.0, 2023)")
    print("  Dynamo 抓图 → Inductor 融合 → Triton/C++ kernel")
    # Dynamo 抓到的图本质上还是 FX Graph (torch.fx.GraphModule).
    # 与 symbolic_trace 的关键区别:
    #   symbolic_trace: 读 Python 源码 → AST → 静态推导图
    #   Dynamo:        截获 CPython 字节码执行 → 运行时记录图
    # Inductor 拿到 FX Graph 后做 lowering + fusion + codegen,
    # 最终生成 Triton/C++ kernel. 所以 torch.compile pipeline 中
    # FX 依然是核心 IR.


# ═══════════════════════════════════════════════════════════════
# 3. 模拟 compile pipeline
# ═══════════════════════════════════════════════════════════════


def demo_pipeline():
    import torch

    def fn(x):
        return torch.sigmoid(torch.relu(x))

    print("\n" + "=" * 60)
    print("torch.compile = Dynamo + Inductor 全链路")
    print("=" * 60)

    print("\n  def fn(x):")
    print("      return torch.sigmoid(torch.relu(x))")
    print()
    print("    ↓ ① Dynamo (PEP 523 字节码拦截)")
    print("  FX Graph: x → relu → sigmoid → output")
    print()
    print("    ↓ ② AOTAutograd (推理时跳过)")
    print()
    print("    ↓ ③ Inductor (lowering → fusion → codegen)")
    print("  一个 Triton kernel: load x → max(x,0) → sigmoid → store")
    print()
    print("    ↓ ④ 执行")
    print("  guard check → cudaLaunchKernel → 返回结果")

    # 实际编译演示
    torch._dynamo.reset()

    def show(gm, inputs):
        ops = [n.name for n in gm.graph.nodes if n.op != "output"]
        print(f"\n  Dynamo 实际输出的图: {' → '.join(ops)}")
        return gm.forward

    c = torch.compile(fn, backend=show)
    c(torch.randn(100).cuda())


# ═══════════════════════════════════════════════════════════════
# 4. Dynamo 源码级调用链: torch.compile → PEP 523 → FX Graph
# ═══════════════════════════════════════════════════════════════
#
# torch.compile(fn)
#   └─ _TorchDynamoContext.__call__()            # eval_frame.py:783
#        └─ 返回 compile_wrapper，内部注册 PEP 523 帧回调
#
# compile_wrapper(*args)
#   └─ set_eval_frame(callback)                  # eval_frame.py:950, PEP 523 hook
#        └─ fn(*args, **kwargs)                  # 触发 CPython 帧执行
#             └─ callback(frame) 被调用
#                  └─ ConvertFrame.__call__()    # convert_frame.py:1914
#                       └─ _compile()            # convert_frame.py:1390
#                            ├─ InstructionTranslator(code, ...)  # 创建
#                            └─ tracer.run()                      # 执行
#
# InstructionTranslator.run()                   # symbolic_convert.py:1647
#   └─ while self.step(): ...                   # 逐字节码指令循环
#        └─ self.dispatch_table[opcode](self, inst)  # :1334, 按 opcode 分发
#             ├─ LOAD_FAST     → 读局部变量
#             ├─ CALL_FUNCTION → 记录算子到 self.output.graph (fx.Graph)
#             ├─ RETURN_VALUE  → 图完成
#             └─ 不支持的 op   → graph break, 退回到 eager
#
# dispatch_table 生成机制:
#   元类 OpcodeDispatcherMeta (symbolic_convert.py:996) 在类创建时扫描
#   dis.opmap 中所有字节码指令, 找出类上同名方法填入 dispatch_table[256] 数组.
#   step() 中 dispatch_table[inst.opcode](self, inst) 直接数组查表分发.
#
# 关键源码位置:
#   torch/_C/_dynamo/eval_frame.pyi → C 扩展   PEP 523 set_eval_frame hook
#   torch/_dynamo/eval_frame.py:783             torch.compile 装饰器入口
#   torch/_dynamo/convert_frame.py:1390         帧→FX Graph 编排层
#   torch/_dynamo/symbolic_convert.py:1291      step() 逐字节码分发
#   torch/_dynamo/output_graph.py:2950          self.graph = torch.fx.Graph()
#   torch/_dynamo/guards.py                     守卫系统: 缓存复用判定
#
# ═══════════════════════════════════════════════════════════════
# FX Graph = 统一 IR, 前后端解耦
# ═══════════════════════════════════════════════════════════════
#
# symbolic_trace(通过 AST) 和 Dynamo(通过字节码) 虽然抓图手段不同,
# 但输出都是 torch.fx.GraphModule. 后续所有环节只认这个格式:
#
#   symbolic_trace(AST 解析)
#              ↘
#           FX Graph → 后端(Inductor / 自定义 backend / 图变换)
#              ↗
#      Dynamo(字节码拦截)
#
# 后端不关心图是谁抓的, 只管拿到 FX Graph 后做 lowering → fusion → codegen.
# 这就是 FX 作为统一 IR 的价值: 前端可换, 后端可插.


if __name__ == "__main__":
    demo_graph_types()
    demo_evolution()
    demo_pipeline()

    print("""
╔══════════════════════════════════════════════════════╗
║  一句话：                                            ║
║  FX = 图的格式    Dynamo = 抓图的人                  ║
║  Inductor = 吃图的人    torch.compile = 三者的胶水   ║
║                                                    ║
║  学习路径:                                          ║
║   00_overview.py       ← 全景图（本文件）            ║
║   01_fx_graph.py       ← FX 基础 + 手动构建         ║
║   02_fx_internals.py   ← FX 底层数据结构            ║
║   03_dynamo.py         ← Dynamo 字节码抓图          ║
║   04_compile.py        ← backend/mode + 实际产物     ║
║   05_graph_break.py    ← graph break 诊断修复        ║
║   06_architecture.py   ← 全链路源码级架构            ║
╚══════════════════════════════════════════════════════╝
""")
