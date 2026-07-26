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

    # 阶段 3: torch.compile
    compiled = torch.compile(Tiny(), backend="inductor")
    compiled(x)
    print("\n阶段 3: torch.compile = Dynamo + Inductor (PyTorch 2.0, 2023)")
    print("  Dynamo 抓图 → Inductor 融合 → Triton/C++ kernel")


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
