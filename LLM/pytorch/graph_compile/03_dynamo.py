"""
03_dynamo.py — Dynamo 抓图：字节码拦截 → FX Graph

核心机制:
  1. PEP 523: 替换 CPython 帧评估函数，拦截 Python 字节码
  2. VariableTracker: 符号世界的变量（不是真实 tensor）
  3. 逐条字节码模拟执行 → 是 PyTorch 操作就创建 FX Node
  4. 不是 PyTorch 操作 → graph break
"""

import torch
import dis


# ═══════════════════════════════════════════════════════════════
# Part 1: 看 Dynamo 的输入 —— CPython 字节码
# ═══════════════════════════════════════════════════════════════


def demo_bytecode():
    def fn(x):
        a = torch.cos(x)
        b = torch.sin(a)
        return b

    print("=" * 60)
    print("Dynamo 的输入: CPython 字节码")
    print("=" * 60)
    print("\nfn(x) = sin(cos(x))")
    print("Python 源码 → 字节码（Dynamo 逐条处理）:\n")
    dis.dis(fn)
    print("""
Dynamo 逐条处理:
  LOAD_GLOBAL torch  → 加载 torch 模块
  LOAD_ATTR cos      → 获取 cos 函数
  LOAD_FAST x        → 从参数取出 x → 创建 placeholder Node
  CALL 1             → 调用 cos(x) → 创建 call_function Node
  STORE_FAST a       → 存入符号局部变量
  ... 对 sin 重复 ...
  RETURN_VALUE       → 创建 output Node
""")


# ═══════════════════════════════════════════════════════════════
# Part 2: 自定义 backend 截获 Dynamo 的输出
# ═══════════════════════════════════════════════════════════════


def demo_capture():
    """
    自定义 backend = 一个函数 (gm, example_inputs) → callable。
    在 Dynamo 之后、Inductor 之前被调用。
    """

    print("=" * 60)
    print("自定义 backend 截获 Dynamo 的 FX Graph")
    print("=" * 60)

    def fn(x):
        a = x @ x.T
        b = torch.relu(a)
        return torch.sigmoid(b)

    gm_list = []

    def inspect(gm, inputs):
        gm_list.append(gm)
        print(f"\n  ▸ 子图 {len(gm_list)} ({len(list(gm.graph.nodes))} nodes)")
        for n in gm.graph.nodes:
            if n.op != "output":
                t = (
                    n.target.__name__
                    if hasattr(n.target, "__name__")
                    else str(n.target)
                )
                print(f"      {n.op:<14} {n.name:<8} → {t}")
        return gm.forward

    torch._dynamo.reset()
    f = torch.compile(fn, backend=inspect)
    result = f(torch.randn(4, 4))

    print(
        f"\n  Dynamo 抓图: {' → '.join([n.name for n in gm_list[0].graph.nodes if n.op != 'output'])}"
    )

    # backend="inductor" 和自定义 backend 的区别
    print(f"""
  backend 参数的本质:
    backend=inspect  → 收到 FX Graph，打印后返回 gm.forward（不优化）
    backend="inductor" → 收到 FX Graph，做 lowering+fusion+codegen（编译）

    两者收到的是同一张 FX Graph，签名都是 (gm, inputs) → callable
    """)


# ═══════════════════════════════════════════════════════════════
# Part 3: Dynamo 逐字节码追踪过程
# ═══════════════════════════════════════════════════════════════


def demo_step_by_step():
    print("=" * 60)
    print("Dynamo 逐字节码符号执行详解")
    print("=" * 60)

    trace = [
        ("—— 初始化 ——", "创建空的 FX Graph，符号栈: []", ""),
        (
            "LOAD_FAST x",
            "x 是函数入参 → 创建 placeholder Node",
            "→ 压入 TensorVariable(x)。栈: [TV(x)]",
        ),
        (
            "LOAD_FAST x (第二次)",
            "同一个 x → 取出已有 TV(x)",
            "栈: [TV(x), TV(x)]  ← 两个引用指向同一 placeholder",
        ),
        (
            "LOAD_ATTR T",
            "访问 .T 属性 → 创建 getattr Node",
            "→ 压入 TV(x.T)。栈: [TV(x), TV(x.T)]",
        ),
        (
            "BINARY_OP @",
            "创建 matmul Node(x, x.T)",
            "→ 压入 TV(matmul)。栈: [TV(matmul)]",
        ),
        (
            "STORE_FAST a",
            "local['a'] = TV(matmul)。栈: []",
            "局部变量表: {a: TV(matmul)}",
        ),
        (
            "LOAD_GLOBAL torch, LOAD_ATTR relu",
            "加载 torch.relu → UserFunctionVariable",
            "→ 压入 UFV(relu)。栈: [UFV(relu)]",
        ),
        (
            "LOAD_FAST a, CALL 1",
            "取出 TV(matmul) → 调用 UFV(relu)",
            "→ 创建 call_function Node(relu, args=[matmul])",
        ),
        ("", "→ 压入 TV(relu_result)。栈: [TV(relu)]", ""),
        (
            "... (sigmoid 同理) ...",
            "创建 call_function Node(sigmoid, args=[relu])",
            "→ 压入 TV(sigmoid_result)",
        ),
        ("RETURN_VALUE", "创建 output Node → 子图完成！", ""),
    ]

    for instr, what, detail in trace:
        if instr.startswith("——"):
            print(f"\n  {instr}\n    {what}")
        elif instr == "":
            print(f"    {what}")
        else:
            print(f"\n  {instr}")
            print(f"    → {what}")
            if detail:
                print(f"    → {detail}")


# ═══════════════════════════════════════════════════════════════
# Part 4: graph break 演示
# ═══════════════════════════════════════════════════════════════


def demo_graph_break():
    print("\n" + "=" * 60)
    print("Graph break: 遇到不能追踪的代码就会断开")
    print("=" * 60)

    sc = [0]

    def count(gm, inputs):
        sc[0] += 1
        ops = [n.name for n in gm.graph.nodes if n.op != "output"]
        print(f"  子图 {sc[0]}: {' → '.join(ops)}")
        return gm.forward

    def fn_with_print(x):
        y = x @ x.T
        print(f"sum={y.sum().item():.2f}")  # ← graph break
        return torch.relu(y)

    torch._dynamo.reset()
    f = torch.compile(fn_with_print, backend=count)
    f(torch.randn(4, 4))

    print(f"\n  print() 不是 PyTorch 操作 → Dynamo 断开")
    print(f"  → {sc[0]} 张子图（print 前一张，print 后重新抓一张）")
    print("  详细 graph break 诊断见 05_graph_break.py")


if __name__ == "__main__":
    demo_bytecode()
    demo_capture()
    demo_step_by_step()
    demo_graph_break()
