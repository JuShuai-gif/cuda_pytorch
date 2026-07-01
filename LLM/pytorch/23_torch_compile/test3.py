"""Dynamo 源码分析: bytecode 拦截 → 符号执行 → guard → 图捕获全链路。

使用工具: dis.dis() / torch._dynamo.explain() / TORCH_LOGS /
         torch._dynamo.eval_frame / 自定义 bytecode 拦截

运行:
  python test3.py                  # 全链路分析
  python test3.py bytecode         # 查看 Python bytecode
  python test3.py dynamo           # Dynamo 符号执行 + FX图
  python test3.py guards           # Guard 条件生成
  python test3.py graph_break      # 追踪 graph break 原因

参考源码:
  torch/_dynamo/eval_frame.py     — set_eval_frame (CPython 帧拦截)
  torch/_dynamo/symbolic_convert.py — 符号执行引擎
  torch/_dynamo/guards.py         — guard 条件生成
  torch/_dynamo/variables/        — VariableTracker 子类
"""

import dis
import sys
import torch
import torch.nn as nn


# ============ 1. Python Bytecode 分析 ============
def exp_bytecode():
    """展示 Dynamo 拦截的原始 Python bytecode。"""
    print("=" * 60)
    print("1. Python Bytecode: Dynamo 拦截的入口")
    print("=" * 60)

    def simple_fn(x):
        y = x * 2
        z = y + 1
        return z.relu()

    print("  simple_fn 的 CPython bytecode:")
    dis.dis(simple_fn)

    print("\n  每条指令的含义:")
    for instr in dis.get_instructions(simple_fn):
        print(f"    {instr.offset:4d} {instr.opname:12s} {instr.argrepr}")

    print("\n  Dynamo 如何工作:")
    print("  1. set_eval_frame() 替换 CPython 的帧求值函数")
    print("  2. 每次调用 compiled 函数 → Dynamo 逐条读取 bytecode")
    print("  3. 对每条指令做符号执行 (SymbolicVariable)")
    print("  4. 遇到支持的 op → 记录到 FX 图")
    print("  5. 遇到不支持的 op → graph break → 回退到 eager")
    print()


# ============ 2. Dynamo 符号执行跟踪 ============
def exp_dynamo():
    """使用 torch._dynamo.explain 查看 graph break 和编译过程。"""
    print("=" * 60)
    print("2. Dynamo explain: 图捕获过程分析")
    print("=" * 60)

    def fn(x):
        a = x * 2
        b = a + 1
        # data-dependent control flow → graph break
        if b.sum() > 0:
            b = b.relu()
        c = b * 3
        return c.sum()

    x = torch.randn(4)

    # 使用 explain 查看 graph break
    explanation = torch._dynamo.explain(fn)(x)
    print(f"  Graph count:       {explanation.graph_count}")
    print(f"  Graph break count: {explanation.graph_break_count}")
    print(f"  Break reasons:")
    for i, (graph, reason) in enumerate(
        zip(explanation.graphs, explanation.break_reasons)
    ):
        print(f"    Graph {i}: break because {reason}")
        print(f"      ops: {[n.name for n in graph.graph.nodes]}")

    print("\n  Dynamo 内部流程:")
    print("  1. InstructionTranslator 逐条翻译 bytecode")
    print("  2. 遇到 if → VariableTracker 符号执行条件")
    print("  3. 条件依赖 tensor 值 → 无法静态判断 → graph break")
    print("  4. Dynamo 保存已捕获的图, 让当前 chunk 在 eager 执行")
    print()


# ============ 3. Guard 条件分析 ============
def exp_guards():
    """查看 Dynamo 生成的 guard 条件。"""
    print("=" * 60)
    print("3. Guard 条件: 决定是否需要 recompile")
    print("=" * 60)

    @torch.compile
    def fn(x):
        return x * 2 + 1

    # 第一次调用: 编译 + 生成 guard
    x1 = torch.randn(4)
    fn(x1)

    # 第二次: shape 相同, guard 通过 → 复用编译结果
    x2 = torch.randn(4)
    fn(x2)

    # 第三次: shape 不同, guard 失败 → recompile
    x3 = torch.randn(8)
    fn(x3)

    print("  Guard 检查的内容 (从源代码 torch/_dynamo/guards.py 生成):")
    print("  - tensor.shape:  是否为已知形状")
    print("  - tensor.dtype:  数据类型是否一致")
    print("  - tensor.device: 设备是否一致")
    print("  - tensor.stride: 步幅是否一致 (用于 contiguous 判断)")
    print("  - tensor.layout: 布局 (strided/sparse/... )")
    print("  - requires_grad: 梯度标志")
    print("  - 全局状态:      torch.is_grad_enabled() 等")
    print()
    print("  Guard 类型:")
    print("  - TENSOR_MATCH:  张量身份匹配 (同一个 Python 对象)")
    print("  - SHAPE_MATCH:   形状匹配")
    print("  - NO_HASATTR:    禁止动态属性访问")
    print("  - GLOBAL_STATE:  如 torch.is_grad_enabled()")
    print("  - DATA_PTR:      数据指针 (CUDA Graph 需要)")
    print()

    # 查看 recompile 日志
    print("  运行 TORCH_LOGS=+recompiles 可以看到 guard failure 详情:")
    print("    export TORCH_LOGS='+recompiles'")
    print("    python -c '...'  # 会打印 guard 失败的具体原因")
    print()


# ============ 4. FX 图内部结构 ============
def exp_fx_internals():
    """深入查看 FX Graph 的节点结构。"""
    print("=" * 60)
    print("4. FX Graph 内部节点: placeholder/call_function/output")
    print("=" * 60)

    def fn(x):
        a = x + 1
        b = a.relu()
        return b * 2

    gm = torch.fx.symbolic_trace(fn)
    x = torch.randn(4)

    print("  FX Graph 节点类型和元数据:")
    for node in gm.graph.nodes:
        print(f"    {node.op:>15s} | {node.name:10s} | target={node.target!r}")
        print(
            f"       args={[a.name if hasattr(a, 'name') else str(a)[:30] for a in node.args]}"
        )
        print(f"       kwargs={node.kwargs}")
        print(f"       users={[u.name for u in node.users]}")
        if "val" in node.meta:
            print(f"       fake_val={node.meta['val']}")
        print()

    print("  节点类型说明:")
    print("  placeholder:     图输入 (函数参数)")
    print("  get_attr:        模块属性访问 (self.weight)")
    print("  call_function:   Python 函数调用 (torch.add)")
    print("  call_module:     子模块调用 (self.fc)")
    print("  call_method:     方法调用 (tensor.relu)")
    print("  output:          图输出")
    print()

    # GraphModule 内部 Python 代码
    print("  GraphModule 生成的 Python 代码:")
    print("  --- code ---")
    print(gm.code)
    print("  ------------")
    print()


# ============ 5. set_eval_frame 最小复现 ============
def exp_eval_frame():
    """最小复现 Dynamo 的 CPython 帧拦截机制。"""
    print("=" * 60)
    print("5. set_eval_frame: CPython 帧拦截机制 (最小复现)")
    print("=" * 60)

    # Dynamo 本质: 替换 CPython 的帧求值钩子
    original = torch._C._get_eval_frame()

    def my_callback(frame, cache_size):
        """自定义帧回调 — 打印被调用的函数, 然后回退到默认求值。"""
        code = frame.f_code
        print(
            f"  Dynamo intercepted: {code.co_name} "
            f"in {code.co_filename.split('/')[-1]}:{code.co_firstlineno}"
        )
        # 你可以在这里做任意分析!
        return None  # None = 跳过 Dynamo, 原生执行

    try:
        torch._C._set_eval_frame(my_callback)

        @torch.compile
        def test_fn(x):
            return x * 2

        # 调用时触发 my_callback
        test_fn(torch.randn(4))
        print()

    finally:
        torch._C._set_eval_frame(original)

    print("  torch._C._set_eval_frame 的实现:")
    print("  C++ 端: torch/csrc/dynamo/eval_frame.cpp")
    print("  Python 端: torch/_dynamo/eval_frame.py:set_eval_frame")
    print("  核心: 替换 CPython 解释器的 _PyInterpreterFrame.eval_frame 函数指针")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_bytecode()
        exp_dynamo()
        exp_fx_internals()
        exp_guards()
        exp_eval_frame()

    print("[Dynamo source analysis] DONE")


if __name__ == "__main__":
    main()
