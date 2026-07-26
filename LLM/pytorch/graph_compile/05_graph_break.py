"""
Dynamo 抓图：graph break 完整手册

包含：
  一、12 种 graph break 场景 + 触发原因
  二、如何 debug graph break（工具和命令）
  三、如何消除每种 graph break（修复方案）

关键前提：Dynamo 在符号层面模拟执行，没有真实 tensor 值。
         它无法做出需要真实值的决策，遇到这类代码就 graph break。
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

gm_count = [0]
_break_results = []


def _check(gm, inputs):
    gm_count[0] += 1
    return gm.forward


def run_test(fn, x=None):
    """测试一个函数触发多少子图"""
    torch._dynamo.reset()
    gm_count[0] = 0
    inp = x if x is not None else torch.randn(100)
    f = torch.compile(fn, backend=_check)
    f(inp)
    return gm_count[0]


# ═══════════════════════════════════════════════════════════════
# 第一部分：12 种 graph break 场景
# ═══════════════════════════════════════════════════════════════


def part1_all_breaks():
    print("=" * 65)
    print("一、12 种 Graph Break 场景")
    print("=" * 65)

    tests = []

    # [1] 数据依赖 if
    def b1(x):
        y = torch.cos(x)
        return y * 2 if y.sum() > 0 else y

    tests.append(
        ("[1] 数据依赖 if", b1, "if y.sum() > 0 → Dynamo 不知道条件真假 → graph break")
    )

    # [2] 数据依赖循环
    def b2(x):
        y = torch.cos(x)
        s = y[0].item()
        for _ in range(int(min(abs(s), 3))):
            y = y + 1.0
        return y

    tests.append(
        ("[2] 数据依赖循环", b2, "range 依赖 tensor 值 → Dynamo 不知道循环几次")
    )

    # [3] .item()
    def b3(x):
        y = torch.cos(x)
        val = y.sum().item()
        return y + val

    tests.append(
        ("[3] .item() 逃逸", b3, "tensor → Python 标量 → Dynamo 失去对值的追踪")
    )

    # [4] .cpu() / .numpy()
    def b4(x):
        import numpy as np

        y = torch.cos(x)
        return torch.from_numpy(np.abs(y.cpu().numpy()))

    tests.append(("[4] numpy 调用", b4, "数据逃逸出 PyTorch → Dynamo 无法继续符号追踪"))

    # [5] print
    def b5(x):
        y = torch.cos(x)
        print(f"m={y.mean().item():.4f}")
        return torch.sin(y)

    tests.append(("[5] print 副作用", b5, "print 不是 PyTorch 操作 → graph break"))

    # [6] Python 数据结构
    def b6(x):
        y = torch.cos(x)
        lst = [y, y + 1]
        return lst[0] + lst[1]

    tests.append(("[6] list 构造", b6, "Python list 包含 tensor → Dynamo 部分追踪"))

    # [7] 数据依赖索引
    def b7(x):
        y = torch.cos(x)
        idx = (y > 0.5).nonzero(as_tuple=True)[0]
        return y[idx].sum() if idx.numel() > 0 else y.sum()

    tests.append(("[7] 数据依赖索引", b7, "tensor 作为 index → 索引值依赖运行时数据"))

    # [8] try/except
    def b8(x):
        try:
            return torch.cos(x) if x.sum() < -1e9 else torch.sin(x)
        except Exception:
            return x

    tests.append(("[8] try/except", b8, "异常处理中带数据依赖分支 → graph break"))

    # [9] generator / yield
    def b9(x):
        y = torch.cos(x)
        for i in range(3):
            return y + i

    tests.append(("[9] 复杂控制流", b9, "return 在循环内 → 控制流复杂"))

    # [10] shape 变化 → recompile
    print(f"\n[10] 输入 shape 变化 → 重编译（不是 graph break 但类似效果）")

    class M10(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return torch.relu(self.fc(x))

    model = M10()
    torch._dynamo.reset()
    cmodel = torch.compile(model, backend=_check)
    cmodel(torch.randn(5, 10))
    n1 = gm_count[0]
    gm_count[0] = 0
    cmodel(torch.randn(10, 10))
    n2 = gm_count[0]
    tests.append(
        (
            "[10] shape 变化 recompile",
            lambda x: x,
            f"shape 变了 → guard 失效 → 重新编译 (子图 {n1}→{n2})",
        )
    )

    # [11] 非 tensor 输入类型
    print(f"\n[11] 非 tensor 参数 → Dynamo 可能 graph break")

    def b11(x, flag: bool):
        return torch.cos(x) if flag else torch.sin(x)

    n = run_test(lambda x: b11(x, True))
    tests.append(("[11] bool 参数分支", b11, f"Dynamo 对常量控制流可处理 (子图: {n})"))

    # [12] autograd Function
    print(f"\n[12] 自定义 autograd Function")

    class MyFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        def backward(ctx, g):
            return g * 2

    def b12(x):
        return MyFunc.apply(x)

    n = run_test(b12)
    tests.append(
        ("[12] 自定义 autograd", b12, f"torch.autograd.Function 通常可追踪 (子图: {n})")
    )

    # 汇总
    print("\n" + "=" * 65)
    print("Graph Break 汇总表")
    print("=" * 65)
    for name, fn, desc in tests:
        if (
            name.startswith("[10]")
            or name.startswith("[11]")
            or name.startswith("[12]")
        ):
            print(f"  {name:<28} {desc}")
        else:
            n = run_test(fn)
            tag = "✗ BREAK" if n > 1 else "✓ OK"
            print(f"  {name:<28} {n} 子图 {tag}")
        _break_results.append((name, desc))


# ═══════════════════════════════════════════════════════════════
# 第二部分：如何 debug graph break
# ═══════════════════════════════════════════════════════════════


def part2_debug():
    print("\n\n" + "=" * 65)
    print("二、如何 Debug Graph Break")
    print("=" * 65)

    print("""
[方法 1] TORCH_LOGS="graph_breaks"
  最常用的调试方式，直接打印 break 位置和原因:
    $ TORCH_LOGS="graph_breaks" python script.py
    Graph break: from user code at:
      File "xx.py", line 42, in forward
        if y.sum() > 0:     ← 精确行号

[方法 2] torch._dynamo.explain(fn)(*args)
  在代码中直接检查，不需要环境变量:
    explanation = torch._dynamo.explain(fn)(*args)
    print(explanation.graph_break_count)   # 几次 break
    print(explanation.break_reasons)       # 每次 break 原因

[方法 3] TORCH_COMPILE_DEBUG=1
  把所有中间产物 dump 到磁盘:
    TORCH_COMPILE_DEBUG=1 python script.py
    产物: torch_compile_debug/run_xxx/
      fx_graph_runnable.py     Dynamo 抓出的 FX Graph
      output_code.py           Inductor 生成的 Triton kernel

[方法 4] 自定义 backend 逐段二分定位
  不知道哪行 break？把函数切成两半，分别 compile 看:
    @torch.compile(backend=inspect)
    def first_half(x): return cos(sin(x))
    @torch.compile(backend=inspect)
    def second_half(x): return relu(x)

[方法 5] 看全部编译日志
    TORCH_LOGS="+dynamo,inductor,graph_breaks,recompiles" python script.py
""")


# ═══════════════════════════════════════════════════════════════
# 第三部分：如何消除每种 graph break
# ═══════════════════════════════════════════════════════════════


def part3_fixes():
    print("\n\n" + "=" * 65)
    print("三、消除 Graph Break 的方法")
    print("=" * 65)

    # [修复 1]
    print("\n[修复 1] 数据依赖 if/else → torch.where")
    print("  if y.sum() > 0: y = y * 2  →  y = torch.where(y.sum() > 0, y * 2, y)")

    def b1(x):
        return torch.cos(x) * 2 if torch.cos(x).sum() > 0 else torch.cos(x)

    def g1(x):
        y = torch.cos(x)
        return torch.where(y.sum() > 0, y * 2, y)

    print(f"    修复前: {run_test(b1)} 子图  |  修复后: {run_test(g1)} 子图")

    # [修复 2]
    print("\n[修复 2] .item() → capture_scalar_outputs=True")
    print("  torch._dynamo.config.capture_scalar_outputs = True")
    torch._dynamo.config.capture_scalar_outputs = True

    def b2(x):
        val = torch.cos(x).sum().item()
        return x + val

    print(f"    开启后: {run_test(b2)} 子图")
    torch._dynamo.config.capture_scalar_outputs = False

    # [修复 3]
    print("\n[修复 3] print → 移到 compile 外部或用 TORCH_LOGS")
    print("  把 print 从 forward 里移出去，调试用 TORCH_LOGS 看中间值")

    # [修复 4]
    print("\n[修复 4] numpy → torch 等价操作")
    print("  np.abs(y) → torch.abs(y)   np.where(c,a,b) → torch.where(c,a,b)")

    def g4(x):
        return torch.abs(torch.cos(x))

    print(f"    修复后 torch 版: {run_test(g4)} 子图")

    # [修复 5]
    print("\n[修复 5] Python for/while → 向量化 tensor 操作")
    print("  for i in range(n): y+=x[i]  →  y = x.sum()")

    def g5(x):
        y = torch.cos(x)
        return y + y.sum()

    print(f"    修复后向量化: {run_test(g5)} 子图")

    # [修复 6]
    print("\n[修复 6] 动态 shape 重编译 → dynamic=True 或 mark_dynamic")
    print("  torch.compile(fn, dynamic=True)  # 标记 batch 维度动态")
    print("  torch._dynamo.mark_dynamic(x, 0)  # 标记第 0 维动态")

    # [修复 7]
    print("\n[修复 7] 自定义函数 → torch.compiler.allow_in_graph")
    print("""
  @torch.compiler.allow_in_graph
  def my_custom_op(x):
      return some_custom_kernel(x)  # Dynamo 现在能追踪了
""")


# ═══════════════════════════════════════════════════════════════
# 第四部分：速查表
# ═══════════════════════════════════════════════════════════════


def part4_cheatsheet():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              Graph Break Debug + Fix 速查表                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  查看 break 位置和原因:                                           ║
║    TORCH_LOGS="graph_breaks" python script.py                    ║
║                                                                  ║
║  在 Python 代码中检查:                                            ║
║    torch._dynamo.explain(fn)(*args)                              ║
║                                                                  ║
║  dump 所有中间产物:                                               ║
║    TORCH_COMPILE_DEBUG=1 python script.py                        ║
║                                                                  ║
║  修复方案:                                                        ║
║    if/else           → torch.where / torch.cond                 ║
║    .item()           → capture_scalar_outputs=True              ║
║    print/IO          → 移出 compile / TORCH_LOGS                ║
║    numpy/scipy       → torch 等价函数                            ║
║    Python for/while  → 向量化 tensor 操作                        ║
║    dynamic shape     → dynamic=True / mark_dynamic              ║
║    自定义函数         → torch.compiler.allow_in_graph            ║
║    自定义 C++ 扩展    → torch.library.register_fake             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    part1_all_breaks()
    part2_debug()
    part3_fixes()
    part4_cheatsheet()
