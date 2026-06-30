"""Autograd 源码分析: backward 图构造, SavedVariable, grad_fn 链。

使用工具: grad_fn / .next_functions / torch.autograd.grad /
         detect_anomaly / backward hook

运行:
  python test3.py                  # 全链路分析
  python test3.py graph_build      # backward graph 构造过程
  python test3.py saved_variable   # SavedVariable 原理
  python test3.py grad_fn_chain    # grad_fn 链遍历
  python test3.py detect_nan       # NaN 梯度追踪

参考源码:
  torch/csrc/autograd/engine.cpp   — backward 引擎
  torch/csrc/autograd/variable.h   — AutogradMeta
  torch/csrc/autograd/saved_variable.cpp — SavedVariable
"""

import sys
import torch
import torch.nn as nn


# ============ 1. Backward Graph 构造过程 ============
def exp_graph_build():
    """追踪 backward 图是如何逐步构建的。"""
    print("=" * 60)
    print("1. Backward Graph: grad_fn 链的构造过程")
    print("=" * 60)

    x = torch.tensor([2.0], requires_grad=True)
    print(f"  x: grad_fn={x.grad_fn}  (leaf tensor — None)")

    y = x * 3
    print(f"  y = x*3: grad_fn={y.grad_fn}")
    print(f"    type: {type(y.grad_fn).__name__}")
    print(f"    next_functions[0] → x: {y.grad_fn.next_functions[0][0] is None}")

    z = y + 1
    print(f"  z = y+1: grad_fn={z.grad_fn}")
    print(f"    type: {type(z.grad_fn).__name__}")
    print(f"    next_functions[0] → y: {z.grad_fn.next_functions[0][0] is y.grad_fn}")

    w = z.relu()
    print(f"  w = z.relu(): grad_fn={w.grad_fn}")
    print(f"    type: {type(w.grad_fn).__name__}")

    # 双向 op (matmul) 的 grad_fn 有多个输入
    a = torch.tensor([[1.0, 2.0]], requires_grad=True)
    b = torch.tensor([[1.0], [2.0]], requires_grad=True)
    c = a @ b  # matmul
    print(f"\n  c = a@b: grad_fn={c.grad_fn}")
    print(f"    type: {type(c.grad_fn).__name__}")
    for i, (fn, _) in enumerate(c.grad_fn.next_functions):
        print(
            f"    next_functions[{i}]: {type(fn).__name__ if fn is not None else 'AccumulateGrad'}"
        )

    print(f"\n  grad_fn 链遍历 (从 w 反向到 x):")

    def walk_grad_fn(tensor, depth=0):
        fn = tensor.grad_fn
        while fn is not None:
            print(f"    {'  ' * depth}{type(fn).__name__}")
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    print(f"    {'  ' * (depth + 1)}-> {type(next_fn).__name__}")
            # Move to first non-leaf next function for demo
            next_fns = [n for n, _ in fn.next_functions if n is not None]
            fn = next_fns[0] if next_fns else None
            depth += 1

    walk_grad_fn(w)
    print()


# ============ 2. SavedVariable 原理 ============
def exp_saved_variable():
    """分析 SaveVariable 如何决定保存/丢弃中间变量。"""
    print("=" * 60)
    print("2. SavedVariable: 哪些中间值被保存, 哪些被丢弃")
    print("=" * 60)

    # 使用 backward hook 查看 saved tensors
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    z = x * y  # MulBackward0: 需要保存 x 和 y 用于 backward
    loss = z.sum()

    # 在 backward 时查看 saved_tensors
    def inspect_saved(grad_output):
        fn = z.grad_fn
        if hasattr(fn, "saved_tensors"):
            for i, st in enumerate(fn.saved_tensors):
                print(
                    f"    saved_tensors[{i}]: shape={list(st.shape)} dtype={st.dtype}"
                )

    z.register_hook(inspect_saved)
    loss.backward()

    print(f"  MulBackward 保存了 x 和 y 用于 backward")
    print(f"  x.grad = {x.grad}  (d(loss)/dx = y)")
    print(f"  y.grad = {y.grad}  (d(loss)/dy = x)")

    print(f"\n  SavedVariable 保存规则:")
    print(f"  - 被 backward kernel 需要的输入 → 保存")
    print(f"  - 不需要的 → 丢弃 (节省显存)")
    print(f"  - 可通过 pack / unpack hooks 自定义")
    print(f"  - ctx.save_for_backward() 在自定义 Function 中")
    print(f"  - 底层: torch/csrc/autograd/saved_variable.cpp")
    print()


# ============ 3. grad_fn 链完整遍历 ============
def exp_grad_fn_chain():
    """完整遍历一个复杂计算图的 grad_fn 链。"""
    print("=" * 60)
    print("3. grad_fn 链: 完整遍历 + 查看每个函数的元信息")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    x = torch.randn(2, 4, requires_grad=True)
    y = model(x)
    loss = y.sum()

    # 遍历 chain
    def walk(fn, prefix=""):
        """递归遍历 grad_fn 树。"""
        if fn is None:
            return
        name = type(fn).__name__
        print(f"  {prefix}{name}")

        if hasattr(fn, "next_functions"):
            for i, (child, _) in enumerate(fn.next_functions):
                if child is not None:
                    walk(child, prefix + "  ")
                elif i == len(fn.next_functions) - 1:
                    break
                else:
                    print(f"  {prefix}  [leaf AccumulateGrad]")

    print("  Backward graph tree:")
    walk(loss.grad_fn)

    # 统计节点类型
    from collections import Counter

    type_counts = Counter()

    def count_types(fn):
        if fn is None:
            return
        type_counts[type(fn).__name__] += 1
        if hasattr(fn, "next_functions"):
            for child, _ in fn.next_functions:
                count_types(child)

    count_types(loss.grad_fn)
    print(f"\n  Backward graph 节点类型统计:")
    for t, c in type_counts.most_common():
        print(f"    {t:30s}: {c}")
    print()


# ============ 4. NaN 梯度追踪 ============
def exp_detect_nan():
    """使用 detect_anomaly 追踪 NaN 梯度来源。"""
    print("=" * 60)
    print("4. NaN 梯度追踪: torch.autograd.detect_anomaly")
    print("=" * 60)

    # 构造会产生 NaN 梯度的计算
    x = torch.tensor([0.0, 0.0], requires_grad=True)

    # 安全计算
    y_safe = x * 2 + 1
    loss_safe = y_safe.sum()
    loss_safe.backward()
    print(f"  y = x*2+1, x.grad = {x.grad}  (安全)")

    x.grad = None
    # 产生 NaN 的计算: 对 0 取 log
    with torch.autograd.detect_anomaly(check_nan=True):
        try:
            y_nan = torch.log(x.abs())  # log(0) = -inf
            loss_nan = y_nan.sum()
            loss_nan.backward()
        except RuntimeError as e:
            lines = str(e).split("\n")
            print(f"\n  detect_anomaly 捕获 NaN:")
            for line in lines[:5]:
                print(f"    {line[:120]}")

    print(f"\n  内置 detect_anomaly 原理:")
    print(f"  - 在每次 backward op 执行后检查 grad 是否包含 NaN")
    print(f"  - 包含 NaN → 立即抛出 RuntimeError + 完整 stack trace")
    print(f"  - 性能损失: 约 10-20% (生产环境不建议常开)")
    print(f"  - 调试用: detect_anomaly(check_nan=True)")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_graph_build()
        exp_saved_variable()
        exp_grad_fn_chain()
        exp_detect_nan()

    print("[Autograd source analysis] DONE")


if __name__ == "__main__":
    main()
