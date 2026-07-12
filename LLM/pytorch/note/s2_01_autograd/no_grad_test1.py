"""torch.no_grad / enable_grad / inference_mode demo + 源码分析。

Companion script for no_grad/no_grad.md.
  1. basic no_grad:        装饰器 vs 上下文管理器
  2. requires_grad trace:  追踪 no_grad 如何影响 tensor
  3. enable_grad:          no_grad 内局部恢复
  4. inference_mode:       vs no_grad 的区别
  5. dispatch key检查:     确认 Autograd key 是否被排除
  6. custom autograd fn:   no_grad 在 Function.forward 中的规范

Run:
    python test1.py                # full demo
    python test1.py basic          # 基础用法
    python test1.py trace          # requires_grad 追踪链
    python test1.py enable         # enable_grad 局部恢复
    python test1.py inference      # inference_mode 对比
    python test1.py dispatch       # DispatchKey 检查
"""

import sys
import torch
import torch.nn as nn


# ============ 1. 装饰器 vs 上下文管理器 ============
def exp_basic():
    print("=" * 60)
    print("1. @no_grad() 装饰器 vs with no_grad(): 完全等价")
    print("=" * 60)

    x = torch.tensor([2.0, 3.0], requires_grad=True)

    # 方式 A: 上下文管理器
    with torch.no_grad():
        y_a = x * 2
    print(f"  with no_grad(): y.requires_grad = {y_a.requires_grad}")
    print(f"                   y.grad_fn = {y_a.grad_fn}")

    # 方式 B: 装饰器
    @torch.no_grad()
    def fn(x):
        return x * 2

    y_b = fn(x)
    print(f"  @no_grad():      y.requires_grad = {y_b.requires_grad}")
    print(f"                   y.grad_fn = {y_b.grad_fn}")

    # Factory 函数是例外
    with torch.no_grad():
        p = nn.Parameter(torch.randn(3))
    print(f"\n  Factory 函数例外: Parameter.requires_grad = {p.requires_grad}")
    print(f"  → nn.Parameter() 显式设置 requires_grad=True, 不受 no_grad 影响")
    print()


# ============ 2. requires_grad 追踪链 ============
def exp_trace():
    print("=" * 60)
    print("2. requires_grad 传播: no_grad 如何切断计算图")
    print("=" * 60)

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # 正常: grad_fn 链完整
    y1 = x * 2
    z1 = y1 + 1
    print(f"  正常路径:")
    print(
        f"    y1.requires_grad = {y1.requires_grad}, grad_fn = {type(y1.grad_fn).__name__}"
    )
    print(
        f"    z1.requires_grad = {z1.requires_grad}, grad_fn = {type(z1.grad_fn).__name__}"
    )

    # no_grad: 链断裂
    with torch.no_grad():
        y2 = x * 2
        z2 = y2 + 1
    print(f"\n  no_grad 路径:")
    print(f"    y2.requires_grad = {y2.requires_grad}, grad_fn = {y2.grad_fn}")
    print(f"    z2.requires_grad = {z2.requires_grad}, grad_fn = {z2.grad_fn}")

    # 验证: no_grad 内的操作用了 no_grad 外的输入, 也不会建图
    # 但 no_grad 外的操作看到 no_grad 的输出是 requires_grad=False → 也不建图
    z3 = z2 + 5  # z2 来自 no_grad, requires_grad=False
    print(f"\n  no_grad 后使用: z3 = z2 + 5")
    print(f"    z3.requires_grad = {z3.requires_grad}, grad_fn = {z3.grad_fn}")
    print(f"  → 一旦断了, 后续就全断了")
    print()


# ============ 3. enable_grad 局部恢复 ============
def exp_enable():
    print("=" * 60)
    print("3. enable_grad: 在 no_grad 内部局部恢复梯度")
    print("=" * 60)

    x = torch.tensor([2.0], requires_grad=True)

    @torch.no_grad()
    def mixed_fn(x):
        a = x * 2  # no_grad → 不追踪
        with torch.enable_grad():  # 局部恢复 Autograd key
            b = x * 3  # enable_grad → 追踪!
        c = a + b  # b 有 grad_fn, a 没有
        return c

    y = mixed_fn(x)
    print(f"  a (no_grad):     requires_grad=False")
    print(f"  b (enable_grad): requires_grad=True")
    print(f"  c = a + b:        requires_grad={y.requires_grad}")
    print(f"                    grad_fn={type(y.grad_fn).__name__}")
    print(f"  → b 有 grad_fn, a 没有, 但 y = a + b 时, 加法只对 b 建立 grad_fn")

    # 嵌套恢复
    with torch.no_grad():
        a = x * 2
        with torch.enable_grad():
            b = x * 3
            with torch.no_grad():
                c = x * 4  # 又关了
        d = a + b + c  # b 有 grad_fn

    print(f"\n  嵌套示例:")
    print(f"    a: requires_grad={a.requires_grad}  (外层 no_grad)")
    print(f"    b: requires_grad={b.requires_grad}  (enable_grad 恢复)")
    print(f"    c: requires_grad={c.requires_grad}  (内层 no_grad 又关了)")
    print()


# ============ 4. inference_mode ============
def exp_inference():
    print("=" * 60)
    print("4. inference_mode: 比 no_grad 更强 (不可被 enable_grad 覆盖)")
    print("=" * 60)

    x = torch.tensor([5.0], requires_grad=True)

    with torch.inference_mode():
        y = x * 2
        print(f"  inference_mode 内: y.requires_grad = {y.requires_grad}")

        # inference_mode 内 enable_grad 会报错!
        try:
            with torch.enable_grad():
                z = x * 3
        except RuntimeError as e:
            print(f"  enable_grad in inference_mode: {type(e).__name__}")

    # inference_mode 同 no_grad: 支持 @装饰器
    @torch.inference_mode()
    def infer(x):
        return x * 2

    print(f"\n  inference_mode vs no_grad 区别:")
    print(f"  |                    | no_grad | inference_mode |")
    print(f"  |--------------------|---------|----------------|")
    print(f"  | Autograd key       | 排除    | 排除 + 禁用 fallback |")
    print(f"  | enable_grad 可恢复 | ✓       | ✗              |")
    print(f"  | view 版本检查      | ✓       | ✗ (更快)       |")
    print(f"  | 适用场景           | eval    | production 推理 |")
    print()


# ============ 5. DispatchKey 检查 ============
def exp_dispatch():
    print("=" * 60)
    print("5. DispatchKey 检查: no_grad 如何排除 Autograd key")
    print("=" * 60)

    x = torch.tensor([1.0], requires_grad=True)

    # 正常: Autograd key 在 key set 中
    y_normal = x * 2
    keys_normal = str(torch._C._dispatch_keys(y_normal))
    print(f"  正常:  Autograd in keys = {'Autograd' in keys_normal}")
    print(f"         所有 keys = {keys_normal.split(', ')[:3]}...")

    # no_grad: Autograd key 被排除
    with torch.no_grad():
        y_ng = x * 2
        keys_ng = str(torch._C._dispatch_keys(y_ng))
        print(f"\n  no_grad: Autograd in keys = {'Autograd' in keys_ng}")
        print(f"           所有 keys = {keys_ng.split(', ')[:3]}...")

    # 检查 disabled 状态
    print(f"\n  torch.is_grad_enabled() = {torch.is_grad_enabled()}")
    with torch.no_grad():
        print(f"  torch.is_grad_enabled() (in no_grad) = {torch.is_grad_enabled()}")

    # C++ 端对应: tls_set_dispatch_key_excluded(Autograd, !enabled)
    # Python 端 set_grad_enabled(False) → C++ 端设置 TLS exclude
    print(f"\n  源码链路:")
    print(f"  Python: torch.set_grad_enabled(False)")
    print(f"    → torch._C._set_grad_enabled(False)")
    print(f"      → c10::impl::tls_set_dispatch_key_excluded(Autograd, true)")
    print(f"        → 线程局部 DispatchKeySet 中排除 Autograd key")
    print(f"          → 所有后续算子 dispatch 跳过 autograd 包装")
    print()


# ============ 6. autograd.Function 中的规范 ============
def exp_custom_fn():
    print("=" * 60)
    print("6. autograd.Function.forward 中必须用 no_grad?")
    print("=" * 60)

    # PyTorch 已经保证: autograd.Function.forward 运行时,
    # Autograd key 已被自动排除 → 你不需要手动加 no_grad
    class MyFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # 这里 Autograd key 已被自动排除
            # → x * 2 不会建图
            print(f"    forward 内 is_grad_enabled = {torch.is_grad_enabled()}")
            ctx.save_for_backward(x)
            return x * 2

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * 2

    x = torch.tensor([3.0], requires_grad=True)
    y = MyFunc.apply(x)
    print(f"  y.requires_grad = {y.requires_grad}")
    print(f"  → autograd.Function.forward 自动在 no_grad 环境中运行")

    # 但如果你在 backward 中做计算, 需要手动检查:
    # backward 中也已经在 no_grad 中 (不需要建图, 只需要计算梯度)
    print(f"\n  规范:")
    print(f"  - forward: PyTorch 自动 no_grad → 你不需要加")
    print(f"  - backward: PyTorch 自动 no_grad → 你不需要加")
    print(f"  - 但在 backward 内部创建新 tensor 时, 注意不需要 requires_grad")
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "trace": exp_trace,
    "enable": exp_enable,
    "inference": exp_inference,
    "dispatch": exp_dispatch,
    "custom_fn": exp_custom_fn,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[no_grad demo] DONE")


if __name__ == "__main__":
    main()
