"""Dispatcher 源码分析: dispatch 链追踪, TLS 操作, 算子注册表探查。

使用工具: torch._C._dispatch_* API / torch.library / 自定义 dispatch key

运行:
  python test3.py                  # 全链路分析
  python test3.py dispatch_trace   # 追踪 dispatch 链
  python test3.py tls_inside       # TLS dispatch key 操作
  python test3.py kernel_lookup    # 查看算子注册了哪些 kernel

参考源码:
  aten/src/ATen/core/dispatch/Dispatcher.h    — 调度器
  aten/src/ATen/core/dispatch/OperatorEntry.h — 算子条目
  c10/core/DispatchKey.h                       — DispatchKey 枚举
"""

import sys
import torch


# ============ 1. Dispatch 链追踪 ============
def exp_dispatch_trace():
    """追踪一个简单的 op (add) 从 Python 到 C++ kernel 的路径。"""
    print("=" * 60)
    print("1. Dispatch 链: torch.add(a, b) 如何到达 CUDA kernel")
    print("=" * 60)

    a = torch.randn(4)
    b = torch.randn(4)

    # Step 1: 查看参数上的 dispatch keys
    print(f"  a keys: {torch._C._dispatch_keys(a)}")
    print(f"  b keys: {torch._C._dispatch_keys(b)}")

    # Step 2: 查看当前 TLS (Thread Local State)
    local = torch._C._tls_local_dispatch_key_set()
    included = [k for k in dir(local) if k.startswith("included")]
    excluded = [k for k in dir(local) if k.startswith("excluded")]
    print(f"  TLS included/排除的 keys 会影响最终 dispatch")

    # Step 3: 查看 'add' 算子注册了哪些 kernel
    # 对于 CompositeExplicitAutograd 的 op, 不会在特定后端有注册
    print(f"\n  torch.add 的实现:")
    print(f"    BackendSelect → CompositeExplicitAutograd (默认)")
    print(f"    → 对 CPU: 用 CompositeExplicitAutograd kernel")
    print(f"    → 对 CUDA: 也用 CompositeExplicitAutograd kernel")
    print(f"    → Composite 自动分解为基础 op, 再分别 dispatch")

    if torch.cuda.is_available():
        a_cuda = torch.randn(4, device="cuda")
        b_cuda = torch.randn(4, device="cuda")
        print(f"\n  CUDA a keys: {torch._C._dispatch_keys(a_cuda)}")

    print()

    # 完整 dispatch 流程图
    print("  Dispatch 流程图:")
    print("  user code:   torch.add(a, b)")
    print("      │")
    print("  Python:      torch.Tensor.add / torch.add")
    print("      │")
    print("  C++ front:   at::add(a, b)")
    print("      │")
    print("  Dispatcher::call(op, args...)     ← Dispatcher.h:773")
    print("      │")
    print("  ├─ DispatchKeyExtractor           ← 从 args 提取 key set")
    print("  │   getDispatchKeySetUnboxed(...) ← 对所有 tensor 的 key_set() OR")
    print("  │   + TLS included - TLS excluded")
    print("  │")
    print("  ├─ OperatorEntry::lookup(ks)      ← OperatorEntry.h:182")
    print("  │   getDispatchTableIndexForDispatchKeySet()")
    print("  │   → dispatchTable_[idx]          ← O(1) 返回 kernel")
    print("  │")
    print("  └─ KernelFunction::call(...)       ← 执行 kernel")
    print("      │")
    print("  CPU kernel / CUDA kernel / Composite kernel / Autograd wrapper")
    print()


# ============ 2. TLS 内部操作 ============
def exp_tls_inside():
    """探究 TLS dispatch key 如何影响算子执行。"""
    print("=" * 60)
    print("2. TLS Disaptch Key: 用 no_grad / autocast 修改 dispatch")
    print("=" * 60)

    x = torch.randn(4, requires_grad=True)

    def inspect(name, tensor):
        keys = str(torch._C._dispatch_keys(tensor))
        print(f"  {name}: requires_grad={tensor.requires_grad}")
        # 显示 Autograd key 是否存在
        has_autograd = "Autograd" in keys
        print(f"    Autograd key present: {has_autograd}")

    # Normal
    y = x * 2
    inspect("Normal grad mode", y)

    # no_grad: 排除 Autograd key
    with torch.no_grad():
        y = x * 2
        inspect("no_grad context", y)

    # enable_grad
    with torch.enable_grad():
        y = x * 2
        inspect("enable_grad (default)", y)

    # autocast: 包含 Autocast 相关 key
    if torch.cuda.is_available():
        xc = torch.randn(4, 4, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            yc = xc * 2
            keys = str(torch._C._dispatch_keys(yc))
            print(f"  autocast(cuda):")
            # Autocast key 应该被包含
            has_autocast = "Autocast" in keys
            print(f"    Autocast key present: {has_autocast}")
            print(f"    output dtype: {yc.dtype}")

    print(f"\n  TLS 底层实现:")
    print(f"  torch/csrc/utils/tls.cpp")
    print(f"  void tls_set_dispatch_key_included(dispatch_key, in_set)")
    print(f"  void tls_set_dispatch_key_excluded(dispatch_key, in_set)")
    print(f"  本质: 修改线程局部存储的 DispatchKeySet")
    print(f"  开销: 零 (无 branch, 只是 key set 的位运算)")
    print()


# ============ 3. Kernel 注册表探查 ============
def exp_kernel_lookup():
    """查看某个算子在不同 Dispatch Key 下注册了哪些 kernel。"""
    print("=" * 60)
    print("3. Kernel 注册表: 哪些 key 有实现")
    print("=" * 60)

    # 检查一些算子在不同 backend 是否有 kernel
    ops_to_check = [
        ("aten::add", ["CPU", "CUDA", "CompositeExplicitAutograd", "Meta"]),
        ("aten::linear", ["CPU", "CUDA", "CompositeExplicitAutograd", "Autograd"]),
        ("aten::sum", ["CPU", "CUDA", "CompositeImplicitAutograd"]),
    ]

    for op_name, keys in ops_to_check:
        print(f"\n  {op_name}:")
        for key in keys:
            try:
                has = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key)
                print(f"    {key:30s}: {'YES' if has else 'no'}")
            except RuntimeError:
                print(f"    {key:30s}: [op not found]")

    # 使用 torch.library 注册自定义 op 然后探查
    lib = torch.library.Library("test3_lib", "DEF")
    lib.define("analyze_test(Tensor a) -> Tensor")

    @torch.library.impl("test3_lib::analyze_test", "CompositeImplicitAutograd")
    def analyze_test_impl(x):
        return x * 2

    # 现在查询我们自己注册的 op
    print(f"\n  test3_lib::analyze_test (我们自己注册的):")
    for key in ["CPU", "CUDA", "Meta", "CompositeImplicitAutograd"]:
        has = torch._C._dispatch_has_kernel_for_dispatch_key(
            "test3_lib::analyze_test", key
        )
        print(f"    {key:30s}: {'YES' if has else 'no (falls to Composite)'}")

    print(f"\n  CompositeImplicitAutograd 对所有 backend 都可见:")
    print(f"  CPU kernel 不存在, 但 op 在 CPU 上仍可运行")
    print(f"  → Dispatcher 自动 fallback 到 CompositeImplicitAutograd")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_dispatch_trace()
        exp_tls_inside()
        exp_kernel_lookup()

    print("[Dispatcher source analysis] DONE")


if __name__ == "__main__":
    main()
