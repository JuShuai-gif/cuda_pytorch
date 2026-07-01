"""AMP 源码分析: autocast dispatch 链, C++ CastPolicy table, dtype tracing.

使用工具: torch._C._dispatch_has_kernel / 自定义 autocast 遍历 /
         torch.amp 内部 API / params_by_dtype 探查

运行:
  python test3.py                  # 全链路分析
  python test3.py dispatch_path    # Autocast dispatch key 路径
  python test3.py policy_table     # CastPolicy 枚举映射
  python test3.py dtype_trace      # 追踪每条 op 的输入输出 dtype
  python test3.py cached_cast      # cached_cast 缓存分析

参考源码:
  aten/src/ATen/autocast_mode.cpp  — C++ dispatcher autocast 层
  aten/src/ATen/autocast_mode.h    — CastPolicy 枚举 + WrapFunction_
  torch/amp/autocast_mode.py       — Python 端 autocast 上下文
"""

import sys
import torch
import torch.nn as nn


# ============ 1. Autocast dispatch 链追踪 ============
def exp_dispatch_path():
    """追踪 autocast 如何通过 dispatch key 拦截算子。"""
    print("=" * 60)
    print("1. Autocast Dispatch: 从 Python 到 C++ 的精度选择链")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    # 无 autocast: 算子不经过 Autocast key
    x = torch.randn(4, 4, device="cuda")
    y_norm = x * 2
    keys_norm = torch._C._dispatch_keys(y_norm)
    print(f"  无 autocast: output keys = {keys_norm}")
    print(f"    dtype = {y_norm.dtype}")

    # 有 autocast: Autocast key 被包含到 TLS
    with torch.autocast("cuda", dtype=torch.float16):
        y_auto = x * 2
        keys_auto = torch._C._dispatch_keys(y_auto)
        print(f"\n  有 autocast: output keys = {keys_auto}")
        print(f"    dtype = {y_auto.dtype}")

    print(f"\n  Dispatch 链:")
    print(f"  user: with autocast():                     ← Python 上下文")
    print(f"    → torch.set_autocast_enabled(True)        ← Python 侧 (:308)")
    print(f"      → tls_set_dispatch_key_excluded(        ← C++ 侧 TLS")
    print(f"          Autocast, !enabled)                 ← 排除=启用")
    print(f"  运行时:")
    print(f"  Dispatcher: key_set |= Autocast (from TLS)")
    print(f"    → OperatorEntry::lookup → AutocastDispatchKey")
    print(f"      → WrapFunction_<lower_precision_fp>    ")
    print(f"        → cached_cast(all_inputs → fp16)")
    print(f"        → 调用真正的 backend kernel (fp16)")
    print()


# ============ 2. Policy table 映射 ============
def exp_policy_table():
    """展示 autocast 的 CastPolicy 分类表。"""
    print("=" * 60)
    print("2. CastPolicy 表: 每个 op 归哪类策略")
    print("=" * 60)

    # 从源码 autocast_mode.h 提取的分类
    policies = {
        "lower_precision_fp (→ fp16)": {
            "宏": "AT_FORALL_LOWER_PRECISION_FP (:819)",
            "典型 op": "conv1d/2d/3d, matmul, mm, bmm, "
            "addmm, linear, einsum, attention, mul/add/sub",
        },
        "fp32 (← 强制 fp32)": {
            "宏": "AT_FORALL_FP32 (:854)",
            "典型 op": "exp, log, pow, softplus, layer_norm, "
            "group_norm, 各种 loss, cdist",
        },
        "fp32_set_opt_dtype": {
            "宏": "AT_FORALL_FP32_SET_OPT_DTYPE (:915)",
            "典型 op": "softmax, log_softmax, sum, prod, cumsum",
        },
        "promote (宽 dtype 对齐)": {
            "宏": "AT_FORALL_PROMOTE (:945)",
            "典型 op": "addcdiv, atan2, cross, dot, index_put",
        },
    }

    for policy, info in policies.items():
        print(f"  [{policy}]")
        print(f"    宏定义: {info['宏']}")
        print(f"    算子: {info['典型 op']}")
        print()

    print(f"  每个 policy 有对应的 WrapFunction_ 模板特化:")
    print(f"  WrapFunction_<lower_precision_fp> (:470)")
    print(f"    → cached_cast(args → fp16) → call(kernel)")
    print(f"  WrapFunction_<fp32> (:494)")
    print(f"    → cached_cast(args → fp32) → call(kernel)")
    print(f"  WrapFunction_<fp32_set_opt_dtype> (:515)")
    print(f"    → 仅当用户没指定输出 dtype 时设 fp32")
    print(f"  WrapFunction_<promote> (:566)")
    print(f"    → promote_type → 全部对齐到最宽 dtype")
    print()


# ============ 3. dtype 逐算子追踪 ============
def exp_dtype_trace():
    """追踪 autocast 下每个算子的输入输出 dtype。"""
    print("=" * 60)
    print("3. Dtype 追踪: autocast 如何逐算子决定精度")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    class DtypeTracer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4).cuda()
            self.log = []

        def forward(self, x):
            self.log.append(("input", x.dtype))

            # matmul → lower_precision → fp16
            y = self.fc(x)
            self.log.append(("linear(output)", y.dtype))

            # relu → follows input → fp16
            z = torch.relu(y)
            self.log.append(("relu", z.dtype))

            # softmax → fp32_set_opt_dtype → fp32
            w = torch.softmax(z, dim=-1)
            self.log.append(("softmax", w.dtype))

            # log → fp32
            v = torch.log(w + 1e-8)
            self.log.append(("log", v.dtype))

            return v

    tracer = DtypeTracer()
    x = torch.randn(2, 4, device="cuda")

    with torch.autocast("cuda", dtype=torch.float16):
        y = tracer(x)

    print(f"  Autocast dtype 决策链:")
    for op_name, dt in tracer.log:
        print(f"    {op_name:20s}: {dt}")
    print()
    print(f"  分析:")
    print(f"  - matmul (lower_precision_fp) → fp16")
    print(f"  - relu (lower_precision_fp) → fp16")
    print(f"  - softmax (fp32_set_opt_dtype) → fp32  ← 精度敏感")
    print(f"  - log (fp32) → fp32  ← 数值稳定性要求")
    print()


# ============ 4. cached_cast 缓存分析 ============
def exp_cached_cast():
    """分析 autocast 如何缓存 fp32→fp16 转换。"""
    print("=" * 60)
    print("4. cached_cast: fp32 权重只转 fp16 一次")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    m = nn.Linear(16, 16).cuda()
    # 验证权重始终是 fp32
    print(f"  weight dtype: {m.weight.dtype}")
    print(f"  weight storage: {m.weight.untyped_storage().data_ptr():#x}")

    # autocast 下多次 forward
    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            y = m(torch.randn(4, 16, device="cuda"))
            print(f"    forward output dtype: {y.dtype}")

    print(f"\n  autocast 后 weight dtype: {m.weight.dtype}  (仍为 fp32)")
    print()
    print(f"  cached_cast 原理 (autocast_mode.cpp:122):")
    print(f"  1. 对每个 fp32 参数, 第一次 forward 时调用 arg.to(fp16)")
    print(f"  2. 把 fp16 结果缓存 (weak_intrusive_ptr 防地址复用)")
    print(f"  3. 同一 forward 内再次遇到此参数 → 直接返回缓存的 fp16")
    print(f"  4. __exit__ 时 clear_autocast_cache()")
    print(f"  → 权重物理存储永远是 fp32, 缓存只在一次 forward 内有效")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_dispatch_path()
        exp_policy_table()
        exp_dtype_trace()
        exp_cached_cast()

    print("[AMP source analysis] DONE")


if __name__ == "__main__":
    main()
