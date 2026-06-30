"""Inductor 源码分析: IR 内部结构, fusion 过程, codegen 输出。

使用工具: TORCH_LOGS=output_code / torch._inductor.config /
         print_readable / debug 模式

运行:
  python test3.py                  # 全链路分析 (需 CUDA)
  python test3.py ir_structure     # Inductor IR 内部结构
  python test3.py fusion_trace     # 融合决策追踪
  python test3.py codegen_trace    # 代码生成过程观察

参考源码:
  torch/_inductor/graph.py       — GraphLowering (FX→IR)
  torch/_inductor/scheduler.py   — 算子融合决策
  torch/_inductor/codegen/triton.py — TritonKernel + codegen
"""

import sys
import torch
import torch.nn as nn


# ============ 1. Inductor IR 内部结构 ============
def exp_ir_structure():
    """探究 Inductor 的内部 IR 节点类型。"""
    print("=" * 60)
    print("1. Inductor IR: InputBuffer / Pointwise / Reduction / ComputedBuffer")
    print("=" * 60)

    @torch.compile(backend="inductor")
    def fn(x):
        a = torch.floor(x)
        b = torch.ceil(x)
        c = a + b
        d = c.sum(dim=-1)
        return d + 1

    x = torch.randn(2, 3, 16, device="cuda" if torch.cuda.is_available() else "cpu")

    # Warmup to trigger compilation
    fn(x)

    print(f"  Inductor IR 节点类型 (对应源代码 torch/_inductor/ir.py):")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │ InputBuffer         ← placeholder (输入张量)    │")
    print(f"  │   name / layout / device / dtype / strides      │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │ Pointwise           ← element-wise 运算         │")
    print(f"  │   inner_fn: (index) → value                     │")
    print(f"  │   被融合时不实际存在, 只在 codegen 时展开       │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │ Reduction           ← sum / max / mean          │")
    print(f"  │   ranges: [非约简维度]                          │")
    print(f"  │   reduction_ranges: [约简维度]                  │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │ ComputedBuffer      ← 需要显式存储的结果         │")
    print(f"  │   name / layout / data (Pointwise|Reduction)    │")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  IR vs 实际内存:")
    print(f"  Pointwise 节点 = 纯配方 (recipe), 不分配内存")
    print(f"  ComputedBuffer = 需要分配内存的中间/最终结果")
    print(f"  调度器决定哪些 ComputedBuffer 可以融合 → 减少内存分配")
    print()


# ============ 2. Fusion 决策追踪 ============
def exp_fusion_trace():
    """分析 Inductor 如何决定融合哪些 op。"""
    print("=" * 60)
    print("2. Fusion 追踪: can_fuse / vertical / horizontal")

    # 查看 Inductor 的融合日志
    import os

    os.environ["TORCH_LOGS"] = "+fusion"

    @torch.compile
    def fn(x):
        a = x * 2  # pointwise
        b = a + 1  # pointwise (与 a 垂直融合)
        c = b.relu()  # pointwise (与 b 垂直融合)
        d = c * 3  # pointwise (与 c 垂直融合)
        return d

    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        fn(x)

        # Inductor 融合决策规则 (源代码 scheduler.py + simd.py):
        print(f"  融合规则 (从 scheduler.py:5102 和 simd.py:2029):")
        print(f"  1. pointwise + pointwise: 相同 shape → 垂直融合")
        print(f"     x*2 + 1 + relu + *3 → 全部融合进单 kernel")
        print(f"  2. pointwise + reduction: broadcast shape → 融合")
        print(f"  3. reduction + reduction: 相同 shape → 融合")
        print(f"  4. 有 shared buffer 的节点对 → 融合候选")
        print(f"  5. 融合后按 memory saving 评分排序 → 优先融合收益大的")
    else:
        print("  [SKIP] CUDA not available for fusion logging")

    # Clean up env
    del os.environ["TORCH_LOGS"]
    print()


# ============ 3. Generated Triton code inspection ============
def exp_codegen_trace():
    """查看 Inductor 生成的 Triton kernel 代码。"""
    print("=" * 60)
    print("3. Codegen 输出: 查看生成的 Triton kernel")
    print("=" * 60)

    # 使用 TORCH_LOGS=output_code 查看生成的代码
    import os

    os.environ["TORCH_LOGS"] = "output_code"

    @torch.compile
    def fn(x):
        return (x * 2 + 1).relu()

    if torch.cuda.is_available():
        x = torch.randn(1024 * 1024, device="cuda")
        fn(x)
        print(f"  运行 TORCH_LOGS=output_code 可看到生成代码")
        print(f"  生成代码包含:")
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │ @pointwise(size_hints=[16777216], ...)        │")
        print(f"  │ @triton.jit                                   │")
        print(f"  │ def triton_(in_ptr0, out_ptr0, xnumel, ...):  │")
        print(f"  │   xoffset = tl.program_id(0) * XBLOCK         │")
        print(f"  │   x0 = xoffset + tl.arange(0, XBLOCK)[:]      │")
        print(f"  │   tmp0 = tl.load(in_ptr0 + x0, None)          │")
        print(f"  │   tmp1 = tmp0 * 2.0                           │")
        print(f"  │   tmp2 = tmp1 + 1.0                           │")
        print(f"  │   tmp3 = tl.where(tmp2 > 0, tmp2, 0)          │")
        print(f"  │   tl.store(out_ptr0 + x0, tmp3, None)         │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"  注意: 3 个 Python op (mul, add, relu) → 1 个 Triton kernel")
    else:
        print("  [SKIP] CUDA not available, but run with:")
        print("    TORCH_LOGS=output_code python test3.py codegen_trace")

    del os.environ["TORCH_LOGS"]
    print()


# ============ 4. Inductor 内部配置探查 ============
def exp_config():
    """探究 Inductor 的内部配置项如何影响编译行为。"""
    print("=" * 60)
    print("4. Inductor 配置: 影响编译和融合的关键选项")
    print("=" * 60)

    import torch._inductor.config as inductor_config

    # 关键配置项
    configs = [
        (
            "max_autotune",
            inductor_config.max_autotune,
            "是否自动搜索最优 tile size (慢但效果好)",
        ),
        (
            "triton.cudagraphs",
            inductor_config.triton.cudagraphs,
            "是否用 CUDA Graph 包装 Triton kernel",
        ),
        (
            "fx_graph_cache",
            inductor_config.fx_graph_cache,
            "是否缓存 FX 图编译结果 (加速重复编译)",
        ),
        (
            "epilogue_fusion",
            inductor_config.epilogue_fusion,
            "是否融合 epilogue (如 bias/batchnorm)",
        ),
        (
            "coordinate_descent_tuning",
            inductor_config.coordinate_descent_tuning,
            "是否用坐标下降法多轮 tuning",
        ),
    ]

    for name, val, desc in configs:
        print(f"  inductor_config.{name} = {val}")
        print(f"    {desc}")

    print(f"\n  配置如何影响编译:")
    print(f"  设置: torch._inductor.config.max_autotune = True")
    print(f"  或环境变量: TORCHINDUCTOR_MAX_AUTOTUNE=1")
    print(f"  → Inductor 会 benchmark 多种 kernel 配置, 选最优")
    print(f"  → 编译时间 ↑↑, 推理速度 ↑ (一次性代价)")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_ir_structure()
        exp_fusion_trace()
        exp_codegen_trace()
        exp_config()

    print("[Inductor source analysis] DONE")


if __name__ == "__main__":
    main()
