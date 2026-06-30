"""torch.compile debug demo: TORCH_LOGS, explain, graph break analysis.

Companion script for compile_debug/compile_debug.md.
  1. TORCH_LOGS overview:     all available log categories
  2. torch._dynamo.explain:   analyze graph breaks
  3. graph_breaks log:        real-time break monitoring
  4. recompiles log:          guard failure analysis
  5. output_code log:         see generated Triton/C++ code
  6. fullgraph mode:          force zero graph breaks

Run:
    python test1.py                    # overview demo
    python test1.py explain            # explain graph breaks
    python test1.py fullgraph          # fullgraph enforced
    python test1.py log_categories     # list all TORCH_LOGS options

Actual logging requires env vars, run like:
    TORCH_LOGS=graph_breaks python test1.py explain
    TORCH_LOGS=output_code python test1.py
"""

import sys
import torch
import torch.nn as nn


# ============ 1. TORCH_LOGS categories ============
def exp_log_categories():
    print("=" * 60)
    print("1. TORCH_LOGS categories: all available log streams")
    print("=" * 60)

    # Known log categories from torch/_dynamo/utils.py
    categories = {
        "graph_breaks": "每次 graph break 的原因 + 代码位置",
        "recompiles": "每次重新编译的 guard failure 详情",
        "output_code": "Inductor 生成的 Triton / C++ kernel 代码",
        "graph_code": "Dynamo 捕获的 FX 图 Python 源码",
        "guards": "生成的 guard 条件",
        "dynamic": "dynamic shapes 相关日志",
        "fusion": "Inductor 融合决策",
        "schedule": "Inductor scheduler 日志",
        "bytecode": "Dynamo 处理的 bytecode",
        "inlining": "函数内联决策",
        "+all": "所有日志 (输出很大!)",
    }

    print(f"  Usage: TORCH_LOGS=<category> python script.py")
    print(f"  Multiple: TORCH_LOGS=graph_breaks,recompiles python script.py")
    print()
    for name, desc in sorted(categories.items()):
        print(f"    {name:18s}: {desc}")

    print(f"\n  代码内设置:")
    print(f"    import torch._dynamo.config as dcfg")
    print(f"    dcfg.log_level = logging.DEBUG")
    print(f"  → 等价于 TORCH_LOGS=+all")
    print()


# ============ 2. torch._dynamo.explain ============
def exp_explain():
    print("=" * 60)
    print("2. torch._dynamo.explain: graph break analysis")
    print("=" * 60)

    def fn_with_breaks(x):
        a = x * 2
        b = a + 1
        # Data-dependent control flow → graph break
        if b.sum() > 0:
            b = b.relu()
        c = b * 3
        # .item() → graph break
        val = c.sum().item()
        d = c + val
        return d

    x = torch.randn(8)
    explanation = torch._dynamo.explain(fn_with_breaks)(x)

    print(f"  Graph count:        {explanation.graph_count}")
    print(f"  Graph break count:  {explanation.graph_break_count}")
    print(f"\n  Break reasons:")
    for i, (reason, user_stack) in enumerate(
        zip(explanation.break_reasons, explanation.user_stacks)
    ):
        print(f"    Break {i}: {reason}")
        # Print first line of user stack
        if user_stack:
            stack_lines = user_stack.strip().split("\n")
            for line in stack_lines[:2]:
                print(f"      {line.strip()[:100]}")

    print(f"\n  Graphs captured:")
    for i, gm in enumerate(explanation.graphs):
        ops = [
            n.name for n in gm.graph.nodes if n.op != "placeholder" and n.op != "output"
        ]
        print(f"    Graph {i}: {ops}")

    print(f"\n  Usage from CLI:")
    print(f"  TORCH_LOGS=graph_breaks python script.py  # realtime break monitoring")
    print()


# ============ 3. fullgraph mode ============
def exp_fullgraph():
    print("=" * 60)
    print("3. fullgraph=True: force zero graph breaks")
    print("=" * 60)

    # Clean function: no data-dependent control flow
    def clean_fn(x):
        return (x * 2 + 1).relu()

    clean_compiled = torch.compile(clean_fn, fullgraph=True)
    x = torch.randn(8)
    y = clean_compiled(x)
    print(f"  clean_fn: OK — output shape={list(y.shape)}")

    # Dirty function: has graph break
    def dirty_fn(x):
        y = x * 2
        if y.sum() > 0:  # graph break!
            y = y + 1
        return y

    dirty_compiled = torch.compile(dirty_fn, fullgraph=True)
    try:
        dirty_compiled(torch.randn(8))
    except Exception as e:
        print(f"\n  dirty_fn: {type(e).__name__}")
        lines = str(e).split("\n")
        for line in lines[:3]:
            print(f"    {line[:120]}")

    print(f"\n  → fullgraph=True makes compilation errors EXPLICIT")
    print(f"  → Without fullgraph: silently falls back to eager for dirty parts")
    print(f"  → With fullgraph:    raises error showing exactly where the problem is")
    print()


# ============ 4. Dynamo config probe ============
def exp_config_probe():
    print("=" * 60)
    print("4. Dynamo config: key settings affecting compilation")
    print("=" * 60)

    import torch._dynamo.config as dcfg

    key_settings = [
        (
            "dynamic_shapes",
            dcfg.dynamic_shapes,
            "When True, treat all shapes as dynamic (like compile(dynamic=True))",
        ),
        (
            "capture_scalar_outputs",
            dcfg.capture_scalar_outputs,
            "Capture .item() calls as scalar outputs",
        ),
        (
            "capture_dynamic_output_shape_ops",
            dcfg.capture_dynamic_output_shape_ops,
            "Capture ops that produce dynamic output shapes",
        ),
        (
            "suppress_errors",
            dcfg.suppress_errors,
            "If True, suppress errors and fall back to eager silently",
        ),
        (
            "cache_size_limit",
            dcfg.cache_size_limit,
            "Max number of compiled graphs to cache (prevents unlimited recompiles)",
        ),
        (
            "recompile_limit",
            dcfg.recompile_limit,
            "Max recompiles before giving up and raising error",
        ),
    ]

    for name, value, desc in key_settings:
        print(f"  {name:40s} = {value}")
        print(f"    {desc}")

    print(f"\n  设置方式:")
    print(f"  torch._dynamo.config.dynamic_shapes = True")
    print(f"  或: export TORCHDYNAMO_DYNAMIC_SHAPES=1")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_log_categories()
        exp_explain()
        exp_fullgraph()
        exp_config_probe()

    print("[compile_debug demo] DONE")


if __name__ == "__main__":
    main()
