"""FX Graph passes demo: DCE, CSE, constant folding, pattern replacement.

Companion script for graph_passes/graph_passes.md. Covers:
  1. DCE:                   dead code elimination
  2. symbolic trace:        capture computation graph
  3. CSE:                   common subexpression elimination
  4. Transformer:           pattern rewriting (add -> mul)
  5. const_fold:            pre-compute constant subgraphs
  6. fuse + DCE pipeline:   combine passes
  7. replace_pattern:       subgraph pattern matching & replacement

Run:
    python test1.py                # full demo
    python test1.py dce            # DCE demo
    python test1.py cse            # common subexpression elimination
    python test1.py transformer    # custom Transformer pass
    python test1.py const_fold     # constant folding
    python test1.py pattern        # subgraph pattern replacement
    python test1.py pipeline       # multi-pass optimization

=== DEBUG 常见问题 ===
  Q: symbolic_trace 报 "Cannot trace..."?
  A: 模型包含 Python control flow (if/for) 或 data-dependent op;
     尝试用 torch.fx.wrap() 包装不可 trace 的函数, 或使用 torch.compile

  Q: DCE 后输出变了?
  A: eliminate_dead_code 从 output 反向遍历, 如果有副作用节点 (print)
     未被正确标记为有 effect, 需要手动添加 output

  Q: CSE 没有消除我期望的重复?
  A: CSE 基于 (target, args_hash, kwargs_hash) 判断等价;
     如果 args 的顺序/形状不同, 即使是相同计算也不会消除

  Q: Transformer 替换后图输出不对?
  A: 检查 call_function 返回的新节点是否 dtype/shape 兼容;
     在不改变 dtype 的情况下做替换
"""

import sys

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.dialect.common.cse_pass import CSEPass


# ============ 1. DCE — Dead Code Elimination ============
def exp_dce():
    print("=" * 60)
    print("1. Dead Code Elimination (DCE)")
    print("=" * 60)

    class ModelWithDeadCode(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            a = self.linear(x)
            b = a * 2  # dead — never used
            c = torch.sum(b)  # dead — depends on b
            d = a + 1  # used
            return d

    model = ModelWithDeadCode()
    gm = fx.symbolic_trace(model)

    print(f"  Before DCE: {len(gm.graph.nodes)} nodes")
    for n in gm.graph.nodes:
        print(f"    {n.op:>10s} {n.name:>10s}  args={n.args}")

    gm.graph.eliminate_dead_code()
    gm.recompile()

    print(f"\n  After DCE: {len(gm.graph.nodes)} nodes")
    for n in gm.graph.nodes:
        print(f"    {n.op:>10s} {n.name:>10s}  args={n.args}")

    # Verify: output is identical
    x = torch.randn(2, 4)
    with torch.no_grad():
        y_orig = model(x)
        y_dce = gm(x)
    print(f"\n  Output match: {torch.allclose(y_orig, y_dce)}")
    print()


# ============ 2. CSE — Common Subexpression Elimination ============
def exp_cse():
    print("=" * 60)
    print("2. Common Subexpression Elimination (CSE)")
    print("=" * 60)

    class ModelWithDups(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            a = self.linear(x)
            b = a * 2
            c = a * 2  # duplicate of b
            d = b + c  # = b + b after CSE
            return d

    model = ModelWithDups()
    gm = fx.symbolic_trace(model)

    print(f"  Before CSE: {len(gm.graph.nodes)} nodes")

    # Apply CSE
    cse = CSEPass()
    result = cse(gm)
    gm = result.graph_module

    print(f"  After CSE:  {len(gm.graph.nodes)} nodes")
    for n in gm.graph.nodes:
        print(
            f"    {n.op:>10s} {n.name:>10s}  target={n.target if n.op != 'output' else ''}"
        )

    # Verify
    x = torch.randn(2, 4)
    with torch.no_grad():
        y_orig = model(x)
        y_cse = gm(x)
    print(f"\n  Output match: {torch.allclose(y_orig, y_cse)}")
    print()


# ============ 3. Transformer — custom pass ============
def exp_transformer():
    print("=" * 60)
    print("3. Transformer: custom pattern rewriting")
    print("=" * 60)

    class Model(nn.Module):
        def forward(self, x):
            a = x + 1
            b = a * 2
            c = b + 3
            return c

    model = Model()
    gm = fx.symbolic_trace(model)

    print(f"  Before transformation:")
    gm.graph.print_tabular()

    # Custom Transformer: replace all 'add' with 'mul'
    class AddToMulTransformer(fx.Transformer):
        def call_function(self, target, args, kwargs):
            if target == torch.add:
                # Get the operands
                return torch.mul(*args, **kwargs)
            return super().call_function(target, args, kwargs)

    transformer = AddToMulTransformer(gm)
    gm_transformed = transformer.transform()

    print(f"\n  After transformation (add → mul):")
    gm_transformed.graph.print_tabular()

    # Verify behavior changed
    x = torch.tensor([3.0])
    with torch.no_grad():
        y_before = gm(x)
        y_after = gm_transformed(x)
    print(
        f"  Before transform: f(3) = {y_before.item():.1f}  (original: (3+1)*2+3 = 11)"
    )
    print(
        f"  After  transform: f(3) = {y_after.item():.1f}  (modified: (3*1)*2*3 = 12)"
    )
    print()


# ============ 4. Multi-pass pipeline ============
def exp_pipeline():
    print("=" * 60)
    print("4. Multi-pass optimization pipeline")
    print("=" * 60)

    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3)
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.fc = nn.Linear(16 * 4 * 4, 10)

        def forward(self, x):
            a = self.conv1(x)
            b = torch.relu(a)
            c = self.conv2(b)
            d = c * 2  # pointwise
            e = d + 1  # pointwise
            unused = e * 0  # DEAD CODE
            f = torch.relu(e)
            g = f * f  # duplicate of... nothing useful
            h = g.view(g.size(0), -1)
            out = self.fc(h)
            return out

    model = BigModel().eval()
    gm = fx.symbolic_trace(model)

    # Pipeline: CSE → DCE
    initial_nodes = len(gm.graph.nodes)
    print(f"  Nodes before optimization: {initial_nodes}")

    # Pass 1: CSE
    from torch.fx.passes.dialect.common.cse_pass import CSEPass

    pass_cse = CSEPass()
    result1 = pass_cse(gm)
    gm = result1.graph_module
    after_cse = len(gm.graph.nodes)
    print(f"  After CSE:  {after_cse}")

    # Pass 2: DCE
    gm.graph.eliminate_dead_code()
    gm.recompile()
    after_dce = len(gm.graph.nodes)
    print(f"  After DCE:  {after_dce}")

    print(
        f"  Total reduction: {initial_nodes} → {after_dce} nodes ({after_dce / initial_nodes * 100:.0f}%)"
    )

    # Verify correctness
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y_orig = model(x)
        y_opt = gm(x)
    print(f"  Output match:  {torch.allclose(y_orig, y_opt, atol=1e-5)}")
    print()


# ============ 5. Constant folding ============
def exp_const_fold():
    print("=" * 60)
    print("5. Constant folding: pre-compute constant subgraphs")
    print("=" * 60)

    class ModelWithConstants(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        def forward(self, x):
            a = x + 1.0  # constant add
            b = self.c * 3.1415  # constant multiply
            c = a + b  # depends on both constant and input
            d = c * 2
            unused = x * 0  # dead
            return d

    model = ModelWithConstants()
    gm = fx.symbolic_trace(model)

    initial_nodes = len(gm.graph.nodes)
    print(f"  Before: {initial_nodes} nodes")

    # DCE first to remove obvious dead nodes
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # Try const fold
    try:
        from torch.fx.experimental.const_fold import split_const_subgraphs

        # This splits constant-only ops into separately-evaluated sub-modules
        folded = split_const_subgraphs(gm)
        folded_nodes = len(folded.graph.nodes)
        print(f"  After const_fold + DCE: {folded_nodes} nodes")
        print(
            f"  Reduction: {initial_nodes} -> {folded_nodes} ({folded_nodes / initial_nodes * 100:.0f}%)"
        )
    except Exception as e:
        print(f"  const_fold: {type(e).__name__} — {e}")
        print(f"  After DCE: {len(gm.graph.nodes)} nodes")

    # Verify
    x = torch.randn(3)
    with torch.no_grad():
        y_orig = model(x)
        y_opt = gm(x)
    print(f"  Output match: {torch.allclose(y_orig, y_opt)}")
    print()


# ============ 6. Pattern replacement ============
def exp_pattern():
    print("=" * 60)
    print("6. Subgraph pattern replacement")
    print("=" * 60)

    class ModelWithFusion(nn.Module):
        def forward(self, x):
            a = x * 2
            b = a + 1
            c = b.relu()
            d = c * 3
            e = d + 2
            f = e.relu()
            return f

    model = ModelWithFusion()
    gm = fx.symbolic_trace(model)
    print(f"  Original graph: {len(gm.graph.nodes)} nodes")

    # Pattern: mul -> add -> relu  (we want to replace with something)
    class PatternModule(nn.Module):
        def forward(self, x):
            return (x * 2 + 1).relu()

    # Replacement: fused op (just as example)
    class ReplacementModule(nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x)  # replace with GELU

    pattern_gm = fx.symbolic_trace(PatternModule())
    replacement_gm = fx.symbolic_trace(ReplacementModule())

    try:
        from torch.fx.subgraph_rewriter import replace_pattern

        matches = replace_pattern(gm, pattern_gm, replacement_gm)
        print(f"  Matches found: {len(matches)}")
        gm.graph.eliminate_dead_code()
        gm.recompile()
        print(f"  After replace + DCE: {len(gm.graph.nodes)} nodes")

        # Verify behavior changed (GELU vs RELU)
        x = torch.randn(10)
        with torch.no_grad():
            y_orig = model(x)
            y_new = gm(x)
        diff = (y_orig - y_new).norm()
        print(f"  Output changed: norm diff = {diff:.3f}")
        print(f"  (expected: GELU != relu-chain for some values)")
    except Exception as e:
        print(f"  pattern replacement: {e}")

    print()


EXPERIMENTS = {
    "dce": exp_dce,
    "cse": exp_cse,
    "transformer": exp_transformer,
    "const_fold": exp_const_fold,
    "pattern": exp_pattern,
    "pipeline": exp_pipeline,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[graph_passes demo] DONE")


if __name__ == "__main__":
    main()
