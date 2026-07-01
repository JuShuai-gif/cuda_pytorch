"""torch.export demo: ExportedProgram, dynamic shapes, serialization.

Companion script for deploy/deploy.md. Covers:
  1. basic export:       export -> ExportedProgram -> run
  2. dynamic shapes:     Dim, ShapesCollection, constraints
  3. constraint violation: what happens when new input violates Dim range
  4. serialization:      save/load
  5. graph inspection:   print_readable, graph.nodes
  6. programmatic run:   forward with new inputs (within constraints)

Run:
    python test1.py              # full demo
    python test1.py basic        # basic export
    python test1.py dynamic      # dynamic shapes
    python test1.py constraint   # constraint violation debug
    python test1.py serialize    # save/load
    python test1.py inspect      # graph inspection

=== DEBUG 常见问题 ===
  Q: export 失败 "Cannot capture data-dependent control flow"?
  A: torch.export 要求图是 static single assignment (SSA), 不能有
     if x.sum()>0 / for i in range(n) 等数据依赖控制流;
     用 torch.cond / torch.map 替代, 或使用 torch.compile 而非 export

  Q: dynamic_shapes constraint 违反报 "input shape outside declared range"?
  A: 检查 Dim(min=, max=) 范围, 确保运行时输入在此范围内;
     用 python test1.py constraint 观察报错信息

  Q: export 后输出与原始模型不一致?
  A: 检查 strict=False 时是否有 op 被自动填充; 对比 graph.nodes
     中的 target 与原始 forward 中的 ops 是否一致
"""

import sys
import os
import tempfile

import torch
import torch.nn as nn
from torch.export import export, Dim, ShapesCollection, save, load


# ============ 1. Basic export ============
def exp_basic():
    print("=" * 60)
    print("1. Basic export: Module → ExportedProgram")
    print("=" * 60)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)
            self.bn = nn.BatchNorm1d(3)

        def forward(self, x):
            x = self.linear(x)
            x = torch.relu(x)
            return self.bn(x)

    model = SimpleModel()
    model.eval()

    x = torch.randn(2, 4)

    ep = export(model, (x,))
    print(f"  Type:        {type(ep).__name__}")
    print(f"  Graph nodes: {len(ep.graph.nodes)}")
    print(f"  State dict:  {len(ep.state_dict)} entries")

    # Run with new inputs
    x2 = torch.randn(2, 4)
    y = ep.module()(x2)
    print(f"  Output shape: {list(y.shape)}")

    # Compare with original model
    with torch.no_grad():
        y_ref = model(x2)
    print(f"  Match orig:   {torch.allclose(y, y_ref, atol=1e-5)}")
    print()


# ============ 2. Dynamic shapes ============
def exp_dynamic():
    print("=" * 60)
    print("2. Dynamic shapes: batch dimension")
    print("=" * 60)

    model = nn.Linear(16, 8).eval()

    batch = Dim("batch", min=1, max=128)

    # Old API: dict-style
    x = torch.randn(4, 16)
    ep_dict = export(model, (x,), dynamic_shapes={"x": {0: batch}})

    print(f"  Dynamic dims (dict style):")
    for node in ep_dict.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            if val is not None:
                print(f"    placeholder '{node.name}': shape={list(val.shape)}")

    # New API: ShapesCollection
    x2 = torch.randn(4, 16)
    dim = Dim("batch_dim", max=256)
    ep_new = export(model, (x2,), dynamic_shapes=ShapesCollection(x=(dim, 16)))

    # Run with different batch sizes
    for bs in [2, 8, 16]:
        inp = torch.randn(bs, 16)
        y = ep_new.module()(inp)
        print(
            f"  batch={bs}: output shape={list(y.shape)}, match={torch.allclose(y, model(inp), atol=1e-5)}"
        )

    print("  -> dynamic shapes allow varying batch size without re-export")
    print()


# ============ 3. Serialization ============
def exp_serialize():
    print("=" * 60)
    print("3. Export serialization: save/load")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    x = torch.randn(3, 8)

    ep = export(model, (x,))
    y_orig = ep.module()(x)

    # Save
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model.pt2")
    save(ep, path)
    fsize = os.path.getsize(path) / 1024
    print(f"  Saved to:  {path} ({fsize:.1f} KB)")

    # Load
    ep_loaded = load(path)
    y_loaded = ep_loaded.module()(x)

    print(f"  Loaded ok: {type(ep_loaded).__name__}")
    print(f"  Match:     {torch.allclose(y_orig, y_loaded, atol=1e-5)}")

    # Check state_dict intact
    orig_param = ep.state_dict["0.weight"]
    loaded_param = ep_loaded.state_dict["0.weight"]
    print(f"  Params ok: {torch.allclose(orig_param, loaded_param)}")

    import shutil

    shutil.rmtree(tmpdir)
    print()


# ============ 4. Graph inspection ============
def exp_inspect():
    print("=" * 60)
    print("4. Graph inspection")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    x = torch.randn(2, 4)

    ep = export(model, (x,))

    # Print readable graph
    print("  Graph structure:")
    graph_str = str(ep.graph)
    for line in graph_str.split("\n")[:15]:
        print(f"    {line}")
    print("    ...")

    # Node-level inspection
    print(f"\n  Node types: {set(n.op for n in ep.graph.nodes)}")
    print(f"  Nodes:")
    for n in ep.graph.nodes:
        if n.op == "call_function":
            # Show ATen op name
            print(
                f"    {n.op:>15s}: {str(n.target).split('.')[-1]:30s} "
                f"args={[a.name if hasattr(a, 'name') else str(a)[:20] for a in n.args]}"
            )
        elif n.op == "call_module":
            print(f"    {n.op:>15s}: {n.target:30s}")
        elif n.op in ("placeholder", "output"):
            print(f"    {n.op:>15s}: {n.name}")

    # Graph signature
    sig = ep.graph_signature
    print(f"\n  Input specs:  {len(sig.input_specs)}")
    print(f"  Output specs: {len(sig.output_specs)}")
    print(f"  Parameters:   {len(sig.parameters)}")

    # Use print_readable for a clean textual representation
    print("\n  print_readable:")
    readable = ep.print_readable(print_output=False)
    for line in readable.split("\n")[:10]:
        print(f"    {line}")
    print("    ...")
    print()


# ============ 3b. Constraint violation debug ============
def exp_constraint():
    print("=" * 60)
    print("3b. Constraint violation: what happens with wrong shapes")
    print("=" * 60)

    model = nn.Linear(8, 4).eval()

    batch = Dim("batch", min=1, max=16)
    x = torch.randn(4, 8)
    ep = export(model, (x,), dynamic_shapes={"x": {0: batch}})

    # Run within constraints: OK
    y_ok = ep.module()(torch.randn(8, 8))
    print(f"  batch=8 (within [1,16]): OK, shape={list(y_ok.shape)}")

    # Run outside constraints: runtime error
    try:
        ep.module()(torch.randn(32, 8))  # 32 > max=16!
    except RuntimeError as e:
        msg = str(e)
        # Show key parts of the error
        for line in msg.split("\n")[:3]:
            print(f"  batch=32 (exceeds max=16):")
            print(f"    {line}")
        if len(msg.split("\n")) > 3:
            print(f"    ... ({len(msg.splitlines())} lines total)")

    print("\n  -> constraint violations produce detailed guard failure messages")
    print("  -> check: Dim(min=, max=) ranges, then verify runtime inputs")
    print("  -> fix: increase max in Dim(), or handle re-export at runtime")
    print()


EXPERIMENTS = {
    "basic": exp_basic,
    "dynamic": exp_dynamic,
    "constraint": exp_constraint,
    "serialize": exp_serialize,
    "inspect": exp_inspect,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[deploy demo] DONE")


if __name__ == "__main__":
    main()
