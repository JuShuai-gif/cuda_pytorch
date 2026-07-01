"""IValue Type case study: JIT type system.

Run: python 01_jit_types.py
"""

import sys, torch

def exp_jit_trace_types():
    print("=" * 60)
    print("1. JIT trace: automatic type inference")
    print("=" * 60)
    def f(x):
        return x.relu().sum()
    traced = torch.jit.trace(f, torch.randn(4, 8))
    print(f"  Traced graph:\n{traced.graph}")
    print(f"  Each node's input/output is IValue in C++")

def exp_schema():
    print("=" * 60)
    print("2. Operator schema = IValue types")
    print("=" * 60)
    schema = torch.ops.aten.add.Tensor.default._schema
    print(f"  add.Tensor schema: {schema}")
    print(f"  Arguments:")
    for arg in schema.arguments:
        print(f"    {arg.name}: {arg.type}")

def exp_ivalue_runtime():
    print("=" * 60)
    print("3. IValue types at runtime")
    print("=" * 60)
    try:
        m = torch.jit.script(torch.nn.Linear(4, 2))
        x = torch.randn(3, 4)
        y = m(x)
        print(f"  Script module forward: shape={list(y.shape)}")
        print(f"  Each value wraps IValue(Tensor) internally")
    except Exception as e:
        print(f"  {e}")

EXPERIMENTS = {"trace": exp_jit_trace_types, "schema": exp_schema, "runtime": exp_ivalue_runtime}

def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS: continue
        EXPERIMENTS[name]()
    print("[ivalue_type] DONE")

if __name__ == "__main__": main()
