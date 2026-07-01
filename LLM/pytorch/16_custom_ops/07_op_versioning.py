"""Custom Ops case study 7: operator versioning and backward compatibility.

Companion script for custom_ops/custom_ops.md. Covers:
  1. Op versioning via schema changes
  2. Backward compatibility strategies
  3. Deprecation pattern

Run:
    python 07_op_versioning.py
"""

import sys

import torch


def exp_schema_versioning():
    print("=" * 60)
    print("1. Op schema versioning: adding new args")
    print("=" * 60)

    # Version 1: simple op
    lib = torch.library.Library("version_demo", "DEF")
    lib.define("my_op_v1(Tensor x) -> Tensor")

    @torch.library.impl("version_demo::my_op_v1", "CPU")
    def my_op_v1_cpu(x):
        return x * 2

    # Version 2: add optional parameter (backward compatible)
    lib.define("my_op_v2(Tensor x, float scale=1.0) -> Tensor")

    @torch.library.impl("version_demo::my_op_v2", "CPU")
    def my_op_v2_cpu(x, scale=1.0):
        return x * 2 * scale

    x = torch.randn(3)

    y1 = torch.ops.version_demo.my_op_v1(x)
    y2 = torch.ops.version_demo.my_op_v2(x)        # default scale=1.0
    y3 = torch.ops.version_demo.my_op_v2(x, 2.0)   # explicit scale

    print(f"  v1 (x*2):            {y1.tolist()}")
    print(f"  v2 (x*2*1.0):        {y2.tolist()}")
    print(f"  v2 (x*2*2.0):        {y3.tolist()}")
    print(f"  Match (v1==v2 default): {torch.allclose(y1, y2)}")

    print(f"\n  Backward compatibility:")
    print(f"    - Old code calling v1 still works")
    print(f"    - v2 adds optional scale with default=1.0")
    print(f"    - Schema change is backward compatible")
    print()


def exp_deprecation_pattern():
    print("=" * 60)
    print("2. Deprecation pattern for custom ops")
    print("=" * 60)

    print(f"  Deprecation strategy:")
    print(f"")
    print(f"  Step 1: Add new op (recommended)")
    print(f"    lib.define('my_op_v2(...)')")
    print(f"")
    print(f"  Step 2: Mark old op as deprecated")
    print(f"    @torch.library.impl('myops::my_op_v1', 'CPU')")
    print(f"    def my_op_v1_deprecated(x):")
    print(f"        import warnings")
    print(f"        warnings.warn('my_op_v1 is deprecated, use my_op_v2')")
    print(f"        return torch.ops.myops.my_op_v2(x)")
    print(f"")
    print(f"  Step 3: Remove old op (major version)")
    print(f"    # Remove from codebase, users must migrate")
    print()


EXPERIMENTS = {
    "version": exp_schema_versioning,
    "deprecation": exp_deprecation_pattern,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 7] DONE")


if __name__ == "__main__":
    main()
