"""Functionalization case study 7: version counter and mutation tracking.

Companion script for functionalization/functionalization.md. Covers:
  1. Tensor _version counter for mutation detection
  2. Autograd in-place check
  3. Version bump on mutation

Run:
    python 07_version_counter.py
"""

import sys

import torch


def exp_version_counter():
    print("=" * 60)
    print("1. Tensor._version: mutation tracker")
    print("=" * 60)

    x = torch.randn(3)
    print(f"  Initial version: {x._version}")

    x.add_(1)
    print(f"  After add_(1):   {x._version} (bumped)")

    y = x + 1
    print(f"  After x + 1:     {x._version} (unchanged, x not modified)")

    x.copy_(torch.zeros(3))
    print(f"  After copy_:     {x._version} (bumped)")

    v = x.view(-1)
    print(f"  After x.view:    {x._version} (unchanged, view doesn't mutate)")

    v.add_(1)
    print(f"  After v.add_:    {x._version} (bumped via view alias!)")

    print(f"\n  _version is how Autograd detects mutation")
    print(f"  Saved versions in AutogradNode vs current = re-compute needed")
    print()


def exp_autograd_inplace_check():
    print("=" * 60)
    print("2. Autograd in-place check failure")
    print("=" * 60)

    x = torch.randn(3, requires_grad=True)
    y = x * 2
    loss = y.sum()

    # Mutate x after building the graph
    x.add_(1)

    try:
        loss.backward()
        print(f"  backward succeeded (unexpected)")
    except RuntimeError as e:
        print(f"  backward FAILED: {str(e)[:100]}")
        print(f"  -> version mismatch detected by autograd engine")

    # Correct: clone before mutation
    x2 = torch.randn(3, requires_grad=True)
    z = x2 * 2
    loss2 = z.sum()
    x2_clone = x2.clone().add_(1)  # clone breaks autograd, mutation on clone
    x2.data = x2_clone.data       # replace data without breaking graph
    try:
        loss2.backward()
        print(f"\n  With clone + data replace: OK")
    except RuntimeError:
        print(f"\n  With clone + data replace: still may fail")
    print()


def exp_multi_view_version():
    print("=" * 60)
    print("3. Multiple views: all share version counter")
    print("=" * 60)

    x = torch.randn(4, 4)
    v1 = x[0]
    v2 = x[:, 0]
    v3 = x.view(-1)

    print(f"  Initial: x._version={x._version}, v1._version={v1._version}, v3._version={v3._version}")

    v1.add_(1)
    print(f"  After v1.add_: x._version={x._version}, v1._version={v1._version}, v3._version={v3._version}")
    print(f"  -> All aliases share the same version counter")
    print()


EXPERIMENTS = {
    "version": exp_version_counter,
    "autograd": exp_autograd_inplace_check,
    "multi": exp_multi_view_version,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 7] DONE")


if __name__ == "__main__":
    main()
