"""Design Patterns case study 2: Observer (hook) and Singleton.

Companion script for 40_design_patterns/design_patterns.md.

Run:
    python 02_observer_singleton.py
"""

import sys
import torch


def exp_autograd_observer():
    print("=" * 60)
    print("1. Observer: autograd hooks as callback list")
    print("=" * 60)
    x = torch.randn(3, requires_grad=True)
    events = []
    def hook_a(g):
        events.append("A")
        return g * 2
    def hook_b(g):
        events.append("B")
        return g
    x.register_hook(hook_a)
    x.register_hook(hook_b)
    y = (x * 2).sum()
    y.backward()
    print(f"  Hook execution order: {events}")
    print(f"  x.grad: {x.grad}  (hook_A doubled it)")


def exp_dispatcher_singleton():
    print("=" * 60)
    print("2. Singleton: all ops share one Dispatcher")
    print("=" * 60)
    lib1 = torch.library.Library("s1", "DEF")
    lib1.define("op1(Tensor x) -> Tensor")
    lib2 = torch.library.Library("s2", "DEF")
    lib2.define("op2(Tensor x) -> Tensor")
    d1 = torch._C._dispatch_dump_table("s1::op1")
    d2 = torch._C._dispatch_dump_table("s2::op2")
    print(f"  Two ops, one global Dispatcher singleton")
    print(f"  s1::op1 registration OK: {'CPU' in d1}")
    print(f"  s2::op2 registration OK: {'CPU' in d2}")


def exp_module_hook():
    print("=" * 60)
    print("3. Module hooks: Observer for forward/backward")
    print("=" * 60)
    model = torch.nn.Linear(4, 2)
    events = []
    def fw_hook(module, inp, out):
        events.append(f"fw: {inp[0].shape} -> {out.shape}")
    def bw_hook(module, grad_in, grad_out):
        events.append(f"bw: {grad_in[0].shape if grad_in[0] is not None else None}")
    model.register_full_backward_hook(bw_hook)
    model.register_forward_hook(fw_hook)
    x = torch.randn(3, 4, requires_grad=True)
    y = model(x)
    y.sum().backward()
    for e in events:
        print(f"  {e}")


EXPERIMENTS = {"observer": exp_autograd_observer, "singleton": exp_dispatcher_singleton, "module": exp_module_hook}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}'")
            continue
        EXPERIMENTS[name]()
    print("[design_patterns case 2] DONE")


if __name__ == "__main__":
    main()
