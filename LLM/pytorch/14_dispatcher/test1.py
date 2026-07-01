"""Dispatcher demo: register custom ops, dispatch keys, TLS control.

Companion script for dispatcher/dispatcher.md. Covers:
  1. register custom op:     torch.library.define + impl
  2. dispatch key routing:   CPU vs CUDA kernels
  3. TLS dispatch control:   no_grad, autocast effects
  4. dispatch key inspection: see what keys are active

Run:
    python test1.py                # full demo
    python test1.py register       # custom op registration
    python test1.py tls            # TLS effects (no_grad, autocast)
    python test1.py inspect        # inspect dispatch keys
"""

import sys

import torch


# ============ 1. Custom op registration ============
def exp_register():
    print("=" * 60)
    print("1. Custom op registration with torch.library")
    print("=" * 60)

    # Define a new namespace and op
    mylib = torch.library.Library("demo", "DEF")
    mylib.define("myadd(Tensor a, Tensor b) -> Tensor")
    mylib.define("myscale(Tensor a, float scale) -> Tensor")

    # Register CPU implementation
    @torch.library.impl("demo::myadd", "CPU")
    def myadd_cpu(a, b):
        print(f"    [CPU kernel] add {list(a.shape)} + {list(b.shape)}")
        return a + b

    @torch.library.impl("demo::myscale", "CPU")
    def myscale_cpu(a, scale):
        print(f"    [CPU kernel] scale {scale}")
        return a * scale

    # Test
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    c = torch.ops.demo.myadd(a, b)
    print(f"  result: {c.shape}, match: {torch.allclose(c, a + b)}")

    d = torch.ops.demo.myscale(a, 2.0)
    print(f"  myscale: {d.shape}, match: {torch.allclose(d, a * 2.0)}")

    # Register CUDA version if available
    if torch.cuda.is_available():

        @torch.library.impl("demo::myadd", "CUDA")
        def myadd_cuda(a, b):
            print(f"    [CUDA kernel] add {list(a.shape)} + {list(b.shape)}")
            return a + b

        a_cuda = torch.randn(3, 4, device="cuda")
        b_cuda = torch.randn(3, 4, device="cuda")
        c_cuda = torch.ops.demo.myadd(a_cuda, b_cuda)
        print(f"\n  CUDA result: {c_cuda.shape}")

    print()


# ============ 2. TLS dispatch effects ============
def exp_tls():
    print("=" * 60)
    print("2. TLS (Thread Local State) dispatch effects")
    print("=" * 60)

    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4, requires_grad=True)

    # no_grad: excludes Autograd key from dispatch
    with torch.no_grad():
        z = x + y
        print(f"  Inside no_grad:")
        print(f"    z.requires_grad: {z.requires_grad}")
        print(f"    z.grad_fn:       {z.grad_fn}")

    # Outside no_grad: Autograd key is active
    z2 = x + y
    print(f"\n  Outside no_grad:")
    print(f"    z2.requires_grad: {z2.requires_grad}")
    print(f"    z2.grad_fn:       {z2.grad_fn}")

    # autocast: includes AutocastCUDA key
    if torch.cuda.is_available():
        m = torch.nn.Linear(16, 16).cuda()
        xc = torch.randn(8, 16, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yc = m(xc)
            print(f"\n  Inside autocast:")
            print(f"    input dtype: {xc.dtype}    (fp32)")
            print(f"    weight dtype: {m.weight.dtype} (still fp32)")
            print(f"    output dtype: {yc.dtype}   (fp16, Autocast key routed matmul)")

    print()


# ============ 3. Dispatch key inspection ============
def exp_inspect():
    print("=" * 60)
    print("3. Dispatch key inspection")
    print("=" * 60)

    x = torch.randn(3)
    keys = torch._C._dispatch_keys(x)
    print(f"  Tensor dispatch keys: {keys}")

    # torch._C._dispatch_keys returns a DispatchKeySet
    # Check if Autograd is included
    x_grad = torch.randn(3, requires_grad=True)
    keys_grad = torch._C._dispatch_keys(x_grad)
    print(f"  With requires_grad: {keys_grad}")

    # Show active TLS keys
    local_keys = torch._C._tls_local_dispatch_key_set()
    print(f"\n  TLS local dispatch key set:")
    print(f"    included: {local_keys.included}")
    print(f"    excluded: {local_keys.excluded}")

    # Test a composite op (add is CompositeExplicitAutograd)
    print(f"\n  torch.add dispatch:")
    print(
        f"    has kernel for CPU: {torch._C._dispatch_has_kernel_for_dispatch_key('add', 'CPU')}"
    )
    print(
        f"    has kernel for Meta: {torch._C._dispatch_has_kernel_for_dispatch_key('add', 'Meta')}"
    )

    # CompositeExplicitAutograd provides default implementation for all backends
    print(
        f"    has CompositeExplicitAutograd: {torch._C._dispatch_has_kernel_for_dispatch_key('add', 'CompositeExplicitAutograd')}"
    )
    print()


EXPERIMENTS = {
    "register": exp_register,
    "tls": exp_tls,
    "inspect": exp_inspect,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[dispatcher demo] DONE")


if __name__ == "__main__":
    main()
