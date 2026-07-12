"""
Debugging demo: each case intentionally triggers one class of bug and shows
the detection technique from 03_调试与调优手册.md (Part 1).

Run:
    conda activate torch_env
    python debug_demo.py

Every case is isolated in try/except so one failure does not stop the rest.
"""

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"


def case(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


# -------------------------------------------------------------------
def demo_shape_dtype_device():  # Playbook §2
    case("§2 shape/dtype/device mismatch")
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    try:
        _ = a + b
    except RuntimeError as e:
        # Detection: print the four attributes of every tensor involved.
        print("caught:", str(e).splitlines()[0])
        print(f"  a: shape={tuple(a.shape)} dtype={a.dtype} dev={a.device}")
        print(f"  b: shape={tuple(b.shape)} dtype={b.dtype} dev={b.device}")
        print("  fix: align shapes (broadcast rules) before the op")


# -------------------------------------------------------------------
def demo_nan_inf():  # Playbook §4
    case("§4 NaN / Inf hunting with forward hooks")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 4)

        def forward(self, x):
            x = self.a(x)
            x = torch.log(x)  # log of negatives -> NaN (the bug)
            return self.b(x)

    net = Net()

    # Detection: hook every module, report the FIRST one producing non-finite.
    def hook(mod, inp, out, name):
        if not torch.isfinite(out).all():
            print(f"  non-finite output at module: {name} ({type(mod).__name__})")

    for n, m in net.named_modules():
        if n:
            m.register_forward_hook(lambda mod, i, o, n=n: hook(mod, i, o, n))
    _ = net(torch.randn(2, 4))
    print("  fix: guard the domain (e.g. log(relu(x)+eps)); or use anomaly mode")


# -------------------------------------------------------------------
def demo_autograd_inplace():  # Playbook §5
    case("§5 autograd inplace error + set_detect_anomaly")
    torch.autograd.set_detect_anomaly(True)
    try:
        x = torch.randn(4, requires_grad=True)
        y = x.sigmoid()
        y += 1.0  # inplace on a tensor needed for backward (the bug)
        y.sum().backward()
    except RuntimeError as e:
        print("caught:", str(e).splitlines()[0][:80])
        print("  fix: use out-of-place (y = y + 1.0) or clone before mutating")
    finally:
        torch.autograd.set_detect_anomaly(False)


# -------------------------------------------------------------------
def demo_retain_graph():  # Playbook §5
    case("§5 backward twice without retain_graph")
    x = torch.randn(4, requires_grad=True)
    y = (x * x).sum()
    y.backward()  # frees the graph
    try:
        y.backward()  # second time -> error
    except RuntimeError as e:
        print("caught:", str(e).splitlines()[0][:80])
        print("  fix: first_backward(retain_graph=True), or recompute y")


# -------------------------------------------------------------------
def demo_memory_leak():  # Playbook §7
    case("§7 'memory leak' from accumulating graph-attached loss")
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = torch.nn.Linear(4096, 4096).to(dev)
    # BAD: total += loss keeps the whole graph (every iter's activations) alive
    total_bad = torch.zeros((), device=dev)
    for _ in range(50):
        out = model(torch.randn(4096, 4096, device=dev)).sum()
        total_bad = total_bad + out  # graph accumulates!
    bad_mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0

    del total_bad
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # GOOD: detach / .item() so no graph is retained
    total_good = 0.0
    for _ in range(50):
        out = model(torch.randn(4096, 4096, device=dev)).sum()
        total_good += out.item()  # or out.detach()
    good_mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0

    print(f"  peak mem  bad(+=loss)={bad_mem:.0f}MB  good(+=item)={good_mem:.0f}MB")
    print("  detection: memory grows each iter -> torch.cuda.memory_summary()/snapshot")
    print("  fix: accumulate scalars with .item()/.detach(), not graph tensors")


# -------------------------------------------------------------------
def demo_cuda_async_note():  # Playbook §3
    case("§3 CUDA async errors (concept)")
    print("  a bad GPU index raises asynchronously -> traceback points elsewhere.")
    print("  detection: run with  CUDA_LAUNCH_BLOCKING=1  to pin the real line,")
    print("             or  compute-sanitizer python your_script.py")
    print("  (not triggered here: a device-side assert aborts the CUDA context)")
    # Safe illustrative version on CPU (synchronous, exact line):
    try:
        t = torch.arange(4)
        _ = t[torch.tensor([10])]  # out of bounds
    except IndexError as e:
        print("  CPU analogue caught synchronously:", str(e).splitlines()[0][:60])


def main():
    print(f"device={dev} | torch {torch.__version__}")
    for fn in (
        demo_shape_dtype_device,
        demo_nan_inf,
        demo_autograd_inplace,
        demo_retain_graph,
        demo_memory_leak,
        demo_cuda_async_note,
    ):
        try:
            fn()
        except Exception as e:
            print(f"[case crashed: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
