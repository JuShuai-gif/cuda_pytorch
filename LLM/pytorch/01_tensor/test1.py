"""Tensor memory model demo: view, stride, contiguous, storage, as_strided.

Companion script for tensor/tensor.md. Covers:
  1. view vs reshape vs contiguous: zero-copy vs copy semantics
  2. stride & memory layout:   how strides relate to sizes
  3. storage sharing:          multiple tensors sharing same data
  4. as_strided:               custom stride views (dangerous)
  5. inplace view trap:        in-place op on view corrupts source
  6. is_contiguous / channels_last: memory format checks
  7. channels_last perf:       NHWC vs NCHW speed comparison
  8. requires_grad_ & detach:  autograd metadata + non-leaf grad trap
  9. stride debug print:       tool to understand any tensor's memory layout

Run:
    python test1.py                 # full demo
    python test1.py view            # view vs reshape vs contiguous
    python test1.py stride          # stride & memory layout
    python test1.py storage         # shared storage + inplace trap
    python test1.py as_strided      # as_strided views
    python test1.py channels       # channels_last format + perf
    python test1.py gradient        # requires_grad, detach, retain_grad
    python test1.py debug_stride    # stride debug tool (pass any shape)

=== DEBUG 常见问题 ===
  Q: RuntimeError: view size is not compatible...?
  A: tensor 非 contiguous, 用 x.contiguous().view(...) 或 x.reshape(...)

  Q: 修改 view 后原 tensor 也变了?
  A: view/slice/transpose 共享 storage, 需要独立副本用 x.clone()

  Q: in-place op 报错 "a view of a leaf Variable that requires grad"?
  A: 不能对 requires_grad=True 的叶子 tensor 做 in-place;
     用 x.data.mul_(...) 或者先 detach()

  Q: tensor.stride() 的值怎么看?  python test1.py debug_stride
  A: stride[i] = 从当前元素到下一维相邻元素在 storage 中的偏移量(元素数)
     例: shape=(2,3,4), stride=(12,4,1): 每加 [1,0,0] 跳 12, [0,1,0] 跳 4
"""

import sys

import torch


# ============ 1. view vs reshape vs contiguous ============
def exp_view():
    print("=" * 60)
    print("1. view vs reshape vs contiguous")
    print("=" * 60)

    x = torch.arange(12).view(3, 4)
    print(f"  Original: shape={list(x.shape)} strides={x.stride()}")

    # view: zero-copy, must satisfy stride constraints
    y = x.view(6, 2)
    print(f"  view(6,2): shape={list(y.shape)} strides={y.stride()}")
    print(f"  same storage? {x.storage().data_ptr() == y.storage().data_ptr()}")
    print(f"  same data_ptr? {x.data_ptr() == y.data_ptr()}")

    # Transpose breaks contiguity
    xt = x.t()  # or x.transpose(0, 1)
    print(f"\n  After transpose:")
    print(f"  shape={list(xt.shape)} strides={xt.stride()}")
    print(f"  is_contiguous: {xt.is_contiguous()}")

    # view fails on non-contiguous
    try:
        xt.view(12)
    except RuntimeError as e:
        print(f"  view(12) on transpose: ERROR ({str(e)[:60]})")

    # But reshape works (copies if needed)
    z = xt.reshape(12)
    print(f"  reshape(12) on transpose: OK, shape={list(z.shape)}")
    print(
        f"  same storage after reshape? {xt.storage().data_ptr() == z.storage().data_ptr()}"
    )

    # contiguous() makes a copy
    xc = xt.contiguous()
    print(f"\n  contiguous():")
    print(f"  shape={list(xc.shape)} strides={xc.stride()}")
    print(f"  same storage? {xt.storage().data_ptr() == xc.storage().data_ptr()}")
    print()


# ============ 2. Stride & memory layout ============
def exp_stride():
    print("=" * 60)
    print("2. Stride & memory layout")
    print("=" * 60)

    x = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    print(f"  Float32 tensor: shape={list(x.shape)}")
    print(f"  strides (elements): {x.stride()}")
    print(f"  strides (bytes):    {tuple(s * x.element_size() for s in x.stride())}")

    # element(i, j, k) at offset = i*12 + j*4 + k*1
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                offset = i * 12 + j * 4 + k
                print(
                    f"    x[{i},{j},{k}] = {x[i, j, k].item():.0f}   (offset={offset})"
                )

    print(f"\n  is_contiguous:  {x.is_contiguous()}")
    print(
        f"  is_contiguous(channels_last): {x.is_contiguous(memory_format=torch.channels_last)}"
    )

    # Contiguous formula: stride[i] = stride[i+1] * size[i+1]
    print(f"  Check: stride[2]=1, stride[1]=4=1*4, stride[0]=12=4*3")

    # Slice: view with offset (same strides, different storage_offset)
    s = x[:, :, 1:3]
    print(f"\n  Slice [:, :, 1:3]: shape={list(s.shape)} strides={s.stride()}")
    print(f"  storage_offset: {s.storage_offset()}")
    print(f"  is_contiguous:  {s.is_contiguous()}")
    print()


# ============ 3. Shared storage ============
def exp_storage():
    print("=" * 60)
    print("3. Shared storage: multiple tensors, same data")
    print("=" * 60)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    print(f"  Original: {x}")
    print(f"  data_ptr: {x.data_ptr():#x}")

    # View shares storage
    y = x.view(2, 3)
    z = x[2:5]
    print(f"\n  view:   {y.tolist()}")
    print(f"  slice:  {z.tolist()}")
    print(f"  same storage (x,y): {x.storage().data_ptr() == y.storage().data_ptr()}")
    print(f"  same storage (x,z): {x.storage().data_ptr() == z.storage().data_ptr()}")

    # Modify via view modifies original (shared storage!)
    y[0, 0] = 99.0
    print(f"\n  After y[0,0]=99:")
    print(f"  x: {x}")
    print(f"  y: {y.tolist()}")
    print("  -> view/slice writes are visible through original (shared storage)")

    # clone creates new storage
    c = x.clone()
    print(f"\n  After clone():")
    print(
        f"  same storage (x, clone)? {x.storage().data_ptr() == c.storage().data_ptr()}"
    )

    # detach shares storage but breaks autograd
    x.requires_grad_(True)
    d = x.detach()
    print(
        f"\n  detach: requires_grad={d.requires_grad}, same storage={x.storage().data_ptr() == d.storage().data_ptr()}"
    )

    # BUG: in-place on view corrupts source (common pitfall)
    print(f"\n  === In-place view trap ===")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v = a.view(2, 3)
    v[0, 0] = 99.0  # normal assignment — ok, shared storage
    print(f"  After v[0,0]=99:   a={a}")

    a2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v2 = a2.view(2, 3)
    v2 += 1.0  # in-place op — ALSO modifies a2!
    print(f"  After v2+=1:       a2={a2}")
    print("  -> in-place ops on views modify the source tensor silently")
    print("  -> Fix: use v = x.view(...).clone() if you need independent copy")


# ============ 4. as_strided views ============
def exp_as_strided():
    print("=" * 60)
    print("4. as_strided: custom stride views")
    print("=" * 60)

    x = torch.arange(8, dtype=torch.float32)
    print(f"  Original: {x}")

    # Even-indexed elements: stride=2
    even = torch.as_strided(x, (4,), (2,))
    print(f"  stride=2 view (even indices): {even}")

    # 2x2 sliding window: stride=3 not a multiple of original dims
    # This can produce out-of-bounds reads!
    print(f"\n  WARNING: as_strided can create out-of-bounds access")
    try:
        dangerous = torch.as_strided(x, (4,), (3,))
        print(f"  stride=3 (out-of-bounds!): {dangerous}")
    except Exception as e:
        print(f"  stride=3 failed: {e}")

    # Safe usage: broadcasting with expanded strides
    x2d = torch.arange(6).view(2, 3)
    print(f"\n  2D tensor:\n{x2d}")

    # Column view: stride=(3, 1) is contiguous
    col = torch.as_strided(x2d, (6,), (1,))
    print(f"  Flattened (stride=1): {col}")

    # Every other row element: shape=(2,2), stride=(3, 2)
    # NOT contiguous
    sub = torch.as_strided(x2d, (2, 2), (3, 2))
    print(f"  Every 2nd element: \n{sub}")
    print(f"  is_contiguous: {sub.is_contiguous()}")
    print()


# ============ 5. requires_grad / detach / retain_grad ============
def exp_gradient():
    print("=" * 60)
    print("5. requires_grad, detach, retain_grad")
    print("=" * 60)

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"  Leaf tensor:  requires_grad={x.requires_grad}, is_leaf={x.is_leaf()}")

    y = x * 2 + 1  # non-leaf
    z = y.sum()
    print(f"  y = x*2+1:    requires_grad={y.requires_grad}, is_leaf={y.is_leaf()}")
    print(f"  y.grad before backward: {y.grad}")

    z.backward()
    print(f"  x.grad:  {x.grad}")  # [2, 2]

    # Non-leaf tensors don't save .grad by default
    print(f"  y.grad after backward: {y.grad}")  # None

    # retain_grad to save non-leaf gradient
    x2 = torch.tensor([1.0, 1.0], requires_grad=True)
    y2 = x2 * 3
    y2.retain_grad()
    y2.sum().backward()
    print(f"\n  With retain_grad: y2.grad = {y2.grad}")

    # detach: share storage, break autograd
    x3 = torch.tensor([5.0], requires_grad=True)
    d = x3.detach()
    print(f"\n  detach:")
    print(f"    requires_grad: {d.requires_grad}")
    print(f"    same storage:  {x3.storage().data_ptr() == d.storage().data_ptr()}")

    # Only leaf tensors can require_grad=True
    w = torch.tensor([1.0])
    v = w * 2
    try:
        v.requires_grad_(True)
    except RuntimeError as e:
        print(f"\n  v.requires_grad_(True) on non-leaf: ERROR ({str(e)[:80]})")
    v.requires_grad_(False)
    print(f"  v.requires_grad_(False) on non-leaf: OK")

    # Traps: non-leaf grad is None by default
    print(f"\n  Non-leaf grad trap:")
    z2 = (w * 2).sum()
    z2.backward()
    intermediate = w * 2
    print(f"    intermediate.grad after backward: {intermediate.grad}")
    print("    -> non-leaf tensors discard grad after backward")
    print("    -> use .retain_grad() to keep it")
    print()


# ============ 6. channels_last memory format ============
def exp_channels():
    print("=" * 60)
    print("6. channels_last (NHWC) vs contiguous (NCHW)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available for perf comparison")
        return

    N, C, H, W = 32, 64, 56, 56
    x_nchw = torch.randn(N, C, H, W, device="cuda")
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)

    print(
        f"  NCHW:  shape={list(x_nchw.shape)} stride={x_nchw.stride()} contiguous={x_nchw.is_contiguous()}"
    )
    print(
        f"  NHWC:  shape={list(x_nhwc.shape)} stride={x_nhwc.stride()} contiguous={x_nhwc.is_contiguous(memory_format=torch.channels_last)}"
    )

    # Conv2d perf comparison
    conv = torch.nn.Conv2d(C, C, 3, padding=1).cuda()

    for _ in range(5):
        conv(x_nchw)
    torch.cuda.synchronize()

    import time

    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        conv(x_nchw)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(n_iter):
        conv(x_nhwc)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"\n  NCHW conv: {(t1 - t0) * 1000 / n_iter:.3f} ms")
    print(f"  NHWC conv: {(t3 - t2) * 1000 / n_iter:.3f} ms")
    print(f"  speedup:   {(t1 - t0) / (t3 - t2):.2f}x with NHWC")
    print("  -> channels_last enables tensor-core-friendly memory layout")
    print()


# ============ 7. Stride debug tool ============
def _stride_debug(tensor_or_shape, name="tensor"):
    """print memory layout of a tensor in human-readable format"""
    if isinstance(tensor_or_shape, torch.Tensor):
        t = tensor_or_shape
    else:
        t = torch.arange(
            1, int(torch.tensor(tensor_or_shape).prod().item()) + 1, dtype=torch.float32
        ).view(tensor_or_shape)

    print(f"  {name}:")
    print(f"    shape:    {list(t.shape)}")
    print(f"    stride:   {tuple(t.stride())}")
    print(f"    elements: {t.numel()}")
    print(f"    bytes:    {t.numel() * t.element_size()}")

    flat = t.flatten()
    for idx in range(min(t.numel(), 12)):
        multi_idx = []
        temp = idx
        for s in reversed(t.shape):
            multi_idx.insert(0, temp % s)
            temp //= s
        offset = sum(mi * st for mi, st in zip(multi_idx, t.stride()))
        print(
            f"    index{tuple(multi_idx):>15s}  storage[{offset:>3d}]  = {flat[idx].item():.0f}"
        )

    print(f"    contiguous: {t.is_contiguous()}")


def exp_debug_stride():
    print("=" * 60)
    print("7. Stride debug tool: understand any tensor layout")
    print("=" * 60)

    # Example 1: simple 2D
    _stride_debug((2, 4), "simple 2D")

    # Example 2: transposed
    x = torch.arange(6).view(2, 3)
    _stride_debug(x.t(), "x.t() (transposed)")

    # Example 3: slice
    x = torch.arange(24).view(2, 3, 4)
    _stride_debug(x[:, 1:, :2], "x[:, 1:, :2] (sliced)")

    print("\n  Usage: pass any shape or tensor to _stride_debug() to inspect layout")
    print("  Key: stride[i] = elements to skip when index[i] increments by 1")
    print()


EXPERIMENTS = {
    "view": exp_view,
    "stride": exp_stride,
    "storage": exp_storage,
    "as_strided": exp_as_strided,
    "channels": exp_channels,
    "gradient": exp_gradient,
    "debug_stride": exp_debug_stride,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[tensor demo] DONE")


if __name__ == "__main__":
    main()
