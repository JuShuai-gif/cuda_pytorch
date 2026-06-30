"""Tensor 源码分析: TensorImpl 内部, storage 共享, dispatch key 组成。

使用工具: torch._C._dispatch_keys / data_ptr / storage / TensorImpl 属性

运行:
  python test3.py                 # 全链路分析
  python test3.py tensor_impl     # TensorImpl 内部结构
  python test3.py dispatch_keys   # DispatchKeySet 分析
  python test3.py storage_link    # Storage 共享链
  python test3.py autograd_meta   # AutogradMeta 内部

参考源码:
  c10/core/TensorImpl.h          — 核心 tensor 实现
  c10/core/DispatchKeySet.h      — dispatch key 位掩码
  torch/csrc/autograd/variable.h  — AutogradMeta
"""

import sys
import torch
import torch.nn as nn


# ============ 1. TensorImpl 内部结构 ============
def exp_tensor_impl():
    """探究 Tensor 的底层 C++ 结构。"""
    print("=" * 60)
    print("1. TensorImpl 内部: sizes / strides / storage / autograd_meta")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    xt = x.t()

    print(f"  x: shape={list(x.shape)}     stride={x.stride()}")
    print(f"     data_ptr={x.data_ptr():#x}  storage_ptr={x.storage().data_ptr():#x}")
    print(f"     storage_offset={x.storage_offset()}  nbytes={x.storage().nbytes()}")
    print(f"     is_contiguous={x.is_contiguous()}  numel={x.numel()}")
    print()

    print(f"  x.t(): shape={list(xt.shape)}  stride={xt.stride()}")
    print(
        f"         data_ptr={xt.data_ptr():#x}  storage_ptr={xt.storage().data_ptr():#x}"
    )
    print(f"         storage_offset={xt.storage_offset()}")

    print(f"\n  TensorImpl 内部布局 (c10/core/TensorImpl.h):")
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │ Storage (实际数据)              │")
    print(f"  │  data_ptr + nbytes              │")
    print(f"  └─────────────────────────────────┘")
    print(f"          ↑                           ")
    print(f"  ┌───────┴─────────────────────────┐")
    print(f"  │ TensorImpl (metadata)           │")
    print(f"  │  sizes_[] = [2, 3, 4]          │")
    print(f"  │  strides_[] = [12, 4, 1]        │")
    print(f"  │  storage_offset_ = 0            │")
    print(f"  │  numel_ = 24                    │")
    print(f"  │  dtype_, device_, layout_       │")
    print(f"  │  autograd_meta_ (AutogradMeta*) │")
    print(f"  │  key_set_ (DispatchKeySet)      │")
    print(f"  └─────────────────────────────────┘")
    print()


# ============ 2. DispatchKeySet 分析 ============
def exp_dispatch_keys():
    """探究 tensor 内部存储的 DispatchKeySet。"""
    print("=" * 60)
    print("2. DispatchKeySet: tensor 携带的 dispatch 密钥")
    print("=" * 60)

    def show_keys(tensor, label):
        keys = torch._C._dispatch_keys(tensor)
        # keys 是 DispatchKeySet 的字符串表示
        print(f"  {label}:")
        for k in sorted(str(keys).split(", ")):
            print(f"    {k}")

    # 不同 tensor 携带不同的 key set
    x_cpu = torch.randn(2)
    show_keys(x_cpu, "CPU float tensor")

    if torch.cuda.is_available():
        x_cuda = torch.randn(2, device="cuda")
        show_keys(x_cuda, "CUDA float tensor")

    x_grad = torch.randn(2, requires_grad=True)
    show_keys(x_grad, "CPU + requires_grad")

    x_meta = torch.empty(2, device="meta")
    show_keys(x_meta, "Meta tensor")

    print(f"\n  DispatchKeySet 是 64 位位掩码:")
    print(f"  每个 tensor 存储自己的 key_set")
    print(f"  运行时: dispatcher 对所有参数做 OR → 选出最高优先级 key")
    print(f"  TLS: no_grad() 排除 Autograd key, autocast() 包含 Autocast key")
    print()


# ============ 3. Storage 共享链分析 ============
def exp_storage_link():
    """分析多个 tensor 如何共享底层 Storage。"""
    print("=" * 60)
    print("3. Storage 共享: view/slice/transpose 如何复用内存")
    print("=" * 60)

    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    # x storage: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    operations = [
        ("x", x),
        ("x[1]", x[1]),
        ("x[:, 1:3]", x[:, 1:3]),
        ("x.t()", x.t()),
        ("x.view(12)", x.view(12)),
        ("x.reshape(2, 6)", x.reshape(2, 6)),
        ("x.clone()", x.clone()),
        ("x[1].clone()", x[1].clone()),
    ]

    for label, t in operations:
        is_contig = t.is_contiguous()
        same_storage = t.storage().data_ptr() == x.storage().data_ptr()
        offset = t.storage_offset() if same_storage else -1
        nbytes = t.storage().nbytes()
        print(
            f"  {label:20s} shape={list(t.shape):15s} "
            f"same_storage={same_storage} offset={offset:2d} "
            f"nbytes={nbytes:2d} contiguous={is_contig}"
        )

    print(f"\n  view/slice/transpose: 共享 storage, 不拷贝")
    print(f"  clone():            新 storage, 拷贝数据")
    print(f"  reshape:            非 contiguous 时拷贝, 否则 view")
    print()


# ============ 4. AutogradMeta 分析 ============
def exp_autograd_meta():
    """分析 tensor 的 autograd 元数据。"""
    print("=" * 60)
    print("4. AutogradMeta: grad / grad_fn / is_leaf / retain_grad")
    print("=" * 60)

    def inspect_grad(label, t):
        print(f"  {label}:")
        print(f"    requires_grad={t.requires_grad}  is_leaf={t.is_leaf}")
        print(f"    grad_fn={t.grad_fn}")
        print(f"    grad={t.grad}")
        print(
            f"    retains_grad={t._is_zerotensor() if hasattr(t, '_is_zerotensor') else 'N/A'}"
        )

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    inspect_grad("leaf (x)", x)

    y = x * 2
    inspect_grad("\n  non-leaf (y = x*2)", y)

    z = y.sum()
    inspect_grad("\n  output (z = y.sum())", z)

    z.backward()
    inspect_grad("\n  after backward: x", x)
    inspect_grad("\n  after backward: y", y)

    print(f"\n  AutogradMeta 结构 (torch/csrc/autograd/variable.h):")
    print(f"  ┌──────────────────────────────┐")
    print(f"  │ AutogradMeta                 │")
    print(f"  │  grad_         : Tensor      │")
    print(f"  │  grad_fn_      : Node*        │")
    print(f"  │  grad_accumulator_ : Node*    │")
    print(f"  │  requires_grad_ : bool        │")
    print(f"  │  retains_grad_  : bool        │")
    print(f"  │  is_view_       : bool        │")
    print(f"  │  output_nr_     : uint        │")
    print(f"  └──────────────────────────────┘")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_tensor_impl()
        exp_dispatch_keys()
        exp_storage_link()
        exp_autograd_meta()

    print("[Tensor source analysis] DONE")


if __name__ == "__main__":
    main()
