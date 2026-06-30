"""Tensor advanced: indexing tricks, permutation patterns, sparse, striding.

Companion script for tensor/tensor.md.
  1. advanced indexing:      gather/scatter/index_select tricks
  2. permute/transpose:     memory layout impact
  3. unfold:                sliding window implementation
  4. sparse COO:            coordinate format sparse tensor
  5. stride tricks:         broadcast via zero-stride

Run:
    python test2.py               # full demo
    python test2.py index          # advanced indexing
    python test2.py permute        # permute and memory layout
    python test2.py unfold         # sliding windows
    python test2.py sparse         # sparse tensors
    python test2.py broadcast      # zero-stride broadcast
"""

import sys
import torch


# ============ 1. Advanced indexing ============
def exp_index():
    print("=" * 60)
    print("1. Advanced indexing: gather, scatter, index_select")
    print("=" * 60)

    x = torch.arange(24).view(4, 6)
    print(f"  Original:\n{x}")

    # index_select: select rows by index
    idx = torch.tensor([0, 2, 3])
    selected = x.index_select(0, idx)
    print(f"\n  index_select(dim=0, index=[0,2,3]):\n{selected}")

    # gather: collect elements along dim using index tensor
    # gather(dim, index) — output[i][j][k] = input[index[i][j][k]][j][k]
    gather_idx = torch.tensor([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2]])
    gathered = x.gather(0, gather_idx)
    print(f"\n  gather(dim=0):\n  index:\n{gather_idx}\n  result:\n{gathered}")

    # scatter_: scatter values to specified indices
    y = torch.zeros(4, 6, dtype=torch.long)
    scatter_idx = torch.tensor([[0, 1, 2], [3, 0, 1]])
    src = torch.tensor([[100, 200, 300], [400, 500, 600]])
    y.scatter_(0, scatter_idx, src)
    print(f"\n  scatter_(dim=0):\n{y}")

    # Fancy indexing with tensors
    row_idx = torch.tensor([0, 1, 3])
    col_idx = torch.tensor([5, 0, 3])
    fancy = x[row_idx, col_idx]
    print(f"\n  fancy indexing x[[0,1,3], [5,0,3]]:\n{fancy}")
    print()


# ============ 2. Permute and memory layout ============
def exp_permute():
    print("=" * 60)
    print("2. Permute: memory layout impact")
    print("=" * 60)

    N, C, H, W = 4, 64, 56, 56
    x = torch.randn(N, C, H, W)

    # NCHW -> NHWC
    x_nhwc = x.permute(0, 2, 3, 1)  # not contiguous
    print(
        f"  NCHW: shape={list(x.shape)}        stride={x.stride()}    contiguous={x.is_contiguous()}"
    )
    print(
        f"  NHWC: shape={list(x_nhwc.shape)}  stride={x_nhwc.stride()}  contiguous={x_nhwc.is_contiguous()}"
    )

    # contiguous after permute makes a copy
    x_nhwc_c = x_nhwc.contiguous()
    print(
        f"  NHWC (contiguous): stride={x_nhwc_c.stride()}  contiguous={x_nhwc_c.is_contiguous()}"
    )
    print(f"  same storage? {x.storage().data_ptr() == x_nhwc_c.storage().data_ptr()}")

    # Performance: operations on non-contiguous can be slower
    if torch.cuda.is_available():
        xc = x.cuda()
        xnc = x_nhwc.cuda()

        def bench(fn, tensor, n=50):
            torch.cuda.synchronize()
            import time

            t0 = time.perf_counter()
            for _ in range(n):
                fn(tensor)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000

        t_nchw = bench(lambda t: t * 2, xc)
        t_nhwc = bench(lambda t: t * 2, xnc)
        print(f"\n  NCHW pointwise: {t_nchw:.3f} ms  (contiguous, fast)")
        print(f"  NHWC pointwise: {t_nhwc:.3f} ms  (strided, may be slower)")
    print()


# ============ 3. Unfold: sliding windows ============
def exp_unfold():
    print("=" * 60)
    print("3. Unfold: sliding window (for convolution/patchify)")
    print("=" * 60)

    # 1D example: sliding window of size 3
    x = torch.arange(10, dtype=torch.float32)
    windows = x.unfold(0, 3, 1)  # dim=0, size=3, step=1
    print(f"  1D x: {x}")
    print(f"  unfold(size=3, step=1):")
    print(f"    shape={list(windows.shape)}")
    print(f"    windows:\n{windows}")

    # 2D image patchify (like ViT)
    img = torch.arange(64).view(8, 8).float()
    print(f"\n  2D image (8x8):")
    # Patches of 4x4 with stride=4
    patches = img.unfold(0, 4, 4).unfold(1, 4, 4)
    print(f"  unfold(0,4,4).unfold(1,4,4): shape={list(patches.shape)}")
    for i in range(2):
        for j in range(2):
            print(f"    patch[{i},{j}] =\n{patches[i, j]}")

    # Check: unfolds are views (zero copy)
    print(
        f"\n  same storage? {img.storage().data_ptr() == patches.storage().data_ptr()}"
    )
    print("  -> unfold creates views, not copies")
    print()


# ============ 4. Sparse tensors ============
def exp_sparse():
    print("=" * 60)
    print("4. Sparse COO tensors")
    print("=" * 60)

    # Create sparse tensor from indices + values
    indices = torch.tensor(
        [
            [0, 1, 2],  # row
            [2, 0, 3],
        ]
    )  # col
    values = torch.tensor([3.0, 4.0, 5.0])
    sp = torch.sparse_coo_tensor(indices, values, (4, 5))

    print(f"  Sparse tensor: shape={list(sp.shape)}, nnz={sp._nnz()}")
    print(f"    indices:\n{sp._indices()}")
    print(f"    values:  {sp._values()}")

    # Convert to dense
    dense = sp.to_dense()
    print(f"    dense:\n{dense}")

    # Sparse + dense operation
    x = torch.randn(4, 5)
    y = sp + x  # implicit densify may happen
    print(f"\n  sp + dense result:\n{y[:2]}")

    # Check if densified
    print(f"  result is_sparse: {y.is_sparse}")

    # Memory: sparse vs dense
    dense_mem = 4 * 5 * 4  # fp32
    sparse_mem = len(values) * 4 + indices.numel() * 8  # index uses int64
    print(f"\n  Dense memory:  {dense_mem} bytes")
    print(f"  Sparse memory: {sparse_mem} bytes")
    print(f"  Ratio:         {sparse_mem / dense_mem:.0%}")
    print()


# ============ 5. Broadcast via zero-stride ============
def exp_broadcast():
    print("=" * 60)
    print("5. Broadcast: zero-stride (内存复用)")
    print("=" * 60)

    a = torch.arange(3, dtype=torch.float32)  # [3]
    b = a.unsqueeze(0).expand(4, -1)  # [4, 3]

    print(f"  a:      shape={list(a.shape)}  stride={a.stride()}")
    print(f"  b = a.expand(4,3): shape={list(b.shape)}  stride={b.stride()}")
    print(f"  b data:\n{b}")
    print(f"  b[0,0] += 100 -> a[0] also changes:")
    a2 = a.clone()
    b2 = a2.unsqueeze(0).expand(4, -1).clone()  # clone -> new storage
    b3 = a2.unsqueeze(0).expand(4, -1)  # view -> same storage
    b3[0, 0] = 999.0
    print(f"    a2 after b3[0,0]=999: {a2}  (shared storage)")
    b2[0, 0] = -1.0
    print(f"    a2 after b2[0,0]=-1:  {a2}  (b2 had its own storage)")

    # Compare: broadcasting saves memory
    large = torch.randn(1, 1024)
    broadcasted = large.expand(1024, 1024)
    cloned = large.expand(1024, 1024).clone()
    print(f"\n  Memory usage:")
    print(f"    broadcast: {large.storage().nbytes()} bytes")
    print(
        f"    clone:     {cloned.storage().nbytes()} bytes  ({cloned.storage().nbytes() // large.storage().nbytes()}x)"
    )
    print("  -> expand uses zero-stride (same storage, no copy)")
    print()


EXPERIMENTS = {
    "index": exp_index,
    "permute": exp_permute,
    "unfold": exp_unfold,
    "sparse": exp_sparse,
    "broadcast": exp_broadcast,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[tensor test2] DONE")


if __name__ == "__main__":
    main()
