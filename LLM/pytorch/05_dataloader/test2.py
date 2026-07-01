"""DataLoader practical scenarios: pin_memory, persistent_workers, IterableDataset.

Companion script for dataloader/pytorch_dataloader_source_analysis.md.
  1. pin_memory:         verify Pinned memory DMA transfer
  2. persistent_workers: cross-epoch worker reuse
  3. IterableDataset:    infinite stream + sharding
  4. worker_init_fn:     per-worker seed + setup
  5. collate_fn:         variable-length sequences

Run:
    python test2.py                 # full demo
    python test2.py pin             # pin_memory check
    python test2.py persistent      # persistent workers
    python test2.py iterable        # IterableDataset sharding
    python test2.py variable        # variable-length collation
"""

import sys
import time
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# ============ 1. Pin memory DMA verification ============
def exp_pin():
    print("=" * 60)
    print("1. Pin memory: host -> device DMA transfer")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    class SimpleDS(Dataset):
        def __init__(self, n):
            self.data = torch.randn(n, 512)
            self.labels = torch.randint(0, 10, (n,))

        def __getitem__(self, i):
            return self.data[i], self.labels[i]

        def __len__(self):
            return len(self.data)

    ds = SimpleDS(4096)

    # Without pin_memory
    dl_no_pin = DataLoader(ds, batch_size=256, num_workers=2, pin_memory=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in dl_no_pin:
        pass
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # With pin_memory
    dl_pin = DataLoader(ds, batch_size=256, num_workers=2, pin_memory=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for batch in dl_pin:
        _ = batch[0].cuda(), batch[1].cuda()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"  No pin:  {(t1 - t0) * 1000:.1f} ms")
    print(f"  Pin:     {(t3 - t2) * 1000:.1f} ms")

    # Verify tensor is pinned
    sample = next(iter(dl_pin))[0]
    print(f"  Is pinned: {sample.is_pinned()}")
    print(f"  Sample device: {sample.device}")
    print("  -> pin_memory puts tensor in page-locked CPU memory for DMA")
    print()


# ============ 2. Persistent workers ============
def exp_persistent():
    print("=" * 60)
    print("2. Persistent workers: cross-epoch process reuse")
    print("=" * 60)

    class SimpleDS(Dataset):
        def __init__(self, n):
            self.data = torch.randn(n, 64)

        def __getitem__(self, i):
            return self.data[i]

        def __len__(self):
            return len(self.data)

    ds = SimpleDS(256)

    # Non-persistent: restart workers each epoch
    dl = DataLoader(ds, batch_size=32, num_workers=2, persistent_workers=False)
    t0 = time.perf_counter()
    for epoch in range(3):
        for _ in dl:
            pass
    t1 = time.perf_counter()

    # Persistent: reuse workers across epochs
    dl_p = DataLoader(ds, batch_size=32, num_workers=2, persistent_workers=True)
    t2 = time.perf_counter()
    for epoch in range(3):
        for _ in dl_p:
            pass
    t3 = time.perf_counter()

    print(f"  Non-persistent: {(t1 - t0) * 1000:.0f} ms (3 epochs)")
    print(f"  Persistent:     {(t3 - t2) * 1000:.0f} ms (3 epochs)")
    print("  -> persistent workers avoid fork overhead per epoch")

    # Worker IDs visible in worker_init_fn
    seen_ids = set()

    def worker_init(worker_id):
        seen_ids.add(worker_id)

    dl_check = DataLoader(
        ds,
        batch_size=32,
        num_workers=2,
        worker_init_fn=worker_init,
        persistent_workers=True,
    )
    for _ in range(3):
        for _ in dl_check:
            pass
    print(f"  Worker IDs seen: {seen_ids}")
    print()


# ============ 3. IterableDataset with sharding ============
def exp_iterable():
    print("=" * 60)
    print("3. IterableDataset: infinite stream + per-worker sharding")
    print("=" * 60)

    class InfiniteRandStream(IterableDataset):
        """Generate infinite random tensors. Use sharding for multi-worker."""

        def __init__(self, dim, max_samples=None):
            self.dim = dim
            self.max_samples = max_samples

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                start, step, wid = 0, 1, 0
            else:
                start = worker_info.id
                step = worker_info.num_workers
                wid = worker_info.id

            count = 0
            for i in range(start, 10**9, step):
                if self.max_samples and count >= self.max_samples:
                    break
                yield torch.randn(self.dim) * (wid + 1)  # different scale per worker
                count += 1

    ds = InfiniteRandStream(dim=16, max_samples=32)
    dl = DataLoader(ds, batch_size=4, num_workers=2)

    results = []
    for batch in dl:
        results.append(batch)

    # Check: each batch should have consistent magnitude (same worker)
    batch_means = [b.abs().mean().item() for b in results]
    print(f"  Batch abs means: {[f'{m:.2f}' for m in batch_means[:8]]}...")
    print(f"  Unique scales:   {set(round(m) for m in batch_means)}")
    print(
        "  -> each worker produces data with its own scale (worker 0: ~1, worker 1: ~2)"
    )
    print("  -> IterableDataset uses worker_info for per-worker partitioning")
    print()


# ============ 4. Variable-length collation ============
def exp_variable():
    print("=" * 60)
    print("4. Variable-length sequence collation")
    print("=" * 60)

    class VarLenDataset(Dataset):
        def __init__(self):
            lengths = [2, 5, 3, 8, 4, 7, 3, 6, 5, 4]
            self.data = [torch.randn(L, 16) for L in lengths]
            self.labels = [torch.randint(0, 3, (1,)).item() for _ in range(10)]

        def __getitem__(self, i):
            return self.data[i], self.labels[i]

        def __len__(self):
            return len(self.data)

    def pad_collate(batch):
        sequences, labels = zip(*batch)
        # Pad to max length in this batch
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([s.size(0) for s in sequences])
        labels = torch.tensor(labels)
        return padded, lengths, labels

    ds = VarLenDataset()
    dl = DataLoader(ds, batch_size=3, collate_fn=pad_collate)

    for i, (padded, lengths, labels) in enumerate(dl):
        print(
            f"  Batch {i}: seqs padded to {list(padded.shape)}, "
            f"lengths={lengths.tolist()}, labels={labels.tolist()}"
        )
        if i >= 2:
            break

    print("  -> pad_sequence handles variable-length within each batch")
    print()


EXPERIMENTS = {
    "pin": exp_pin,
    "persistent": exp_persistent,
    "iterable": exp_iterable,
    "variable": exp_variable,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[dataloader test2] DONE")


if __name__ == "__main__":
    main()
