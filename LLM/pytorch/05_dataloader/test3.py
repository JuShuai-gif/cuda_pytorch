"""DataLoader 源码分析: multiprocessing worker trace, IPC 队列, pin_memory。

使用工具: multiprocessing.get_context / worker_init_fn /
         DataLoaderIter 内部 / pin_memory 探查

运行:
  python test3.py                 # 全链路分析
  python test3.py worker_trace    # Worker 进程生命周期追踪
  python test3.py ipc_queue       # 进程间队列分析
  python test3.py pin_inside      # pin_memory 内存布局

参考源码:
  torch/utils/data/dataloader.py  — DataLoader + _SingleProcessDataLoaderIter
  torch/utils/data/_utils/worker.py — Worker 循环
  torch/csrc/DataLoader.cpp       — C++ DataLoader 后端
"""

import sys
import time
import torch
from torch.utils.data import DataLoader, Dataset


# ============ 1. Worker 进程生命周期追踪 ============
def exp_worker_trace():
    """追踪 multi-worker DataLoader 的创建和销毁过程。"""
    print("=" * 60)
    print("1. Worker Trace: 进程创建 → 执行 → 销毁")
    print("=" * 60)

    worker_events = []
    import threading

    main_pid = None

    class TraceEvents:
        def __init__(self):
            self.events = []

        def add(self, msg):
            import os

            self.events.append(f"[pid={os.getpid()}] {msg}")

    trace = TraceEvents()

    def worker_init_fn(wid):
        trace.add(f"worker_{wid} INIT")

    class SimpleDS(Dataset):
        def __init__(self, n):
            self.data = torch.randn(n, 16)

        def __getitem__(self, i):
            trace.add(f"worker_getitem({i})")
            return self.data[i]

        def __len__(self):
            return len(self.data)

    trace.add("MAIN creating DataLoader")
    ds = SimpleDS(16)
    dl = DataLoader(ds, batch_size=4, num_workers=2, worker_init_fn=worker_init_fn)

    trace.add("MAIN starting iteration")
    batches = []
    for batch in dl:
        batches.append(batch)
        trace.add(f"MAIN received batch shape={list(batch.shape)}")
        break  # 只取一个 batch

    trace.add("MAIN iteration done")
    del dl  # 触发 worker shutdown
    time.sleep(0.1)  # 等待 worker 退出
    trace.add("MAIN DataLoader deleted")

    for event in trace.events:
        print(f"  {event}")

    print(f"\n  Worker 生命周期:")
    print(f"  1. MAIN 创建 DataLoader → fork N 个 worker 进程")
    print(f"  2. 每个 worker 运行 _worker_loop() (dataloader.py)")
    print(f"  3. while True: idx = index_queue.get() → fetch → result_queue.put()")
    print(f"  4. MAIN 从 result_queue 按序组装 batch")
    print(f"  5. del DataLoader → index_queue.put(None) → worker break → join")
    print()


# ============ 2. IPC 队列分析 ============
def exp_ipc_queue():
    """分析 DataLoader 的进程间通信机制。"""
    print("=" * 60)
    print("2. IPC 队列: index_queue → worker → result_queue")
    print("=" * 60)

    class IDDataset(Dataset):
        def __init__(self, n):
            self.n = n

        def __getitem__(self, i):
            import os

            return {"data": torch.tensor([float(i)]), "worker": os.getpid()}

        def __len__(self):
            return self.n

    ds = IDDataset(16)

    def default_collate_verbose(batch):
        workers = [item["worker"] for item in batch]
        indices = [item["data"].item() for item in batch]
        print(f"    collate: indices={indices} from workers={set(workers)}")
        return torch.tensor([item["data"] for item in batch]).view(-1)

    dl = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=default_collate_verbose)

    print(f"  Collecting 3 batches:")
    for i, batch in enumerate(dl):
        print(f"    Batch {i}: {batch.tolist()}")
        if i >= 2:
            break

    print(f"\n  IPC 数据流 (源码 dataloader.py):")
    print(f"  MAIN → index_queue:  (send_idx, [3, 7, 1, 5])")
    print(f"  Worker: fetch(dataset, indices) → result_queue:  (send_idx, batch)")
    print(f"  MAIN: result_queue.get() → 按 send_idx 保序重排")
    print(f"  _MultiProcessDataLoaderIter 使用:")
    print(f"    - self.send_idx: 已发出任务计数")
    print(f"    - self.rcvd_idx: 期望接收的任务序号")
    print(f"    - self.task_info: {send_idx: data} 乱序缓存")
    print()


# ============ 3. Pin Memory 内存布局探查 ============
def exp_pin_inside():
    """探查 pin_memory 对 tensor 存储的实际影响。"""
    print("=" * 60)
    print("3. Pin Memory: 内存布局与 DMA 传输分析")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    class SmallDS(Dataset):
        def __init__(self, n):
            self.data = torch.randn(n, 256, 256)

        def __getitem__(self, i):
            return self.data[i]

        def __len__(self):
            return len(self.data)

    ds = SmallDS(32)

    # No pin_memory
    dl_cpu = DataLoader(ds, batch_size=4, num_workers=2, pin_memory=False)
    sample_cpu = next(iter(dl_cpu))
    print(f"  No pin_memory:")
    print(f"    is_pinned: {sample_cpu.is_pinned()}")
    print(f"    device:    {sample_cpu.device}")

    # With pin_memory
    dl_pin = DataLoader(ds, batch_size=4, num_workers=2, pin_memory=True)
    sample_pin = next(iter(dl_pin))
    print(f"\n  With pin_memory:")
    print(f"    is_pinned: {sample_pin.is_pinned()}")
    print(f"    device:    {sample_pin.device}")

    # Pinned memory = page-locked host memory
    # CPU pageable memory cannot do DMA; pinned can
    print(f"\n  Pin Memory 原理:")
    print(f"  CPU 普通内存 (pageable):")
    print(f"    → GPU DMA engine 无法直接访问")
    print(f"    → 需要 driver 先拷贝到 staging buffer")
    print(f"  CPU Pinned (page-locked) 内存:")
    print(f"    → GPU DMA engine 直接访问")
    print(f"    → 省一次 CPU 侧的 staging copy")
    print(f"    → 但过多 pinned memory 会拖慢系统 (物理页不可换出)")
    print(f"  源码: torch/csrc/DataLoader.cpp → pin_memory 实现")
    print()


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else []
    if exps:
        for name in exps:
            globals()[f"exp_{name}"]()
    else:
        exp_worker_trace()
        exp_ipc_queue()
        exp_pin_inside()

    print("[DataLoader source analysis] DONE")


if __name__ == "__main__":
    main()
