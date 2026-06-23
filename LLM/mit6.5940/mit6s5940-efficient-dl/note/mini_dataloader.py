"""从零实现一个 mini PyTorch DataLoader, 复刻其核心机制。

覆盖: Dataset / Sampler / BatchSampler / Fetcher / collate / 单进程 /
多进程(prefetch + 保序重排) / pin_memory。仅依赖 torch + 标准库。
"""

import itertools
import queue
import torch
import torch.multiprocessing as mp


# ============ 1. Dataset ============
class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# ============ 2. Sampler ============
class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler:
    def __init__(self, data_source, generator=None):
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        yield from torch.randperm(n, generator=self.generator).tolist()

    def __len__(self):
        return len(self.data_source)


class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        it = iter(self.sampler)
        if self.drop_last:
            args = [it] * self.batch_size  # zip 同一迭代器, 自动丢弃不足一批的尾部
            for batch in zip(*args):
                yield list(batch)
        else:
            while True:
                batch = list(itertools.islice(it, self.batch_size))
                if not batch:
                    break
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# ============ 3. collate ============
def default_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, dict):
        return {k: default_collate([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, (tuple, list)):
        # 转置: [(img, label), (img, label)] -> [imgs, labels]
        return [default_collate(samples) for samples in zip(*batch)]
    raise TypeError(f"不支持的类型: {type(elem)}")


# ============ 4. Fetcher ============
class MapDatasetFetcher:
    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def fetch(self, batched_index):
        data = [self.dataset[i] for i in batched_index]
        return self.collate_fn(data)


# ============ 5. pin_memory ============
def pin_memory(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    elif isinstance(data, dict):
        return {k: pin_memory(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(pin_memory(x) for x in data)
    return data


# ============ 6. DataLoader ============
class MiniDataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        drop_last=False,
        pin_memory=False,
        prefetch_factor=2,
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessIter(self)
        return _MultiProcessIter(self)


# ============ 7. 单进程迭代器 ============
class _SingleProcessIter:
    def __init__(self, loader):
        self.loader = loader
        self.sampler_iter = iter(loader.batch_sampler)
        self.fetcher = MapDatasetFetcher(loader.dataset, loader.collate_fn)

    def __iter__(self):
        return self

    def __next__(self):
        index = next(self.sampler_iter)  # 取一批索引(StopIteration 自然终止)
        data = self.fetcher.fetch(index)  # 取样本 + collate
        if self.loader.pin_memory:
            data = pin_memory(data)
        return data


# ============ 8. 多进程: worker 循环 ============
def _worker_loop(dataset, collate_fn, index_queue, result_queue, worker_id, base_seed):
    torch.set_num_threads(1)
    seed = base_seed + worker_id
    torch.manual_seed(seed)
    fetcher = MapDatasetFetcher(dataset, collate_fn)
    while True:
        r = index_queue.get()
        if r is None:  # 收到结束信号
            break
        idx, batched_index = r
        try:
            data = fetcher.fetch(batched_index)
        except Exception as e:
            data = e  # 把异常传回主进程
        result_queue.put((idx, data))


# ============ 9. 多进程迭代器 ============
class _MultiProcessIter:
    def __init__(self, loader):
        self.loader = loader
        self.sampler_iter = iter(loader.batch_sampler)
        self.num_workers = loader.num_workers
        self.prefetch = loader.prefetch_factor

        ctx = mp.get_context("spawn")
        self.result_queue = ctx.Queue()
        self.index_queues = []
        self.workers = []
        base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        for wid in range(self.num_workers):
            iq = ctx.Queue()
            w = ctx.Process(
                target=_worker_loop,
                args=(
                    loader.dataset,
                    loader.collate_fn,
                    iq,
                    self.result_queue,
                    wid,
                    base_seed,
                ),
                daemon=True,
            )
            w.start()
            self.index_queues.append(iq)
            self.workers.append(w)

        # 保序重排所需的游标
        self.send_idx = 0  # 已派发的任务序号
        self.rcvd_idx = 0  # 期望接收的下一个序号
        self.task_info = {}  # send_idx -> worker_id 或 (worker_id, data)
        self.tasks_outstanding = 0
        self.worker_cycle = itertools.cycle(range(self.num_workers))
        self.exhausted = False

        # 预填充: 维持 prefetch * num_workers 个飞行中任务
        for _ in range(self.prefetch * self.num_workers):
            self._try_put_index()

    def __iter__(self):
        return self

    def _try_put_index(self):
        if self.exhausted:
            return
        try:
            index = next(self.sampler_iter)
        except StopIteration:
            self.exhausted = True
            return
        wid = next(self.worker_cycle)  # 轮询派发给各 worker
        self.index_queues[wid].put((self.send_idx, index))
        self.task_info[self.send_idx] = wid
        self.tasks_outstanding += 1
        self.send_idx += 1

    def __next__(self):
        while True:
            # 已按序备好的结果直接返回
            info = self.task_info.get(self.rcvd_idx)
            if isinstance(info, tuple):  # (worker_id, data) 已到
                _, data = self.task_info.pop(self.rcvd_idx)
                self.rcvd_idx += 1
                self._process(data)
                return self._finalize(data)

            if self.tasks_outstanding == 0:  # 没有飞行中任务 -> 结束
                self._shutdown()
                raise StopIteration

            # 从结果队列拿一个(可能乱序), 存入 task_info
            idx, data = self.result_queue.get()
            self.tasks_outstanding -= 1
            if idx == self.rcvd_idx:
                self.task_info.pop(idx)
                self.rcvd_idx += 1
                self._process(data)
                return self._finalize(data)
            else:
                wid = self.task_info[idx]
                self.task_info[idx] = (wid, data)  # 提前到达, 暂存

    def _process(self, data):
        if isinstance(data, Exception):
            self._shutdown()
            raise data
        self._try_put_index()  # 取走一个就补一个, 维持满载

    def _finalize(self, data):
        if self.loader.pin_memory:
            return pin_memory(data)
        return data

    def _shutdown(self):
        for iq in self.index_queues:
            iq.put(None)
        for w in self.workers:
            w.join(timeout=5)


# ============ 测试 ============
class ToyDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __getitem__(self, i):
        return torch.tensor([i, i * 10], dtype=torch.float32), i % 3

    def __len__(self):
        return self.n


def _run():
    ds = ToyDataset(10)

    print("--- 单进程 ---")
    dl = MiniDataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    print("len:", len(dl))
    for feats, labels in dl:
        print("feats", feats.shape, "labels", labels.tolist())

    print("\n--- 多进程(2 workers, 保序) ---")
    dl = MiniDataLoader(ds, batch_size=3, shuffle=False, num_workers=2)
    for feats, labels in dl:
        print("feats", feats.shape, "labels", labels.tolist())


if __name__ == "__main__":
    _run()
