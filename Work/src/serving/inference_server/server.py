"""A minimal inference server: request queue + worker + batching.

The core of every inference service, stripped of HTTP so the scheduling is
visible.  A single worker thread drains a request queue, assembles batches
according to a strategy, runs the model, and returns results per request.

Batching strategies:
  no_batch    - each request runs alone (batch=1), lowest latency, lowest GPU util
  static      - wait until max_batch requests accumulate, then run together
  dynamic     - assemble a batch, but flush after max_wait even if not full

The server also caps the queue length; when full, new requests are rejected
(the load-shedding primitive, expanded in Stage 17).
"""

from __future__ import annotations

import queue
import threading
import time

import torch


class InferenceServer:
    def __init__(self, model: torch.nn.Module, device, *, strategy: str = "dynamic",
                 max_batch: int = 8, max_wait: float = 0.005, max_queue: int = 128):
        self.model = model.to(device).eval()
        self.device = device
        self.strategy = strategy
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.in_queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ---- client API ----------------------------------------------------
    def infer(self, x: torch.Tensor, timeout: float = 10.0) -> torch.Tensor:
        """Submit one request and block for its result."""
        result: dict = {}
        done = threading.Event()
        try:
            self.in_queue.put((x, done, result), timeout=timeout)
        except queue.Full:
            raise RuntimeError("server overloaded: queue full")
        if not done.wait(timeout):
            raise TimeoutError("inference timed out")
        return result["out"]

    # ---- worker --------------------------------------------------------
    def _worker_loop(self):
        while not self._stop.is_set():
            batch = self._collect_batch()
            if batch is None:
                continue
            self._run_batch(batch)

    def _collect_batch(self):
        if self.strategy == "no_batch":
            try:
                return [self.in_queue.get(timeout=None)]
            except queue.Empty:
                return None

        if self.strategy == "static":
            # Wait until a full batch accumulates (blocking).
            items = []
            for _ in range(self.max_batch):
                try:
                    items.append(self.in_queue.get(timeout=None))
                except queue.Empty:
                    break
            return items if items else None

        # dynamic: wait up to max_wait for the first request, then drain.
        try:
            first = self.in_queue.get(timeout=self.max_wait)
        except queue.Empty:
            return None
        items = [first]
        while len(items) < self.max_batch:
            try:
                items.append(self.in_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _run_batch(self, items):
        xs = [x for x, _, _ in items]
        batch = torch.stack(xs).to(self.device)
        with torch.no_grad():
            out = self.model(batch)
        for (_, done, result), o in zip(items, out):
            result["out"] = o.cpu()
            done.set()

    def stop(self):
        self._stop.set()


def make_model(hidden: int = 512, layers: int = 4):
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                *[nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
                  for _ in range(layers)],
                nn.Linear(hidden, hidden),
            )

        def forward(self, x):
            return self.net(x)

    return MLP()
