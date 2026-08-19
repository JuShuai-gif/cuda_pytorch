"""Discrete-event simulation of LLM request scheduling.

Compares static batching vs continuous batching.  The time model is derived
from the Stage-11 roofline (prefill is compute-bound ~O(S^2), decode is
memory-bound ~O(B*S)); absolute numbers are illustrative, but the *relative*
behavior - continuous batching keeps the batch full and therefore higher
decode throughput and lower TTFT - is the point.

Requests arrive according to a Poisson process; input/output lengths are
lognormal.  Each request goes through prefill (all input tokens at once) then
decode (one token per step).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Request:
    arrive_t: float
    input_len: int
    output_len: int
    generated: int = 0
    ttft: float = 0.0
    finish_t: float = 0.0

    @property
    def seq_len(self) -> int:
        return self.input_len + self.generated


# Model + hardware constants (illustrative; see roofline.py for the model).
L, D = 32, 4096
PEAK_TFLOPS = 60e12        # fp16 tensor-core peak (illustrative)
PEAK_BW = 400e9            # memory bandwidth bytes/s (illustrative)
EFF = 0.5                  # prefill compute efficiency


def prefill_time(input_len: int, batch: int) -> float:
    """Compute-bound prefill of `input_len` tokens for `batch` requests."""
    flops = 2 * L * (12 * D * D + 2 * D * input_len) * input_len * batch
    return flops / (PEAK_TFLOPS * EFF)


def decode_step_time(avg_seq: int, batch: int) -> float:
    """Memory-bound decode step: batch requests each read `avg_seq` KV tokens."""
    # KV cache bytes read per step dominates.
    bytes_read = 2 * L * avg_seq * D * 2 * batch
    return bytes_read / PEAK_BW


def gen_requests(n: int, rate: float, seed: int = 0) -> List[Request]:
    """Poisson arrivals with lognormal input/output lengths."""
    rng = random.Random(seed)
    t = 0.0
    out = []
    for _ in range(n):
        t += rng.expovariate(rate)
        input_len = max(1, int(rng.lognormvariate(math.log(200), 0.5)))
        output_len = max(1, int(rng.lognormvariate(math.log(150), 0.5)))
        out.append(Request(t, input_len, output_len))
    return out


def simulate_static(requests: List[Request], batch_size: int) -> dict:
    """Static batching: wait until a full batch accumulates, then process."""
    t = 0.0
    queue = list(requests)
    done: List[Request] = []
    pending: List[Request] = []

    while queue or pending:
        # Collect arrivals up to now into pending.
        while queue and queue[0].arrive_t <= t:
            pending.append(queue.pop(0))
        if len(pending) < batch_size and queue:
            t = queue[0].arrive_t
            continue
        if not pending:
            break
        # Take a full batch (or whatever is left).
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        max_in = max(r.input_len for r in batch)
        t += prefill_time(max_in, len(batch))
        for r in batch:
            r.ttft = t - r.arrive_t
        # Decode until the shortest-output request finishes.
        remaining = {id(r): r.output_len for r in batch}
        while remaining:
            t += decode_step_time(max_in, len(remaining))
            for r in list(batch):
                if id(r) in remaining:
                    remaining[id(r)] -= 1
                    if remaining[id(r)] <= 0:
                        r.finish_t = t
                        del remaining[id(r)]
            max_in += 1
        done.extend(batch)

    return summarize(done)


def simulate_continuous(requests: List[Request], max_batch: int) -> dict:
    """Continuous batching: requests join/leave the decode batch dynamically."""
    t = 0.0
    queue = list(requests)
    running: List[Request] = []
    done: List[Request] = []

    while queue or running:
        # Admit arrivals that fit within max_batch (prefill one at a time).
        while queue and queue[0].arrive_t <= t and len(running) < max_batch:
            r = queue.pop(0)
            t = max(t, r.arrive_t)
            t += prefill_time(r.input_len, 1)
            r.ttft = t - r.arrive_t
            running.append(r)
        if not running:
            if queue:
                t = queue[0].arrive_t
                continue
            break
        # One decode step for the whole running batch.
        avg_seq = sum(r.seq_len for r in running) // len(running)
        t += decode_step_time(avg_seq, len(running))
        for r in list(running):
            r.generated += 1
            if r.generated >= r.output_len:
                r.finish_t = t
                running.remove(r)
                done.append(r)

    return summarize(done)


def summarize(done: List[Request]) -> dict:
    n = len(done)
    if n == 0:
        return {}
    ttfts = sorted(r.ttft for r in done)
    finish = max(r.finish_t for r in done)
    total_tokens = sum(r.output_len for r in done)
    return {
        "n_requests": n,
        "makespan_s": finish,
        "ttft_p50": ttfts[n // 2],
        "ttft_p95": ttfts[min(n - 1, int(n * 0.95))],
        "throughput_tokens_per_s": total_tokens / finish,
    }
