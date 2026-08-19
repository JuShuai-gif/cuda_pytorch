"""Reliability primitives for a production inference service.

Three mechanisms that together answer "why can't the GPU accept unlimited
requests":

1. Token-bucket rate limiter: caps the *admission rate* so downstream is not
   overwhelmed beyond its capacity.
2. Circuit breaker: when the downstream (worker/GPU) keeps failing, trip and
   fail fast instead of piling up slow retries.
3. Load shedder: a bounded queue that drops overflow instead of letting latency
   and memory grow without bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class TokenBucket:
    """Token-bucket rate limiter (burst allowed up to capacity)."""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate          # tokens per second
        self.capacity = capacity  # burst capacity
        self.tokens = capacity
        self.last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class CircuitBreaker:
    """Three-state circuit breaker: closed / open / half-open."""

    def __init__(self, fail_threshold: int = 3, reset_timeout: float = 1.0):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "closed"      # closed | open | half_open
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.state == "open":
            if time.monotonic() - self.opened_at >= self.reset_timeout:
                self.state = "half_open"
            else:
                return False
        return True

    def record_success(self):
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.fail_threshold:
            self.state = "open"
            self.opened_at = time.monotonic()


class LoadShedder:
    """A bounded queue that drops overflow; tracks admitted/dropped counts."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue: list = []
        self.admitted = 0
        self.dropped = 0

    def try_admit(self, item) -> bool:
        if len(self.queue) >= self.capacity:
            self.dropped += 1
            return False
        self.queue.append(item)
        self.admitted += 1
        return True

    def pop(self):
        return self.queue.pop(0) if self.queue else None

    def __len__(self):
        return len(self.queue)
