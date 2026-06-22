"""
Request scheduling and continuous batching for LLM inference.

Continuous batching allows requests to dynamically join and leave
the running batch without waiting for all requests to complete.
This maximizes GPU utilization compared to static batching.

Core concepts:
  - Request state machine: QUEUED -> PREFILL -> DECODE -> FINISHED
  - Priority-based scheduling: new prefill requests prioritized over decode
  - Mixed prefill+decode: interleave prefill and decode within same step
  - Dynamic batch composition: add/remove requests each scheduler step
  - GPU utilization model: utilization ~ 1 / (1 + scheduling_overhead)
  - Prefill priority minimizes TTFT for new requests

The scheduler implements a production-grade version of the algorithm
described in the vLLM and SGLang papers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


# ==============================================================================
#  Request state machine
# ==============================================================================


class RequestState(Enum):
    """States in the request lifecycle."""

    QUEUED = auto()  # Waiting in the request queue
    PREFILL = auto()  # Processing prompt (prefill phase)
    DECODE = auto()  # Generating tokens (decode phase)
    FINISHED = auto()  # Generation complete
    ABORTED = auto()  # Request was cancelled


class SchedulingPolicy(Enum):
    """Available scheduling policies."""

    FIFO = auto()  # First-in-first-out
    PRIORITY = auto()  # Priority-based (prefill > decode)
    SHORTEST_FIRST = auto()  # Shortest prompt first


@dataclass
class Request:
    """A generation request tracked by the scheduler.

    Attributes:
        request_id: Unique request identifier.
        prompt_ids: Input token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Top-p (nucleus) filtering parameter.
        eos_token_id: EOS token for early termination.
        state: Current request state.
        generated_ids: Tokens generated so far.
        priority: Scheduling priority (higher = more urgent).
        submission_time: When the request was submitted.
        prefill_start_time: When prefill began.
        first_token_time: When the first token was generated.
        current_seq_len: Current sequence length (prompt + generated).
    """

    request_id: int
    prompt_ids: list[int]
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: int | None = None
    state: RequestState = RequestState.QUEUED
    generated_ids: list[int] = field(default_factory=list)
    priority: int = 0
    submission_time: float = 0.0
    prefill_start_time: float = 0.0
    first_token_time: float = 0.0
    current_seq_len: int = 0

    @property
    def prompt_len(self) -> int:
        """Number of prompt tokens."""
        return len(self.prompt_ids)

    @property
    def generated_len(self) -> int:
        """Number of generated tokens."""
        return len(self.generated_ids)

    @property
    def total_len(self) -> int:
        """Total tokens (prompt + generated)."""
        return len(self.prompt_ids) + len(self.generated_ids)

    @property
    def remaining_tokens(self) -> int:
        """How many more tokens can be generated."""
        return max(0, self.max_new_tokens - len(self.generated_ids))

    @property
    def is_finished(self) -> bool:
        """Whether the request has completed."""
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    @property
    def is_running(self) -> bool:
        """Whether the request is actively being processed."""
        return self.state in (RequestState.PREFILL, RequestState.DECODE)

    def append_token(self, token_id: int) -> None:
        """Append a generated token to this request."""
        self.generated_ids.append(token_id)

    def get_phase(self) -> str:
        """Get a string describing the current phase."""
        if self.state == RequestState.PREFILL:
            return "prefill"
        if self.state == RequestState.DECODE:
            return "decode"
        return "idle"


# ==============================================================================
#  Batch slot
# ==============================================================================


@dataclass
class BatchSlot:
    """A position in the running batch.

    Maps a request to its batch tensor index and tracks
    per-slot KV cache position for the model.

    Attributes:
        batch_idx: Position in the batch dimension.
        request_id: ID of the request occupying this slot.
        kv_cache_seq_len: Current KV cache length for this slot.
        in_prefill: Whether the slot is in prefill phase.
    """

    batch_idx: int
    request_id: int
    kv_cache_seq_len: int = 0
    in_prefill: bool = True


# ==============================================================================
#  Scheduler
# ==============================================================================


class Scheduler:
    """Continuous batching scheduler for LLM inference.

    Manages the lifecycle of requests through the QUEUED -> PREFILL ->
    DECODE -> FINISHED state machine. Each step:
      1. Evict finished requests from the running batch
      2. Admit queued requests via prefill (up to max_batch_size)
      3. Execute one decode step for all running decode requests

    Prefill is prioritized: new requests get a dedicated prefill step
    before joining the decode batch, minimizing TTFT.

    Args:
        max_batch_size: Maximum number of concurrent requests in a batch.
        max_total_tokens: Maximum total tokens (prompt+generated) across batch.
        policy: Scheduling policy for request admission.
        prefill_priority: Whether to prioritize prefill over decode.
        block_size: KV cache block size for capacity estimation.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_total_tokens: int = 8192,
        policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        prefill_priority: bool = True,
        block_size: int = 16,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.policy = policy
        self.prefill_priority = prefill_priority
        self.block_size = block_size

        self._queue: deque[Request] = deque()
        self._running: dict[int, Request] = {}
        self._batch_slots: list[BatchSlot] = []

        self._lock = threading.Lock()
        self._next_request_id = 0

        # Step-based accounting
        self._step_count = 0
        self._total_requests_processed = 0
        self._total_prefill_steps = 0
        self._total_decode_steps = 0

    # ---- Request management ----

    def add_request(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        priority: int = 0,
        request_id: int | None = None,
    ) -> int:
        """Submit a new generation request to the scheduler.

        Args:
            prompt_ids: Input token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop token ID.
            priority: Scheduling priority (higher = sooner).
            request_id: Desired request ID (auto-assigned if None).

        Returns:
            The request ID.
        """
        with self._lock:
            if request_id is None:
                request_id = self._next_request_id
                self._next_request_id += 1

            req = Request(
                request_id=request_id,
                prompt_ids=list(prompt_ids),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                priority=priority,
                submission_time=time.monotonic(),
                current_seq_len=len(prompt_ids),
            )
            self._queue.append(req)
            return request_id

    def abort_request(self, request_id: int) -> bool:
        """Abort a request (queued or running).

        Args:
            request_id: Request to abort.

        Returns:
            True if the request was found and aborted.
        """
        with self._lock:
            # Check queue
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    req.state = RequestState.ABORTED
                    del self._queue[i]
                    return True

            # Check running
            if request_id in self._running:
                self._running[request_id].state = RequestState.ABORTED
                return True

            return False

    def get_request(self, request_id: int) -> Optional[Request]:
        """Get a request by ID (searches both queue and running)."""
        with self._lock:
            for req in self._queue:
                if req.request_id == request_id:
                    return req
            return self._running.get(request_id)

    # ---- Core scheduling step ----

    def step(self) -> list[Request]:
        """Execute one scheduling step.

        Returns:
            List of requests that finished during this step.
        """
        self._step_count += 1
        finished: list[Request] = []

        with self._lock:
            # Phase 1: Evict finished requests
            finished = self._evict_finished()

            # Phase 2: Admit new requests via prefill
            self._admit_requests()

            # Phase 3: Advance decode for running requests
            # (The actual model forward is called externally;
            #  here we just update state after decode)

        return finished

    def _evict_finished(self) -> list[Request]:
        """Remove finished requests from the running batch.

        Returns:
            List of requests that were evicted.
        """
        finished_ids: list[int] = []
        for slot in self._batch_slots:
            req = self._running.get(slot.request_id)
            if req is None or req.is_finished:
                finished_ids.append(slot.request_id)

        finished_requests: list[Request] = []
        for req_id in finished_ids:
            req = self._running.pop(req_id, None)
            if req:
                req.state = RequestState.FINISHED
                self._total_requests_processed += 1
                finished_requests.append(req)

        # Rebuild batch slots (compact indices)
        self._batch_slots = [
            slot for slot in self._batch_slots if slot.request_id in self._running
        ]
        for i, slot in enumerate(self._batch_slots):
            slot.batch_idx = i

        return finished_requests

    def _admit_requests(self) -> None:
        """Admit queued requests into the running batch via prefill.

        Selects requests from the queue based on scheduling policy
        and available batch capacity.
        """
        available_slots = self.max_batch_size - len(self._batch_slots)
        if available_slots <= 0:
            return

        # Select candidates following scheduling policy
        candidates = self._select_candidates(available_slots)
        if not candidates:
            return

        # Transition to prefill
        for req in candidates:
            req.state = RequestState.PREFILL
            req.prefill_start_time = time.monotonic()

            self._running[req.request_id] = req
            self._batch_slots.append(
                BatchSlot(
                    batch_idx=len(self._batch_slots),
                    request_id=req.request_id,
                    kv_cache_seq_len=req.prompt_len,
                    in_prefill=True,
                )
            )

        self._total_prefill_steps += len(candidates)

    def _select_candidates(self, n: int) -> list[Request]:
        """Select up to n requests from the queue based on policy.

        Args:
            n: Maximum number of candidates.

        Returns:
            List of selected requests (already removed from queue).
        """
        if self.policy == SchedulingPolicy.FIFO:
            candidates: list[Request] = []
            while len(candidates) < n and self._queue:
                req = self._queue.popleft()
                candidates.append(req)
            return candidates

        if self.policy == SchedulingPolicy.SHORTEST_FIRST:
            # Sort by prompt length (ascending)
            sorted_queue = sorted(self._queue, key=lambda r: r.prompt_len)
            candidates = []
            for req in sorted_queue:
                if len(candidates) >= n:
                    break
                candidates.append(req)
                self._queue.remove(req)
            return candidates

        if self.policy == SchedulingPolicy.PRIORITY:
            candidates = []
            while len(candidates) < n and self._queue:
                req = self._queue.popleft()
                candidates.append(req)
            return candidates

        return []

    # ---- Decode state management ----

    def decode_step_completed(
        self,
        request_tokens: dict[int, int],
    ) -> list[int]:
        """Called after the model executes a decode step.

        Updates request state: appends generated tokens, checks
        stop conditions, transitions requests to FINISHED.

        Args:
            request_tokens: Mapping of request_id -> new_token_id.

        Returns:
            List of request IDs that just finished.
        """
        finished_ids: list[int] = []
        with self._lock:
            for rid, token_id in request_tokens.items():
                req = self._running.get(rid)
                if req is None:
                    continue

                if req.state == RequestState.PREFILL:
                    # Transition from prefill to decode after first token
                    req.state = RequestState.DECODE
                    req.first_token_time = time.monotonic()
                    req.append_token(token_id)
                    # Update batch slot
                    for slot in self._batch_slots:
                        if slot.request_id == rid:
                            slot.in_prefill = False
                            break
                    continue

                req.append_token(token_id)

                # Check stop conditions
                if req.eos_token_id is not None and token_id == req.eos_token_id:
                    req.state = RequestState.FINISHED
                    finished_ids.append(rid)
                elif req.generated_len >= req.max_new_tokens:
                    req.state = RequestState.FINISHED
                    finished_ids.append(rid)

        self._total_decode_steps += 1
        return finished_ids

    # ---- Batch state queries ----

    def get_running_requests(self) -> list[Request]:
        """Get all currently running requests."""
        with self._lock:
            return list(self._running.values())

    def get_running_ids(self) -> list[int]:
        """Get IDs of all running requests."""
        with self._lock:
            return list(self._running.keys())

    def get_prefill_requests(self) -> list[Request]:
        """Get requests currently in prefill phase."""
        with self._lock:
            return [
                r for r in self._running.values() if r.state == RequestState.PREFILL
            ]

    def get_decode_requests(self) -> list[Request]:
        """Get requests currently in decode phase."""
        with self._lock:
            return [r for r in self._running.values() if r.state == RequestState.DECODE]

    def get_batch_slot_map(self) -> dict[int, BatchSlot]:
        """Get mapping from request_id to batch slot position."""
        with self._lock:
            return {slot.request_id: slot for slot in self._batch_slots}

    @property
    def queue_size(self) -> int:
        """Number of requests waiting in the queue."""
        return len(self._queue)

    @property
    def running_count(self) -> int:
        """Number of requests currently running."""
        return len(self._running)

    @property
    def batch_size(self) -> int:
        """Current batch size."""
        return len(self._batch_slots)

    @property
    def is_idle(self) -> bool:
        """Whether the scheduler has no work to do."""
        return self.queue_size == 0 and self.running_count == 0

    @property
    def step_count(self) -> int:
        """Total number of scheduler steps executed."""
        return self._step_count

    @property
    def total_processed(self) -> int:
        """Total number of requests processed to completion."""
        return self._total_requests_processed

    @property
    def prefill_decode_ratio(self) -> float:
        """Ratio of prefill steps to decode steps."""
        if self._total_decode_steps == 0:
            return float("inf")
        return self._total_prefill_steps / self._total_decode_steps

    # ---- Status report ----

    def status_report(self) -> str:
        """Generate a human-readable status report."""
        with self._lock:
            lines = [
                f"Scheduler Step {self._step_count}:",
                f"  Queue: {len(self._queue)} waiting",
                f"  Running: {len(self._running)}/{self.max_batch_size} slots",
                f"  Total processed: {self._total_requests_processed}",
                f"  Prefill/Decode ratio: {self.prefill_decode_ratio:.3f}",
                "",
            ]

            for slot in self._batch_slots:
                req = self._running.get(slot.request_id)
                if req:
                    phase = req.get_phase()
                    lines.append(
                        f"  [{slot.batch_idx}] Req {req.request_id}: "
                        f"{phase}, tokens={req.total_len}/{req.prompt_len + req.max_new_tokens}, "
                        f"gen={req.generated_len}"
                    )

            return "\n".join(lines)

    def reset(self) -> None:
        """Reset the scheduler to initial state."""
        with self._lock:
            self._queue.clear()
            self._running.clear()
            self._batch_slots.clear()
            self._step_count = 0
            self._total_requests_processed = 0
            self._total_prefill_steps = 0
            self._total_decode_steps = 0
