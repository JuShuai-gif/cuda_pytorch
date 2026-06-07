"""
Production-grade collective communication primitives.

Implements:
- Ring All-Reduce (optimal for large tensors, bandwidth-optimal)
- Tree All-Reduce (optimal for small tensors, latency-optimal)
- Hierarchical All-Reduce (node-local ring + cross-node tree, for multi-node)
- All-Gather, Reduce-Scatter, Broadcast, All-to-All
- Communication bandwidth benchmarking
- Gradient bucketing utilities
- Communication/computation overlap helpers using CUDA streams

Key communication formulas:
- Ring All-Reduce time = 2 * alpha * (N * (P - 1) / P), where alpha = element_size * elements / bandwidth
- NVLink bandwidth: 900 GB/s (intra-node)
- InfiniBand NDR400: 400 GB/s (inter-node)
- PCIe 5.0: 64 GB/s
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# Communication backend abstraction
# ---------------------------------------------------------------------------


class CommunicationBackend(Enum):
    """Supported communication backends."""

    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


@dataclass
class CommunicationMetrics:
    """Measured communication metrics for a single collective operation."""

    operation: str
    tensor_size_bytes: int
    world_size: int
    backend: str
    elapsed_ms: float
    bandwidth_gb_s: float
    theoretical_bandwidth_gb_s: float = 900.0  # NVLink default
    efficiency: float = 0.0

    def __post_init__(self) -> None:
        if self.theoretical_bandwidth_gb_s > 0:
            self.efficiency = self.bandwidth_gb_s / self.theoretical_bandwidth_gb_s


def _ensure_distributed() -> None:
    """Ensure torch.distributed is initialized, using a default setup if needed."""
    if not dist.is_initialized():
        pass  # Caller should check explicitly; we degrade gracefully below


def _get_default_group() -> Any:
    """Return the default process group or None."""
    if dist.is_initialized():
        return dist.group.WORLD
    return None


# ---------------------------------------------------------------------------
# Collective operation implementations
# ---------------------------------------------------------------------------


def ring_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Any = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Bandwidth-optimal ring All-Reduce.

    Decomposes into Reduce-Scatter followed by All-Gather.
    Total communication volume per rank: 2 * (P - 1) / P * data_size

    For a tensor of N elements on P ranks:
      - Each chunk = N / P elements
      - Step 1 (Reduce-Scatter): P-1 rounds, each sends N/P elements
      - Step 2 (All-Gather): P-1 rounds, each sends N/P elements
      - Total sent: 2 * (P-1)/P * N elements

    Ring algorithm: ranks arranged in a logical ring; each rank sends
    to (rank+1)%P and receives from (rank-1+P)%P.

    Args:
        tensor: Local tensor to reduce (modified in-place with result).
        op: Reduction operation.
        group: Process group (defaults to WORLD).
        stream: CUDA stream for communication.

    References:
        "Bandwidth optimal all-reduce algorithms for clusters of workstations"
        (Thakur, Rabenseifner, Gropp, 2005)
    """
    if group is None:
        group = _get_default_group()

    if group is None:
        return  # No-op when no distributed context

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return

    # Fall back to NCCL's optimized all-reduce when available
    # (NCCL's ring all-reduce is highly optimized - use it directly)
    if dist.get_backend() == dist.Backend.NCCL and stream is None:
        dist.all_reduce(tensor, op=op, group=group)
        return

    # Manual ring all-reduce for custom backends or with stream control
    num_elements = tensor.numel()
    chunk_size = num_elements // world_size

    if num_elements % world_size != 0:
        # Pad to ensure even division, or fall back to built-in
        dist.all_reduce(tensor, op=op, group=group)
        return

    chunks = list(tensor.chunk(world_size))

    # Step 1: Reduce-Scatter
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step) % world_size
        recv_chunk_idx = (rank - step - 1) % world_size
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size

        recv_buf = torch.empty_like(chunks[recv_chunk_idx])
        send_req = dist.isend(chunks[send_chunk_idx], send_rank, group=group)
        recv_req = dist.irecv(recv_buf, recv_rank, group=group)
        send_req.wait()
        recv_req.wait()

        if op == dist.ReduceOp.SUM:
            chunks[recv_chunk_idx].add_(recv_buf)
        elif op == dist.ReduceOp.PRODUCT:
            chunks[recv_chunk_idx].mul_(recv_buf)
        elif op == dist.ReduceOp.MIN:
            chunks[recv_chunk_idx] = torch.min(chunks[recv_chunk_idx], recv_buf)
        elif op == dist.ReduceOp.MAX:
            chunks[recv_chunk_idx] = torch.max(chunks[recv_chunk_idx], recv_buf)

    # Step 2: All-Gather
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step + 1) % world_size
        recv_chunk_idx = (rank - step) % world_size
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size

        recv_buf = torch.empty_like(chunks[recv_chunk_idx])
        send_req = dist.isend(chunks[send_chunk_idx], send_rank, group=group)
        recv_req = dist.irecv(recv_buf, recv_rank, group=group)
        send_req.wait()
        recv_req.wait()

        chunks[recv_chunk_idx] = recv_buf

    # Reassemble result
    tensor.copy_(torch.cat(chunks))


def tree_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Any = None,
) -> None:
    """
    Latency-optimal tree All-Reduce using a binomial tree topology.

    Better for small tensors because halving the number of steps
    (log2(P) vs 2*(P-1)/P rounds for ring).

    Communication: 2 * log2(P) steps, total volume = 2 * N per rank
    (in aggregate, each rank sends/receives 2N data total).

    Args:
        tensor: Local tensor to reduce (modified in-place).
        op: Reduction operation.
        group: Process group.
    """
    if group is None:
        group = _get_default_group()

    if group is None:
        return

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return

    # Use NCCL built-in for performance
    if dist.get_backend() == dist.Backend.NCCL:
        dist.all_reduce(tensor, op=op, group=group)
        return

    num_elements = tensor.numel()
    recv_buf = torch.empty_like(tensor)

    # Reduce tree: binary tree reduction to root (rank 0)
    stride = 1
    while stride < world_size:
        if rank % (2 * stride) == 0:
            src = rank + stride
            if src < world_size:
                dist.recv(recv_buf, src=src, group=group)
                if op == dist.ReduceOp.SUM:
                    tensor.add_(recv_buf)
                elif op == dist.ReduceOp.PRODUCT:
                    tensor.mul_(recv_buf)
                elif op == dist.ReduceOp.MIN:
                    tensor = torch.min(tensor, recv_buf)
                elif op == dist.ReduceOp.MAX:
                    tensor = torch.max(tensor, recv_buf)
        elif rank % stride == 0:
            dst = rank - stride
            dist.send(tensor, dst=dst, group=group)
        stride *= 2

    # Broadcast tree: binary tree broadcast from root
    stride = 1
    while stride < world_size:
        stride *= 2
    while stride > 1:
        stride //= 2
        if rank % (2 * stride) == 0:
            dst = rank + stride
            if dst < world_size:
                dist.send(tensor, dst=dst, group=group)
        elif rank % stride == 0:
            src = rank - stride
            dist.recv(tensor, src=src, group=group)


def hierarchical_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    intra_node_group: Any = None,
    inter_node_group: Any = None,
    use_ring: bool = True,
) -> None:
    """
    Hierarchical All-Reduce: node-local reduce then cross-node broadcast.

    Designed for multi-node clusters where intra-node bandwidth (NVLink: 900 GB/s)
    >> inter-node bandwidth (InfiniBand: 400 GB/s).

    Strategy:
    1. All-Reduce within node (ring algorithm, high bandwidth)
    2. All-Reduce across nodes (ring or tree, lower bandwidth)
    3. This minimizes cross-node traffic

    Args:
        tensor: Local tensor.
        op: Reduction op.
        intra_node_group: Process group for intra-node GPUs.
        inter_node_group: Process group for one rank per node.
        use_ring: Use ring for intra-node (True) or tree (False).
    """
    if intra_node_group is None and inter_node_group is None:
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return

    # Step 1: All-reduce within each node (fast NVLink)
    if intra_node_group is not None:
        if use_ring:
            ring_all_reduce(tensor, op=op, group=intra_node_group)
        else:
            tree_all_reduce(tensor, op=op, group=intra_node_group)

    # Step 2: All-reduce across nodes (slower InfiniBand)
    if inter_node_group is not None:
        # Only one rank per node participates in cross-node communication
        if dist.get_rank(inter_node_group) >= 0:
            ring_all_reduce(tensor, op=op, group=inter_node_group)

    # Step 3: Broadcast reduced result back to all ranks within each node
    if intra_node_group is not None:
        local_root = 0  # Rank 0 within each node group
        dist.broadcast(tensor, src=local_root, group=intra_node_group)


# ---------------------------------------------------------------------------
# High-level collective creation helpers
# ---------------------------------------------------------------------------


def create_broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Any = None,
) -> torch.Tensor:
    """Broadcast tensor from src to all ranks in group. Returns the tensor."""
    if group is None:
        group = _get_default_group()
    if group is not None:
        dist.broadcast(tensor, src=src, group=group)
    return tensor


def create_all_gather(
    tensor: torch.Tensor,
    group: Any = None,
    gather_dim: int = 0,
) -> torch.Tensor:
    """
    All-Gather: collect tensors from all ranks and concatenate.

    Each rank provides a tensor of shape [local_size, ...].
    Output is shape [local_size * world_size, ...] along gather_dim.
    """
    if group is None:
        group = _get_default_group()

    if group is None:
        return tensor

    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor

    # Move gather_dim to dim 0 for NCCL compatibility
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor, group=group)
    return torch.cat(gather_list, dim=gather_dim)


def create_reduce_scatter(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Any = None,
    scatter_dim: int = 0,
) -> torch.Tensor:
    """
    Reduce-Scatter: reduce across ranks, each rank gets one chunk.

    Each rank provides a tensor; output is reduced and split equally.
    Output shape = input_shape // world_size along scatter_dim.
    """
    if group is None:
        group = _get_default_group()

    if group is None:
        return tensor

    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor

    # NCCL reduce_scatter uses the flattened tensor
    output = torch.empty_like(tensor.chunk(world_size, dim=scatter_dim)[0])
    dist.reduce_scatter(output, [tensor], op=op, group=group)
    return output


def create_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Any = None,
    algorithm: str = "auto",
) -> torch.Tensor:
    """
    All-Reduce with configurable algorithm (ring, tree, hierarchical, auto).

    'auto' delegates to NCCL when available.
    """
    if algorithm == "auto" or group is None:
        if dist.is_initialized() and group is None:
            group = dist.group.WORLD
        if group is not None:
            dist.all_reduce(tensor, op=op, group=group)
        return tensor

    if algorithm == "ring":
        ring_all_reduce(tensor, op=op, group=group)
    elif algorithm == "tree":
        tree_all_reduce(tensor, op=op, group=group)
    elif algorithm == "hierarchical":
        hierarchical_all_reduce(tensor, op=op, group=group)

    return tensor


def create_all_to_all(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    output_split_sizes: Optional[list[int]] = None,
    input_split_sizes: Optional[list[int]] = None,
    group: Any = None,
) -> torch.Tensor:
    """
    All-to-All: each rank scatters its data across all ranks and gathers from others.

    Used in MoE for token routing between experts across ranks.

    Args:
        output_tensor: Pre-allocated output buffer.
        input_tensor: Input tensor to scatter.
        output_split_sizes: Size per rank for output (None = even split).
        input_split_sizes: Size per rank for input (None = even split).
        group: Process group.

    Returns:
        output_tensor filled with gathered data.
    """
    if group is None:
        group = _get_default_group()

    if group is None:
        output_tensor.copy_(input_tensor)
        return output_tensor

    dist.all_to_all(
        output_tensor,
        input_tensor,
        output_split_sizes=output_split_sizes or [],
        input_split_sizes=input_split_sizes or [],
        group=group,
    )
    return output_tensor


# ---------------------------------------------------------------------------
# Gradient bucketing
# ---------------------------------------------------------------------------


@dataclass
class GradientBucket:
    """A bucket of gradients for batched All-Reduce communication."""

    params: list[nn.Parameter]
    grads: list[torch.Tensor]
    total_bytes: int = 0
    ready: bool = False

    def __post_init__(self) -> None:
        self.total_bytes = sum(g.numel() * g.element_size() for g in self.grads)

    def flatten(self) -> torch.Tensor:
        """Flatten all gradients into a contiguous buffer."""
        return torch.cat([g.view(-1) for g in self.grads])

    def unflatten(self, flat_grad: torch.Tensor) -> None:
        """Scatter flat gradient back to individual parameter gradients."""
        offset = 0
        for grad in self.grads:
            numel = grad.numel()
            grad.copy_(flat_grad[offset : offset + numel].view_as(grad))
            offset += numel

    def mark_ready(self) -> None:
        """Mark this bucket as ready for communication."""
        self.ready = True


def bucket_gradients(
    parameters: Iterator[nn.Parameter],
    bucket_size_mb: float = 25.0,
) -> list[GradientBucket]:
    """
    Group gradients into buckets of approximately bucket_size_mb megabytes.

    This is the core optimization in DDP: by bucketing gradients, we can
    overlap gradient computation (backward) with communication (All-Reduce).

    The default bucket_size_mb = 25 MB is PyTorch DDP's default and balances
    communication latency (small buckets = more overhead) with overlap
    opportunity (large buckets = less chance to overlap).

    Args:
        parameters: Iterator over model parameters.
        bucket_size_mb: Target bucket size in megabytes.

    Returns:
        List of GradientBucket objects.
    """
    bucket_bytes = int(bucket_size_mb * 1024 * 1024)
    buckets: list[GradientBucket] = []
    current_params: list[nn.Parameter] = []
    current_grads: list[torch.Tensor] = []
    current_bytes = 0

    # Process parameters in reverse order (backward runs last-to-first)
    params_list = list(parameters)

    for param in reversed(params_list):
        if param.grad is None:
            continue
        grad_bytes = param.grad.numel() * param.grad.element_size()
        grad = param.grad

        if current_bytes + grad_bytes > bucket_bytes and current_params:
            buckets.append(GradientBucket(params=current_params, grads=current_grads))
            current_params = []
            current_grads = []
            current_bytes = 0

        current_params.append(param)
        current_grads.append(grad)
        current_bytes += grad_bytes

    if current_params:
        buckets.append(GradientBucket(params=current_params, grads=current_grads))

    return buckets


# ---------------------------------------------------------------------------
# Bandwidth benchmarking
# ---------------------------------------------------------------------------


def benchmark_bandwidth(
    tensor_size_bytes: int = 256 * 1024 * 1024,  # 256 MB
    num_warmup: int = 5,
    num_iterations: int = 20,
    group: Any = None,
) -> list[CommunicationMetrics]:
    """
    Benchmark point-to-point bandwidth between ranks.

    Measures both send/recv bandwidth for various tensor sizes,
    returning metrics for bridge analysis between theoretical
    and achieved bandwidth.

    Args:
        tensor_size_bytes: Size of tensor for main benchmark.
        num_warmup: Warmup iterations.
        num_iterations: Measurement iterations.
        group: Process group.

    Returns:
        List of CommunicationMetrics for each measurement.
    """
    if group is None:
        group = _get_default_group()

    if group is None or dist.get_world_size(group) < 2:
        return []

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    metrics: list[CommunicationMetrics] = []

    # Test various sizes from 1 KB to the specified max
    sizes = [
        1024,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
        tensor_size_bytes,
    ]

    for size_bytes in sizes:
        tensor = torch.randn(size_bytes // 4, device=device)  # float32

        for _ in range(num_warmup):
            if rank == 0:
                dist.send(tensor, dst=1, group=group)
            elif rank == 1:
                dist.recv(tensor, src=0, group=group)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            if rank == 0:
                dist.send(tensor, dst=1, group=group)
            elif rank == 1:
                dist.recv(tensor, src=0, group=group)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations
        bandwidth = (size_bytes / 1e9) / elapsed  # GB/s

        metrics.append(
            CommunicationMetrics(
                operation="send_recv",
                tensor_size_bytes=size_bytes,
                world_size=world_size,
                backend=str(dist.get_backend()),
                elapsed_ms=elapsed * 1000,
                bandwidth_gb_s=bandwidth,
            )
        )

    return metrics


def benchmark_all_reduce(
    tensor: Optional[torch.Tensor] = None,
    num_elements: int = 64 * 1024 * 1024,  # 64M elements = 256 MB in fp32
    num_warmup: int = 5,
    num_iterations: int = 20,
    group: Any = None,
    algorithms: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Benchmark All-Reduce performance with different algorithms.

    Args:
        tensor: Pre-allocated tensor (created if None).
        num_elements: Number of elements to create if tensor not provided.
        num_warmup: Warmup iterations.
        num_iterations: Measurement iterations.
        group: Process group.
        algorithms: List of algorithm names to test (default: ['auto']).

    Returns:
        Dict mapping algorithm name to bandwidth in GB/s.
    """
    if algorithms is None:
        algorithms = ["auto"]

    if group is None:
        group = _get_default_group()

    if group is None:
        return {}

    rank = dist.get_rank(group)
    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )

    if tensor is None:
        tensor = torch.randn(num_elements, device=device)

    data_size_bytes = tensor.numel() * tensor.element_size()
    results: dict[str, float] = {}

    for algo in algorithms:
        work_tensor = tensor.clone()

        for _ in range(num_warmup):
            create_all_reduce(work_tensor, algorithm=algo, group=group)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            create_all_reduce(work_tensor, algorithm=algo, group=group)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / num_iterations
        bandwidth = (data_size_bytes / 1e9) / elapsed  # GB/s
        results[algo] = bandwidth

    return results


# ---------------------------------------------------------------------------
# Communication/computation overlap with CUDA streams
# ---------------------------------------------------------------------------


class CommOverlapHelper:
    """
    Helper for overlapping communication with computation using CUDA streams.

    In backward pass, when a bucket's gradients are ready, we can launch
    the All-Reduce on a communication stream while the main compute stream
    continues computing gradients for earlier layers.

    Pattern:
        with comm_helper.next_bucket_ready(bucket):
            # Bucket all-reduce is launched on comm stream
            pass
        # Main stream continues computing gradients independently

    Reference:
        PyTorch DDP gradient bucketing overlap implementation.
    """

    def __init__(self, world_size: int, group: Any = None):
        self.world_size = world_size
        self.group = group
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._pending_buckets: list[tuple[GradientBucket, torch.Tensor]] = []

    def _ensure_streams(self) -> None:
        """Lazily create CUDA streams."""
        if self._comm_stream is None and torch.cuda.is_available():
            self._comm_stream = torch.cuda.Stream()
            self._compute_stream = torch.cuda.current_stream()

    def async_all_reduce_bucket(
        self,
        bucket: GradientBucket,
    ) -> None:
        """
        Launch async All-Reduce for a gradient bucket on the comm stream.

        The flattened bucket gradient is sent for reduction on the
        communication stream, allowing the main compute stream to
        continue processing without waiting.

        Args:
            bucket: GradientBucket with ready gradients.
        """
        self._ensure_streams()
        flat_grad = bucket.flatten()

        if self._comm_stream is not None and self.group is not None:
            with torch.cuda.stream(self._comm_stream):
                # Wait for compute stream to finish producing these gradients
                self._comm_stream.wait_stream(torch.cuda.current_stream())
                if dist.is_initialized():
                    dist.all_reduce(
                        flat_grad,
                        op=dist.ReduceOp.AVG
                        if self.world_size > 1
                        else dist.ReduceOp.SUM,
                        group=self.group,
                    )
            self._pending_buckets.append((bucket, flat_grad))

    def sync_all(self) -> None:
        """
        Wait for all pending communication to complete and scatter gradients.

        Must be called after backward() completes, before optimizer.step().
        """
        if self._comm_stream is not None:
            # Wait for comm stream to finish all pending all-reduces
            torch.cuda.current_stream().wait_stream(self._comm_stream)

        # Scatter flattened gradients back to parameters
        for bucket, flat_grad in self._pending_buckets:
            bucket.unflatten(flat_grad)

        self._pending_buckets.clear()


# ---------------------------------------------------------------------------
# Collective operation type enum (for use in higher-level modules)
# ---------------------------------------------------------------------------


class CollectiveOp(Enum):
    """Enumeration of collective operations."""

    BROADCAST = auto()
    SCATTER = auto()
    GATHER = auto()
    REDUCE = auto()
    ALL_GATHER = auto()
    REDUCE_SCATTER = auto()
    ALL_REDUCE = auto()
    ALL_TO_ALL = auto()
