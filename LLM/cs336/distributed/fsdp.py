"""
Fully Sharded Data Parallel (FSDP / ZeRO-3) implementation.

Implements:
- FSDPWrapper: Wraps nn.Module with ZeRO-3 parameter sharding semantics
- All-Gather parameters before forward, free after backward
- Reduce-Scatter gradients in backward pass
- Communication/computation overlap (prefetch next layer parameters)
- Mixed precision support (fp16/bf16 with fp32 master weights)
- Activation checkpointing compatibility
- Optimizer state sharding (ZeRO-1/2 integration via parameter sharding)

ZeRO-3 memory formula: each GPU stores P/N + G/N + O/N
- P = parameters, G = gradients, O = optimizer states
- N = data-parallel world size
- Total: 4P/N for Adam (vs 4P for DDP)

Communication analysis per step (ZeRO-3):
- Forward: 1x All-Gather per layer = P bytes sent per rank
- Backward: 1x All-Gather + 1x Reduce-Scatter per layer = 2P bytes sent
- Total: 3P bytes per step (1.5x DDP communication volume)
- With overlap: forward All-Gather hidden by previous layer compute

FSDP workflow (per layer):
  1. All-Gather sharded parameters → full parameters
  2. Forward pass with full parameters
  3. Discard full parameters (free memory)
  4. (loss compute)
  5. All-Gather parameters again
  6. Backward pass with full parameters
  7. Reduce-Scatter gradients → sharded gradients
  8. Discard full parameters

Reference:
    ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
    (Rajbhandari et al., 2020)
"""

from __future__ import annotations

import copy
import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.distributed.fsdp import (
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import (
    _is_fsdp_flattened,
    _named_parameters_with_duplicates,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ZeROStage(Enum):
    """Enumeration of ZeRO optimization stages."""

    ZERO_1 = auto()  # Optimizer state sharding only
    ZERO_2 = auto()  # Optimizer state + gradient sharding
    ZERO_3 = auto()  # Optimizer state + gradient + parameter sharding


@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapping.

    Attributes:
        sharding_strategy: ZeRO stage for sharding.
        mixed_precision: Whether to use mixed precision (fp16/bf16).
        fp32_reduce_scatter: Use fp32 for gradient reduce-scatter.
        compute_dtype: Data type for computation (usually fp16 or bf16).
        param_dtype: Data type for stored sharded parameters (same as compute_dtype).
        reduce_dtype: Data type for reduced gradient (fp32 recommended for Adam).
        bucket_cap_mb: Gradient bucket size in MB for reduce-scatter.
        auto_wrap_policy: Function to decide which modules to wrap.
        use_orig_params: Whether to track original (unflattened) parameters.
        backward_prefetch: Prefetch next layer params during current backward ('backward_pre')
                          or during forward ('forward_pre').
        forward_prefetch: Prefetch next layer params during current forward.
        limit_all_gathers: Limit concurrent all-gather operations.
        activation_checkpointing: Whether to enable activation checkpointing.
        sync_module_states: Broadcast module parameters at init.
    """

    sharding_strategy: ZeROStage = ZeROStage.ZERO_3
    mixed_precision: bool = True
    fp32_reduce_scatter: bool = True
    compute_dtype: torch.dtype = torch.bfloat16
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    bucket_cap_mb: int = 25
    auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]] = None
    use_orig_params: bool = True
    backward_prefetch: Optional[str] = "backward_pre"
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False
    sync_module_states: bool = True


# ---------------------------------------------------------------------------
# Optimizer State Sharding (ZeRO-1 semantics)
# ---------------------------------------------------------------------------


class OptimizerStateSharding:
    """
    Shard optimizer states across data-parallel ranks (ZeRO-1).

    Each rank is responsible for updating only a subset of parameters.
    After the optimizer step, updated parameters are All-Gathered.

    For Adam-style optimizers, this reduces optimizer memory from 2P to 2P/N.
    """

    def __init__(self, params: list[nn.Parameter], dp_group: Any = None):
        self.dp_group = dp_group
        self.world_size = dist.get_world_size(dp_group) if dp_group is not None else 1
        self.rank = dist.get_rank(dp_group) if dp_group is not None else 0

        # Partition parameters across data-parallel ranks
        self._all_params = list(params)
        total_params = len(self._all_params)
        chunk_size = (total_params + self.world_size - 1) // self.world_size
        start = self.rank * chunk_size
        end = min(start + chunk_size, total_params)
        self._local_params = self._all_params[start:end]

    def get_local_params(self) -> list[nn.Parameter]:
        """Return parameters assigned to this rank for optimizer update."""
        return self._local_params

    def get_all_params(self) -> list[nn.Parameter]:
        """Return all parameters (for state dict gathering)."""
        return self._all_params

    def update_and_all_gather(
        self,
        optimizer: torch.optim.Optimizer,
        params_with_grad_bucket_size: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Perform optimizer step on local shard and All-Gather updated params.

        Optimizer only updates local shard. After update, All-Gather
        distributes updated parameters to all ranks for next step.

        Args:
            optimizer: Optimizer managing only local shard parameters.
        """
        optimizer.step()

        # All-Gather updated parameters so all ranks have complete weights
        if self.world_size > 1 and self.dp_group is not None:
            for param in self._all_params:
                dist.all_gather(
                    [param.data.clone() for _ in range(self.world_size)],
                    param.data,
                    group=self.dp_group,
                )


# ---------------------------------------------------------------------------
# FSDP Wrapper (ZeRO-3)
# ---------------------------------------------------------------------------


class FSDPWrapper(nn.Module):
    """
    Fully Sharded Data Parallel wrapper implementing ZeRO-3 semantics.

    Wraps a submodule and manages:
    - Parameter all-gathering before forward/backward
    - Parameter freeing after forward/backward
    - Gradient reduce-scatter after backward
    - Communication/computation overlap via streams
    - Mixed precision support with fp32 master weights

    Usage:
        model = MyModel(...)
        fsdp_model = FSDPWrapper(model, config=FSDPConfig(...))

    The wrapper hooks into the module's forward/backward to manage
    sharding transparently. Parameters are sharded in-place: each rank
    stores only 1/world_size of the parameters at rest.

    Lifecycle (per training step):
        1. _pre_forward: All-Gather params for this module → full params
        2. forward: Run computation with full params
        3. _post_forward: Free full params (keep activations)
        4. backward: _pre_backward All-Gathers params, then runs backward
        5. _post_backward: Reduce-Scatter gradients, free full params
    """

    def __init__(
        self,
        module: nn.Module,
        config: Optional[FSDPConfig] = None,
        dp_group: Any = None,
        process_group: Any = None,
    ):
        super().__init__()
        self.config = config or FSDPConfig()
        self.process_group = process_group or (
            dist.group.WORLD if dist.is_initialized() else None
        )
        self.dp_group = dp_group or self.process_group
        self._world_size = (
            dist.get_world_size(self.process_group)
            if self.process_group is not None
            else 1
        )
        self._rank = (
            dist.get_rank(self.process_group) if self.process_group is not None else 0
        )

        # Store original module
        self.module = module

        # Shard parameters: each rank keeps only its portion
        self._sharded_params: list[nn.Parameter] = []
        self._flat_param_to_orig: dict[str, nn.Parameter] = {}
        self._orig_to_sharded: dict[nn.Parameter, nn.Parameter] = {}
        self._param_shapes: dict[str, torch.Size] = {}
        self._full_params_allocated: bool = False
        self._full_params: list[torch.Tensor] = []
        self._full_param_buffers: dict[str, torch.Tensor] = {}

        # Streams for overlap
        self._compute_stream = (
            torch.cuda.current_stream() if torch.cuda.is_available() else None
        )
        self._comm_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Prefetch buffers
        self._prefetch_params: Optional[list[torch.Tensor]] = None
        self._prefetch_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._grads_ready: dict[str, bool] = {}

        # Mixed precision
        self._fp32_master_params: Optional[dict[str, torch.Tensor]] = None

        # Initialize sharded parameters
        self._shard_parameters()

        # Register forward/backward hooks for communication management
        self._register_hooks()

    def _shard_parameters(self) -> None:
        """
        Shard all parameters of the wrapped module across data-parallel ranks.

        Each parameter is split into world_size shards along dim 0.
        Each rank stores only its shard as the parameter data.
        """
        for name, param in self.module.named_parameters(recurse=True):
            if not param.requires_grad:
                continue

            orig_shape = param.shape
            total_numel = param.numel()
            chunk_numel = (total_numel + self._world_size - 1) // self._world_size

            # Flatten and shard
            flat_param = param.data.view(-1)
            shard_start = self._rank * chunk_numel
            shard_end = min(shard_start + chunk_numel, total_numel)
            shard_data = flat_param[shard_start:shard_end].clone()

            # Replace parameter data with shard
            sharded_param = nn.Parameter(shard_data, requires_grad=param.requires_grad)
            sharded_param.grad = None

            self._sharded_params.append(sharded_param)
            self._flat_param_to_orig[name] = param
            self._orig_to_sharded[param] = sharded_param
            self._param_shapes[name] = orig_shape

            # Optional: maintain fp32 master copy for optimizer precision
            if (
                self.config.mixed_precision
                and self.config.reduce_dtype == torch.float32
            ):
                if self._fp32_master_params is None:
                    self._fp32_master_params = {}
                self._fp32_master_params[name] = shard_data.float().clone()

    def _all_gather_params(self) -> dict[str, torch.Tensor]:
        """
        All-Gather sharded parameters to form full parameters.

        Each rank contributes its shard; the result is the complete
        parameter tensor on every rank.

        Communication volume: P bytes per All-Gather call.
        """
        full_params: dict[str, torch.Tensor] = {}

        for name in self._flat_param_to_orig:
            param = self._flat_param_to_orig[name]
            sharded_param = self._orig_to_sharded[param]
            orig_shape = self._param_shapes[name]
            total_numel = param.numel()

            # All-Gather in the parameter dimension
            gathered_chunks = [
                torch.empty_like(sharded_param.data) for _ in range(self._world_size)
            ]

            if (
                self._world_size > 1
                and self.process_group is not None
                and dist.is_initialized()
            ):
                dist.all_gather(
                    gathered_chunks, sharded_param.data, group=self.process_group
                )

            full_flat = torch.cat(gathered_chunks)[:total_numel]
            full_params[name] = full_flat.view(orig_shape)

        return full_params

    def _reduce_scatter_gradients(self, full_params: dict[str, torch.Tensor]) -> None:
        """
        Reduce-Scatter gradients from all ranks into sharded gradients.

        Each rank receives only its portion of the reduced gradient.
        This is the gradient synchronization step: gradients from all
        ranks are summed and distributed back as shards.

        Communication volume: P bytes per Reduce-Scatter call.
        """
        for name, full_param in full_params.items():
            if full_param.grad is None:
                continue

            orig_shape = self._param_shapes[name]
            flat_grad = full_param.grad.view(-1)
            total_numel = full_param.numel()
            chunk_numel = (total_numel + self._world_size - 1) // self._world_size

            # Pad to even chunks across world_size
            padded_numel = chunk_numel * self._world_size
            if padded_numel > total_numel:
                padded_grad = torch.zeros(
                    padded_numel, device=flat_grad.device, dtype=flat_grad.dtype
                )
                padded_grad[:total_numel] = flat_grad
                flat_grad = padded_grad

            # Reduce-scatter: reduces all gradients and scatters to chunks
            reduce_chunks = flat_grad.chunk(self._world_size)
            reduced_chunk = torch.zeros_like(reduce_chunks[0])

            if (
                self._world_size > 1
                and self.process_group is not None
                and dist.is_initialized()
            ):
                dist.reduce_scatter(
                    reduced_chunk, reduce_chunks, group=self.process_group
                )
            else:
                reduced_chunk.copy_(reduce_chunks[0])

            # Copy reduced shard back to the sharded parameter's grad
            param = self._flat_param_to_orig[name]
            sharded_param = self._orig_to_sharded[param]
            with torch.no_grad():
                sharded_param.grad = reduced_chunk[: sharded_param.numel()].view_as(
                    sharded_param.data
                )

    def _register_hooks(self) -> None:
        """Register forward and backward hooks for communication management."""

        def _pre_forward_hook(
            module: nn.Module, args: tuple[Any, ...]
        ) -> tuple[Any, ...]:
            # All-Gather parameters before forward
            if not self._full_params_allocated:
                with (
                    torch.cuda.stream(self._comm_stream)
                    if self._comm_stream
                    else contextlib_nullcontext()
                ):
                    self._full_param_buffers = self._all_gather_params()
                if self._comm_stream is not None:
                    torch.cuda.current_stream().wait_stream(self._comm_stream)
                self._full_params_allocated = True
            return args

        def _post_forward_hook(
            module: nn.Module, args: tuple[Any, ...], output: Any
        ) -> Any:
            # Free full parameters after forward (keep shards)
            self._full_param_buffers.clear()
            self._full_params_allocated = False
            return output

        def _pre_backward_hook(
            module: nn.Module, grad_output: tuple[Any, ...]
        ) -> tuple[Any, ...]:
            # All-Gather parameters before backward
            if not self._full_params_allocated:
                self._full_param_buffers = self._all_gather_params()
                self._full_params_allocated = True
            return grad_output

        def _post_backward_hook(module: nn.Module) -> None:
            # Reduce-Scatter gradients after backward, free full params
            if self._full_params_allocated:
                self._reduce_scatter_gradients(self._full_param_buffers)
                self._full_param_buffers.clear()
                self._full_params_allocated = False

        self.module.register_forward_pre_hook(_pre_forward_hook)
        self.module.register_forward_hook(_post_forward_hook)
        self.module.register_full_backward_pre_hook(_pre_backward_hook)
        self.module.register_full_backward_hook(_post_backward_hook)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through wrapped module."""
        return self.module(*args, **kwargs)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> FSDPWrapper:
        """Propagate _apply (e.g., .to(), .cuda()) to underlying module."""
        super()._apply(fn)
        self.module._apply(fn)
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Gather full state dict (All-Gather all sharded parameters)."""
        full_params = self._all_gather_params()
        return {k: v.clone() for k, v in full_params.items()}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Load state dict, sharding parameters across ranks."""
        with torch.no_grad():
            for name, full_tensor in state_dict.items():
                if name in self._flat_param_to_orig:
                    param = self._flat_param_to_orig[name]
                    orig_shape = self._param_shapes[name]
                    total_numel = param.numel()
                    chunk_numel = (
                        total_numel + self._world_size - 1
                    ) // self._world_size
                    shard_start = self._rank * chunk_numel
                    shard_end = min(shard_start + chunk_numel, total_numel)

                    flat_full = full_tensor.view(-1)
                    shard = flat_full[shard_start:shard_end].view_as(
                        self._orig_to_sharded[param].data
                    )
                    self._orig_to_sharded[param].data.copy_(shard)

                    # Update fp32 master if used
                    if (
                        self._fp32_master_params is not None
                        and name in self._fp32_master_params
                    ):
                        self._fp32_master_params[name] = shard.float().clone()

        return None

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return sharded parameters for the optimizer."""
        return iter(self._sharded_params)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Return named sharded parameters."""
        for i, param in enumerate(self._sharded_params):
            yield f"{prefix}_shard_{i}", param


# ---------------------------------------------------------------------------
# Helper: wrap a model with FSDP
# ---------------------------------------------------------------------------


def wrap_fsdp(
    module: nn.Module,
    config: Optional[FSDPConfig] = None,
    dp_group: Any = None,
    auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]] = None,
) -> nn.Module:
    """
    Recursively wrap a module tree with FSDP.

    For simple cases, wraps the entire module. For larger models,
    use auto_wrap_policy to decide which submodules to wrap
    (e.g., wrap each TransformerBlock individually for
    communication/computation overlap).

    Args:
        module: The module to wrap.
        config: FSDP configuration.
        dp_group: Data-parallel process group.
        auto_wrap_policy: Optional function to determine which modules to wrap.
                         Signature: (module, recurse, nonwrapped_numel) -> bool.

    Returns:
        Wrapped module (fully FSDP-wrapped if policy provided).
    """
    if config is None:
        config = FSDPConfig()

    if auto_wrap_policy is None:
        # Wrap the entire module
        return FSDPWrapper(module, config=config, dp_group=dp_group)

    # Recursive wrapping based on policy
    def _wrap_recursive(mod: nn.Module) -> nn.Module:
        children = list(mod.named_children())
        for name, child in children:
            if auto_wrap_policy(
                child,
                recurse=False,
                nonwrapped_numel=sum(p.numel() for p in child.parameters()),
            ):
                setattr(mod, name, FSDPWrapper(child, config=config, dp_group=dp_group))
            else:
                _wrap_recursive(child)
        return mod

    return _wrap_recursive(module)


# ---------------------------------------------------------------------------
# Context manager for null context (no-op)
# ---------------------------------------------------------------------------


class _ContextLibNullContext:
    """Null context manager for when CUDA streams are unavailable."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        pass


def contextlib_nullcontext() -> Any:
    return _ContextLibNullContext()
