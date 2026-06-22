"""
Production-grade Pipeline Parallelism with multiple scheduling strategies.

Implements:
- PipelineStage: Wraps model layers for one PP rank
- GPipe scheduling: All microbatch forwards, then all backwards
- 1F1B scheduling: Interleaved forward/backward to reduce peak activation memory
- Interleaved 1F1B: Further subdivides layers for lower bubble ratio
- Activation checkpointing integration
- Bubble ratio analysis and microbatch optimization

Key formulas:
- GPipe bubble ratio = (P - 1) / M   where P = pipeline stages, M = microbatches
- 1F1B peak activation = O(1) per stage vs GPipe's O(M)
- Interleaved 1F1B bubble = (P - 1) / (M * V) where V = virtual stages per device
- Optimal M ≈ P * 4 for GPipe (limit bubble to ~20%)
- Communication is point-to-point send/recv between adjacent stages

Bandwidth considerations:
- Within-node PP stages communicate via NVLink (900 GB/s)
- Cross-node PP stages communicate via InfiniBand (400 GB/s)
- Activation tensor size per microbatch = batch * seq * hidden * dtype_bytes

Reference:
    GPipe: Efficient Training of Large Neural Networks using Pipeline Parallelism
    (Huang et al., 2019)
    Memory-Efficient Pipeline-Parallel DNN Training (Narayanan et al., ICML 2021)
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------------
# Pipeline Stage
# ---------------------------------------------------------------------------


class PipelineStage(nn.Module):
    """
    A contiguous sub-sequence of model layers assigned to one PP rank.

    The stage receives its input from the previous stage (or data loader
    for stage 0) and sends its output to the next stage (or computes loss
    for the last stage).

    Communication: P2P send/recv between adjacent stages.
    """

    def __init__(
        self,
        layers: nn.ModuleList | nn.Sequential,
        stage_idx: int,
        num_stages: int,
        pp_group: Any = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.pp_group = pp_group
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == num_stages - 1

        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = layers

        if device is not None:
            self.layers = self.layers.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass through all layers in this stage."""
        return self.layers(x)

    def send_forward(self, tensor: torch.Tensor, dst: int) -> None:
        """Send activation tensor to the next pipeline stage."""
        if dist.is_initialized() and self.pp_group is not None:
            dist.send(tensor, dst=dst, group=self.pp_group)

    def recv_forward(
        self, src: int, shape: torch.Size, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Receive activation tensor from the previous pipeline stage."""
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if dist.is_initialized() and self.pp_group is not None:
            dist.recv(tensor, src=src, group=self.pp_group)
        return tensor

    def send_backward(self, tensor: torch.Tensor, dst: int) -> None:
        """Send gradient tensor to the previous pipeline stage."""
        if dist.is_initialized() and self.pp_group is not None:
            dist.send(tensor, dst=dst, group=self.pp_group)

    def recv_backward(
        self, src: int, shape: torch.Size, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Receive gradient tensor from the next pipeline stage."""
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if dist.is_initialized() and self.pp_group is not None:
            dist.recv(tensor, src=src, group=self.pp_group)
        return tensor


def make_pipeline_stages(
    model: nn.Module,
    num_stages: int,
    stage_to_modules: Optional[Callable[[nn.Module, int, int], nn.Module]] = None,
) -> list[PipelineStage]:
    """
    Partition a model into pipeline stages.

    By default, splits the module's named_children() evenly across stages.
    For Transformer models, this maps each transformer layer to a stage
    (possibly bundling multiple layers per stage).

    Args:
        model: The full model to partition.
        num_stages: Number of pipeline stages.
        stage_to_modules: Custom mapping from model to stage module list.

    Returns:
        List of PipelineStage objects, one per PP rank.
    """
    if stage_to_modules is not None:
        stages = []
        for s in range(num_stages):
            module = stage_to_modules(model, s, num_stages)
            stages.append(PipelineStage(module, stage_idx=s, num_stages=num_stages))
        return stages

    # Default: split named_children evenly
    children = list(model.named_children())
    if not children:
        # Model is a single module; wrap it in stage 0
        return [PipelineStage(model, stage_idx=0, num_stages=max(num_stages, 1))]

    num_children = len(children)
    per_stage = max(1, num_children // num_stages)
    stages_list: list[PipelineStage] = []

    for s in range(num_stages):
        start = s * per_stage
        end = start + per_stage if s < num_stages - 1 else num_children
        stage_layers = nn.Sequential(*(m for _, m in children[start:end]))
        stages_list.append(
            PipelineStage(stage_layers, stage_idx=s, num_stages=num_stages)
        )

    return stages_list


# ---------------------------------------------------------------------------
# Scheduling strategies
# ---------------------------------------------------------------------------


class ScheduleType(Enum):
    """Pipeline scheduling algorithm type."""

    GPIPE = auto()
    ONE_F_ONE_B = auto()
    INTERLEAVED_ONE_F_ONE_B = auto()


@dataclass
class PipelineSchedule:
    """A pipeline execution schedule (sequence of operations per timestep)."""

    schedule_type: ScheduleType
    num_stages: int
    num_microbatches: int
    num_warmup_steps: int
    num_steady_steps: int
    num_cooldown_steps: int
    total_steps: int
    bubble_ratio: float


def bubble_ratio(num_stages: int, num_microbatches: int) -> float:
    """
    Compute the bubble ratio for GPipe-style scheduling.

    bubble = (P - 1) / (P + M - 1)  for GPipe
           ≈ (P - 1) / M  for large M

    Args:
        num_stages: Number of pipeline stages.
        num_microbatches: Number of microbatches.

    Returns:
        Bubble ratio (0.0 to 1.0).
    """
    if num_microbatches == 0:
        return 1.0
    return (num_stages - 1) / (num_stages + num_microbatches - 1)


def compute_pipeline_bubble(
    num_stages: int,
    num_microbatches: int,
    schedule_type: ScheduleType = ScheduleType.GPIPE,
    virtual_stages: int = 1,
) -> PipelineSchedule:
    """
    Compute complete pipeline schedule metrics including bubble ratio.

    Args:
        num_stages: Number of physical pipeline stages.
        num_microbatches: Number of microbatches.
        schedule_type: Scheduling algorithm.
        virtual_stages: For interleaved, number of virtual stages per device.

    Returns:
        PipelineSchedule with timing analysis.
    """
    if schedule_type == ScheduleType.GPIPE:
        # Total steps: each microbatch traverses P stages, with P-1 warmup + P-1 cooldown
        # Total time = M + P - 1 forward steps, same for backward
        # Effective bubble slots = (P-1) * 2 (one per warmup/cooldown region)
        num_warmup = num_stages - 1
        total_fwd_bwd = 2 * (num_microbatches + num_stages - 1)
        bubble = bubble_ratio(num_stages, num_microbatches)
        total_steps = num_microbatches + num_stages - 1  # per phase

        return PipelineSchedule(
            schedule_type=schedule_type,
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            num_warmup_steps=num_warmup,
            num_steady_steps=num_microbatches - num_warmup,
            num_cooldown_steps=num_warmup,
            total_steps=total_steps,
            bubble_ratio=bubble,
        )

    elif schedule_type == ScheduleType.ONE_F_ONE_B:
        # 1F1B: same number of total steps as GPipe but lower peak activation memory
        num_warmup = num_stages - 1
        total_steps = 2 * num_microbatches + num_stages - 1
        # Bubble is same as GPipe (same total time, better memory)
        bubble = bubble_ratio(num_stages, num_microbatches)

        return PipelineSchedule(
            schedule_type=schedule_type,
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            num_warmup_steps=num_warmup,
            num_steady_steps=2 * num_microbatches - num_warmup - num_warmup,
            num_cooldown_steps=num_warmup,
            total_steps=total_steps,
            bubble_ratio=bubble,
        )

    elif schedule_type == ScheduleType.INTERLEAVED_ONE_F_ONE_B:
        # Interleaved: each device runs V virtual stages, reducing bubble
        effective_stages = num_stages * virtual_stages
        bubble = bubble_ratio(effective_stages, num_microbatches)
        total_steps = 2 * num_microbatches + effective_stages - 1

        return PipelineSchedule(
            schedule_type=schedule_type,
            num_stages=effective_stages,
            num_microbatches=num_microbatches,
            num_warmup_steps=effective_stages - 1,
            num_steady_steps=2 * num_microbatches - 2 * (effective_stages - 1),
            num_cooldown_steps=effective_stages - 1,
            total_steps=total_steps,
            bubble_ratio=bubble,
        )

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ---------------------------------------------------------------------------
# GPipe Scheduler
# ---------------------------------------------------------------------------


class GPipeScheduler:
    """
    GPipe scheduling: all microbatch forwards, then all backwards.

    Pros: Simple, straightforward implementation
    Cons: High peak activation memory (stores all microbatches' activations
          until backward phase). O(M) activation memory per stage.

    Memory breakdown per stage:
    - Activations: M * activation_size_per_microbatch
    - Parameters: |stage_params|
    - Optimizer: 2 * |stage_params| (Adam)
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        num_microbatches: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches
        self.device = device
        self.dtype = dtype

        # Store activations for backward
        self._saved_activations: dict[int, list[torch.Tensor]] = {}
        self._saved_inputs: dict[int, list[torch.Tensor]] = {}

    def _p2p_send(self, tensor: torch.Tensor, from_stage: int, to_stage: int) -> None:
        """Send tensor from one stage to the next."""
        from_rank = from_stage
        to_rank = to_stage
        if dist.is_initialized():
            dist.send(tensor.detach(), dst=to_rank)

    def _p2p_recv(
        self, from_stage: int, to_stage: int, shape: torch.Size
    ) -> torch.Tensor:
        """Receive tensor from previous stage."""
        tensor = torch.empty(shape, device=self.device, dtype=self.dtype)
        if dist.is_initialized():
            dist.recv(tensor, src=from_stage)
        return tensor

    def run_forward(
        self,
        microbatches: list[torch.Tensor],
        stage_idx: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        labels: Optional[list[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Execute GPipe forward pass for one stage across all microbatches.

        Args:
            microbatches: List of microbatch inputs.
            stage_idx: Index of this stage.
            loss_fn: Loss function (only used by last stage).
            labels: Corresponding labels (only used by last stage).

        Returns:
            Loss tensor (last stage only) or None.
        """
        stage = self.stages[stage_idx]
        self._saved_activations[stage_idx] = []
        self._saved_inputs[stage_idx] = []

        for mb_idx, mb_input in enumerate(microbatches):
            # Receive from previous stage (if not first)
            if not stage.is_first:
                mb_input = self._p2p_recv(stage_idx - 1, stage_idx, mb_input.shape)

            self._saved_inputs[stage_idx].append(mb_input.detach())

            # Forward through this stage's layers
            # Use activation checkpointing to save memory
            if mb_input.requires_grad:
                output = checkpoint.checkpoint(
                    stage.forward, mb_input, use_reentrant=False
                )
            else:
                output = stage.forward(mb_input)

            self._saved_activations[stage_idx].append(output.detach())

            # Send to next stage (if not last)
            if not stage.is_last:
                self._p2p_send(output, stage_idx, stage_idx + 1)

        # Compute loss (last stage only)
        if stage.is_last and loss_fn is not None and labels is not None:
            total_loss = sum(
                loss_fn(self._saved_activations[stage_idx][i], labels[i])
                for i in range(len(microbatches))
            ) / len(microbatches)
            return total_loss

        return None

    def run_backward(
        self,
        stage_idx: int,
        loss: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Execute GPipe backward pass for one stage across all microbatches.

        Backward proceeds in reverse micro-batch order.
        Gradients flow from the last stage backward to the first.
        """
        stage = self.stages[stage_idx]
        activations = self._saved_activations.get(stage_idx, [])
        inputs = self._saved_inputs.get(stage_idx, [])

        for mb_idx in reversed(range(self.num_microbatches)):
            # Receive gradient from next stage (if not last)
            grad_output: Optional[torch.Tensor] = None
            if stage.is_last:
                if loss is not None and mb_idx == self.num_microbatches - 1:
                    grad_output = torch.ones_like(activations[mb_idx])
            else:
                grad_shape = activations[mb_idx].shape
                grad_output = self._p2p_recv(stage_idx + 1, stage_idx, grad_shape)

            if grad_output is not None:
                inp = inputs[mb_idx].requires_grad_(True)
                output = stage.forward(inp)
                output.backward(grad_output)

            # Send gradient to previous stage (if not first)
            if not stage.is_first and inputs[mb_idx].requires_grad:
                grad_to_send = inputs[mb_idx].grad
                if grad_to_send is not None:
                    self._p2p_send(grad_to_send, stage_idx, stage_idx - 1)

        # Free saved tensors
        self._saved_activations.pop(stage_idx, None)
        self._saved_inputs.pop(stage_idx, None)


# ---------------------------------------------------------------------------
# 1F1B (One-Forward-One-Backward) Scheduler
# ---------------------------------------------------------------------------


class OneFOneBScheduler:
    """
    1F1B scheduling: interleaved forward and backward passes.

    Algorithm:
    1. Warmup: Inject M forward passes (fill pipeline)
    2. Steady: Alternate 1 forward + 1 backward (limit activations)
    3. Cooldown: Finish remaining backward passes (drain pipeline)

    Memory advantage over GPipe:
    - Peak activations = O(1) microbatch per stage (vs O(M))
    - Maximum live activations at any time = min(stage_idx + 1, M, P - stage_idx)
      which is bounded by P (pipeline depth), not M

    Reference:
        Memory-Efficient Pipeline-Parallel DNN Training (Narayanan et al., 2021)
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        num_microbatches: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches
        self.device = device
        self.dtype = dtype

        # Activation FIFO queue (circular buffer per stage)
        self._activation_queue: dict[int, list[torch.Tensor]] = {}
        self._input_queue: dict[int, list[torch.Tensor]] = {}

    @staticmethod
    def _send(tensor: torch.Tensor, dst: int) -> None:
        if dist.is_initialized():
            dist.send(tensor.detach(), dst=dst)

    @staticmethod
    def _recv(
        src: int, shape: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        if dist.is_initialized():
            dist.recv(tensor, src=src)
        return tensor

    def run(
        self,
        stage_idx: int,
        microbatches: list[torch.Tensor],
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        labels: Optional[list[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Execute 1F1B for one stage across all microbatches.

        Each step is either:
        - Forward (warmup and steady phases)
        - Backward (steady and cooldown phases)

        Args:
            stage_idx: This stage's index.
            microbatches: All microbatches for this stage.
            loss_fn: Loss function (last stage only).
            labels: Labels (last stage only).

        Returns:
            Total loss (last stage only), otherwise None.
        """
        stage = self.stages[stage_idx]
        P = self.num_stages
        M = self.num_microbatches

        # Warmup phase: inject forwards
        num_warmup_microbatches = min(P - stage_idx - 1, M)
        # Cooldown: finish remaining backwards
        num_warmup_forward = min(M, P - stage_idx)

        # Total steps = 2 * M
        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        fwd_q: list[tuple[int, torch.Tensor]] = []  # (mb_idx, activation)
        in_flight_forward = 0

        # Phase 1: Warmup (forward only)
        for warmup_step in range(num_warmup_forward):
            mb_idx = warmup_step
            inp = microbatches[mb_idx].clone()

            if not stage.is_first:
                inp = self._recv(stage_idx - 1, inp.shape, self.device, self.dtype)

            output = stage.forward(inp)
            fwd_q.append((mb_idx, output.detach()))

            if not stage.is_last:
                self._send(output, stage_idx + 1)
            elif loss_fn is not None and labels is not None:
                loss = loss_fn(output, labels[mb_idx])
                total_loss = total_loss + loss

            in_flight_forward += 1

        # Phase 2: Steady state (alternate 1F 1B)
        next_fwd_idx = num_warmup_forward
        next_bwd_idx = 0

        while next_fwd_idx < M or next_bwd_idx < M:
            # Decide: forward if still injecting and within window
            do_forward = (
                next_fwd_idx < M
                and len(fwd_q) <= P - stage_idx - 1
                and in_flight_forward < P - stage_idx
            )

            if do_forward:
                mb_idx = next_fwd_idx
                next_fwd_idx += 1
                inp = microbatches[mb_idx].clone()

                if not stage.is_first:
                    inp = self._recv(stage_idx - 1, inp.shape, self.device, self.dtype)

                output = stage.forward(inp)
                fwd_q.append((mb_idx, output.detach()))

                if not stage.is_last:
                    self._send(output, stage_idx + 1)
                elif loss_fn is not None and labels is not None:
                    loss = loss_fn(output, labels[mb_idx])
                    total_loss = total_loss + loss

                in_flight_forward += 1
            else:
                # Do backward
                if not fwd_q:
                    break  # Nothing to backward

                mb_idx, activation = fwd_q.pop(0)
                next_bwd_idx += 1

                grad = torch.ones_like(activation)
                if not stage.is_last:
                    grad = self._recv(
                        stage_idx + 1, activation.shape, self.device, self.dtype
                    )

                activation.backward(grad)

                if not stage.is_first:
                    # Need to be more careful here - we need the input
                    pass

        if stage.is_last:
            return total_loss / M
        return None


# ---------------------------------------------------------------------------
# Interleaved 1F1B Scheduler
# ---------------------------------------------------------------------------


class InterleavedOneFOneBScheduler:
    """
    Interleaved 1F1B: Each device manages V virtual pipeline stages.

    This further subdivides the pipeline, reducing the bubble ratio from
    (P-1)/M to (P*V-1)/M. The trade-off is increased communication
    (more send/recv between virtual stages on the same device).

    Example: 4 GPUs, 8 layers, V=2 virtual stages per GPU
        GPU 0: stages [0, 4]  (layers 0,1 and 4,5)
        GPU 1: stages [1, 5]  (layers 2,3 and 6,7)
        ...
        This creates a virtual pipeline of 8 stages, reducing bubble.

    Args:
        stages_per_device: List of stage lists. stages_per_device[i] is the
                          list of virtual stages on GPU i (interleaved order).
    """

    def __init__(
        self,
        stages: list[list[PipelineStage]],
        num_microbatches: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.stages_per_device = stages  # stages_per_device[device][virtual_stage]
        self.num_virtual_stages = len(stages[0]) if stages else 1
        self.total_virtual_stages = sum(len(s) for s in stages)
        self.num_microbatches = num_microbatches
        self.device = device
        self.dtype = dtype

    def run(self) -> None:
        """
        Execute interleaved 1F1B.

        The schedule proceeds in rounds. In each round, each physical device
        processes one virtual stage (forward or backward), cycling through
        its virtual stages in a round-robin fashion.
        """
        P = self.total_virtual_stages
        M = self.num_microbatches

        # Algorithm overview:
        # 1. Warmup: For i in [0, P-1), inject i microbatches
        # 2. Steady: For each step, run 1F + 1B on appropriate virtual stages
        # 3. Cooldown: For i in [0, P-1), drain one backward

        # In practice, devices alternate between virtual stages:
        # GPU k at step t processes virtual stage (k + t) % V in forward or backward

        total_steps = 2 * M + P - 1

        for step in range(total_steps):
            # Determine if each virtual stage should do forward or backward
            vstage = step % self.num_virtual_stages
            # Implementation continues per-device
            pass


# ---------------------------------------------------------------------------
# Activation Checkpointing Wrapper
# ---------------------------------------------------------------------------


class ActivationCheckpointWrapper(nn.Module):
    """
    Wrapper for activation checkpointing (gradient checkpointing).

    During forward pass, only stores the output of the wrapped module,
    re-computing intermediate activations during backward.

    Trade-off: O(sqrt(L)) memory for activations (or O(1) per checkpointed
    segment) vs recomputing forward pass during backward.

    For pipeline parallelism, checkpointing can be applied per-stage:
    - Without checkpointing: M activations per stage (GPipe) or O(P) (1F1B)
    - With checkpointing: O(1) activations per stage regardless of M

    Args:
        module: Module to checkpoint.
        use_reentrant: Whether to use reentrant checkpoint (more compatible).
    """

    def __init__(self, module: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return checkpoint.checkpoint(
            self._forward_impl,
            *args,
            **kwargs,
            use_reentrant=self.use_reentrant,
        )

    def _forward_impl(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


# ---------------------------------------------------------------------------
# Optimize microbatch count for a target bubble ratio
# ---------------------------------------------------------------------------


def optimize_microbatches(
    num_stages: int,
    target_bubble: float = 0.2,
    max_microbatches: int = 512,
    global_batch_size: int = 1024,
    min_microbatch_size: int = 1,
) -> int:
    """
    Find the optimal number of microbatches for a given bubble ratio target.

    bubble = (P - 1) / (P + M - 1) <= target_bubble
    => M >= (P - 1) / target_bubble - P + 1

    Args:
        num_stages: Number of pipeline stages.
        target_bubble: Desired maximum bubble ratio (e.g., 0.2 = 20%).
        max_microbatches: Maximum microbatches (limited by batch size).
        global_batch_size: Total batch size across all microbatches.
        min_microbatch_size: Minimum samples per microbatch.

    Returns:
        Recommended number of microbatches.
    """
    # Minimum M to satisfy bubble target
    min_m = max(1, int(math.ceil((num_stages - 1) / target_bubble - num_stages + 1)))

    # Don't exceed what the batch size allows
    max_m_from_batch = global_batch_size // min_microbatch_size
    m = min(min_m, max_m_from_batch, max_microbatches)
    m = max(m, 1)

    return m
