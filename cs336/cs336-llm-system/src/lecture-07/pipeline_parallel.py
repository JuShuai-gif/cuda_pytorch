"""
Simple pipeline parallel simulation.
Splits a model across different "devices" and demonstrates how micro-batches
are pipelined through the model stages.

Concepts demonstrated:
  - Model partition into stages
  - Micro-batch pipelining (GPipe / 1F1B scheduling)
  - Bubble time overhead
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Simple model to partition
# ---------------------------------------------------------------------------


class SimpleTransformerBlock(nn.Module):
    """A single transformer block for pipeline partition."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x + residual
        return x


class PipelineStage:
    """
    A single stage in a pipeline. Represents one or more layers
    assigned to a virtual device.
    """

    def __init__(self, name: str, layers: nn.ModuleList, device_id: int):
        self.name = name
        self.layers = layers
        self.device_id = device_id
        self.activations: list[
            torch.Tensor | None
        ] = []  # Forward activations stored for backward
        self.grads: list[Any] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass through this stage."""
        for layer in self.layers:
            x = layer(x)
        self.activations.append(x.detach())
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Run backward pass (simulated as returning grad unchanged)."""
        return grad


@dataclass
class PipelineSchedule:
    """Represents a pipeline execution schedule."""

    num_stages: int
    num_microbatches: int

    def gpipe_schedule(self) -> list[list[tuple[str, int]]]:
        """
        GPipe schedule: first inject all microbatches forward,
        then process all backward.
        """
        schedule: list[list[tuple[str, int]]] = []
        return schedule

    def one_f_one_b_schedule(self) -> list[list[tuple[str, int]]]:
        """1F1B schedule: alternate forward and backward to reduce memory."""
        return []


# =========================================================================
# Simulation
# =========================================================================


def demo_pipeline_stages() -> None:
    """Demonstrate splitting a model into pipeline stages."""
    print("=" * 60)
    print("Pipeline Parallelism - Model Partition Demo")
    print("=" * 60)

    num_layers = 8
    num_stages = 4
    layers_per_stage = num_layers // num_stages

    print(f"\nTotal layers: {num_layers}")
    print(f"Pipeline stages: {num_stages}")
    print(f"Layers per stage: {layers_per_stage}")
    print()

    # Create model and partition
    all_layers = nn.ModuleList([SimpleTransformerBlock(128) for _ in range(num_layers)])
    stages: list[PipelineStage] = []

    for stage_idx in range(num_stages):
        start = stage_idx * layers_per_stage
        end = start + layers_per_stage
        stage_layers = all_layers[start:end]
        stage = PipelineStage(
            name=f"Stage-{stage_idx}",
            layers=nn.ModuleList(stage_layers),
            device_id=stage_idx,
        )
        stages.append(stage)

    for stage in stages:
        num_params = sum(p.numel() for p in stage.layers.parameters())
        print(
            f"  {stage.name} (device {stage.device_id}): {len(stage.layers)} layers, {num_params:,} parameters"
        )

    # Memory per device
    total_params = sum(p.numel() for p in all_layers.parameters())
    per_device = total_params / num_stages
    print(f"\n  Total model parameters: {total_params:,}")
    print(f"  Parameters per device: {per_device:,.0f}")
    print(
        f"  Memory saving: {((total_params - per_device) / total_params) * 100:.0f}% vs full replica"
    )


def demo_gpipe_bubble() -> None:
    """Demonstrate the bubble overhead in GPipe scheduling."""
    print("\n" + "=" * 60)
    print("GPipe Bubble Overhead Analysis")
    print("=" * 60)

    num_microbatches = 8
    num_stages = 4

    # In GPipe, there are two phases:
    # 1. Warmup: stages start one by one (pipeline fill)
    # 2. Steady state: all stages busy
    # 3. Cooldown: stages finish one by one (pipeline drain)

    print(f"\nMicrobatches: {num_microbatches}, Stages: {num_stages}")

    # Each microbatch takes 1 time unit per stage (simplified)
    # Total time = (num_microbatches + num_stages - 1) * time_per_stage
    total_slots = num_microbatches + num_stages - 1
    useful_slots = num_microbatches * num_stages  # if perfect utilization
    actual_slots = total_slots * num_stages  # each stage active for total_slots

    bubble_slots = total_slots * num_stages - num_microbatches * num_stages
    bubble_ratio = bubble_slots / (total_slots * num_stages)

    print(f"\n  Total time slots: {total_slots}")
    print(f"  Perfect utilization slots: {useful_slots}")
    print(f"  Actual computation: {num_microbatches * num_stages}")
    print(f"  Bubble slots: {bubble_slots}")
    print(f"  Bubble ratio: {bubble_ratio:.1%}")
    print(f"\n  Formula: bubble = (S - 1) / (M + S - 1)")
    print(f"           where S = stages, M = microbatches")
    print(f"  With M={num_microbatches}, S={num_stages}: {bubble_ratio:.1%}")

    # Show schedule visually
    print(f"\n  GPipe Schedule Visualization (F=forward, B=backward):")
    print(f"  {'Time':>5}", end="")
    for t in range(total_slots + 1):
        print(f"{t:>5}", end="")
    print()

    for stage in range(num_stages):
        print(f"  S{stage:<4}", end="")
        for t in range(total_slots + 1):
            # Determine what this stage does at time t
            microbatch_idx = t - stage
            if 0 <= microbatch_idx < num_microbatches:
                print(f" F{stage}{microbatch_idx:<2}", end="")
            elif num_microbatches <= microbatch_idx < 2 * num_microbatches:
                b_idx = microbatch_idx - num_microbatches
                # backward happens after all forwards
                backward_t = b_idx + stage + num_microbatches
                if backward_t <= t <= backward_t:
                    b_actual = num_microbatches - 1 - b_idx
                    print(f" B{stage}{b_actual:<2}", end="")
                else:
                    print(f"{'':>5}", end="")
            else:
                print(f"{'':>5}", end="")
        print()

    print(f"\n  Key insight: increasing M reduces bubble ratio.")
    print(f"  M=8:  bubble={(num_stages - 1) / (8 + num_stages - 1):.1%}")
    print(f"  M=32: bubble={(num_stages - 1) / (32 + num_stages - 1):.1%}")
    print(f"  M=128:bubble={(num_stages - 1) / (128 + num_stages - 1):.1%}")


def demo_1f1b_schedule() -> None:
    """Demonstrate 1F1B (one forward, one backward) scheduling."""
    print("\n" + "=" * 60)
    print("1F1B (One-Forward-One-Backward) Scheduling")
    print("=" * 60)

    num_microbatches = 4
    num_stages = 3

    print(f"\nMicrobatches: {num_microbatches}, Stages: {num_stages}")
    print("\n1F1B Memory Advantage:")
    print(
        "  GPipe: stores activations for ALL microbatches until backward → O(M) memory"
    )
    print("  1F1B:  starts backward as soon as possible → O(1) peak activations")
    print("\n1F1B Schedule:")
    print("  Warmup: inject M forward passes (same as GPipe)")
    print("  Steady state: 1F then 1B, alternating")
    print("  Cooldown: finish remaining backward passes")

    # Show timeline
    total_time = 2 * num_microbatches + num_stages - 1
    print(f"\n  Timeline ({total_time} steps):")
    for t in range(total_time):
        activities = []
        for s in range(num_stages):
            f_idx = t - s
            b_idx = t - (s + num_microbatches)
            if 0 <= f_idx < num_microbatches:
                activities.append(f"S{s}:F{f_idx}")
            elif 0 <= b_idx < num_microbatches:
                activities.append(f"S{s}:B{b_idx}")
        if activities:
            print(f"  Step {t}: {', '.join(activities)}")
        else:
            print(f"  Step {t}: (idle)")


def demo_pipeline_simulation() -> None:
    """Simulate a forward pass through a pipelined model."""
    print("\n" + "=" * 60)
    print("Pipeline Forward Pass Simulation")
    print("=" * 60)

    hidden_size = 128
    num_stages = 3
    layers_per_stage = 2

    stages = []
    for s in range(num_stages):
        layers = nn.ModuleList(
            [SimpleTransformerBlock(hidden_size) for _ in range(layers_per_stage)]
        )
        stages.append(PipelineStage(f"Stage-{s}", layers, s))

    batch_size, seq_len = 2, 16
    num_microbatches = 4
    microbatch_size = batch_size // num_microbatches

    print(
        f"\nBatch size: {batch_size}, Microbatches: {num_microbatches}, Microbatch size: {microbatch_size}"
    )
    print(f"Stages: {num_stages}")

    # Simulate GPipe pipelining
    data = torch.randn(batch_size, seq_len, hidden_size)
    microbatches = data.chunk(num_microbatches, dim=0)

    print(f"\n--- GPipe Forward Pass ---")
    start_time = time.time()

    for mb_idx, mb in enumerate(microbatches):
        # Pass through all stages
        for stage_idx, stage in enumerate(stages):
            # Simulate communication delay between "devices"
            if stage_idx > 0:
                # In real pipeline, this would be a send/recv
                pass
            output = stage.forward(mb)
            mb = output
            print(
                f"  MB{mb_idx} Stage{stage_idx}: input shape → output shape {output.shape}"
            )

        if mb_idx < num_microbatches - 1:
            mb = microbatches[mb_idx + 1]

    elapsed = time.time() - start_time
    print(f"\n  Total simulated time: {elapsed:.4f}s")


def main() -> None:
    demo_pipeline_stages()
    demo_gpipe_bubble()
    demo_1f1b_schedule()
    demo_pipeline_simulation()


if __name__ == "__main__":
    main()
