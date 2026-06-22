"""
Production-grade distributed training module.

Provides implementations of:
- Collective communication primitives (AllReduce, AllGather, ReduceScatter, etc.)
- Megatron-style Tensor Parallelism (ColumnLinear / RowLinear / TransformerLayer)
- Pipeline Parallelism (GPipe, 1F1B, Interleaved 1F1B with activation checkpointing)
- Fully Sharded Data Parallel (FSDP / ZeRO-3 with communication overlap)
- Sequence Parallelism (RingAttention, sequence-parallel embedding/loss)
- Mixture-of-Experts Parallelism (All-to-All token routing, load balancing)
- Parallel Strategy Planner (PTD-P optimization, memory estimation, MFU calculation)
"""

from cs336.distributed.collective_ops import (
    CollectiveOp,
    CommunicationBackend,
    benchmark_all_reduce,
    benchmark_bandwidth,
    bucket_gradients,
    create_broadcast,
    create_all_gather,
    create_reduce_scatter,
    create_all_reduce,
    create_all_to_all,
    GradientBucket,
    hierarchical_all_reduce,
    ring_all_reduce,
    tree_all_reduce,
    CommOverlapHelper,
)

from cs336.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelTransformerLayer,
    VocabularyParallelEmbedding,
    ColumnParallelLMHead,
    TensorParallelAttention,
    TensorParallelMLP,
    DeviceMesh,
    create_device_mesh,
)

from cs336.distributed.pipeline_parallel import (
    PipelineStage,
    make_pipeline_stages,
    GPipeScheduler,
    OneFOneBScheduler,
    InterleavedOneFOneBScheduler,
    ActivationCheckpointWrapper,
    bubble_ratio,
    PipelineSchedule,
    compute_pipeline_bubble,
)

from cs336.distributed.fsdp import (
    FSDPWrapper,
    FSDPConfig,
    wrap_fsdp,
    ZeROStage,
    OptimizerStateSharding,
)

from cs336.distributed.sequence_parallel import (
    SequenceParallel,
    RingAttention,
    SequenceParallelEmbedding,
    SequenceParallelCrossEntropyLoss,
    RingAttentionQKV,
    RingAttentionContext,
)

from cs336.distributed.expert_parallel import (
    MoEExpertParallel,
    ExpertRouter,
    AllToAllTokenDispatch as TokenDispatcher,
    LoadBalancedMoE,
    MoEConfig,
    compute_load_balancing_loss,
)

from cs336.distributed.parallel_planner import (
    ParallelConfig,
    HardwareConfig,
    ModelSpec,
    ParallelRecommendation,
    ParallelPlanner,
    compute_mfu,
    plan_parallel_strategy,
)

__all__ = [
    # collective_ops
    "ring_all_reduce",
    "tree_all_reduce",
    "hierarchical_all_reduce",
    "GradientBucket",
    "bucket_gradients",
    "benchmark_bandwidth",
    "benchmark_all_reduce",
    "CommOverlapHelper",
    "CollectiveOp",
    "CommunicationBackend",
    "create_broadcast",
    "create_all_gather",
    "create_reduce_scatter",
    "create_all_reduce",
    "create_all_to_all",
    # tensor_parallel
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelTransformerLayer",
    "VocabularyParallelEmbedding",
    "ColumnParallelLMHead",
    "TensorParallelAttention",
    "TensorParallelMLP",
    "DeviceMesh",
    "create_device_mesh",
    # pipeline_parallel
    "PipelineStage",
    "make_pipeline_stages",
    "GPipeScheduler",
    "OneFOneBScheduler",
    "InterleavedOneFOneBScheduler",
    "ActivationCheckpointWrapper",
    "bubble_ratio",
    "PipelineSchedule",
    "compute_pipeline_bubble",
    # fsdp
    "FSDPWrapper",
    "FSDPConfig",
    "wrap_fsdp",
    "ZeROStage",
    "OptimizerStateSharding",
    # sequence_parallel
    "SequenceParallel",
    "RingAttention",
    "SequenceParallelEmbedding",
    "SequenceParallelCrossEntropyLoss",
    "RingAttentionQKV",
    "RingAttentionContext",
    # expert_parallel
    "MoEExpertParallel",
    "ExpertRouter",
    "TokenDispatcher",  # alias for AllToAllTokenDispatch
    "LoadBalancedMoE",
    "MoEConfig",
    "compute_load_balancing_loss",
    "AllToAllTokenDispatch",
    # parallel_planner
    "ParallelConfig",
    "HardwareConfig",
    "ModelSpec",
    "ParallelRecommendation",
    "ParallelPlanner",
    "compute_mfu",
    "plan_parallel_strategy",
]
