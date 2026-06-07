"""
cs336.inference - Production-grade LLM inference engine.

Provides a complete inference serving stack:
  - InferenceEngine: Model loading, generation, streaming, batch processing
  - Scheduler: Continuous batching with priority-based scheduling
  - KVCacheManager: Paged KV cache memory management
  - PagedAttention: Non-contiguous KV cache attention computation
  - SpeculativeDecoder: Draft-target speculative decoding
  - PrefixCache: Radix-tree-based prefix caching with copy-on-write
  - MetricsCollector: Request-level and system-level metrics

Reference:
  - Kwon et al., "Efficient Memory Management for Large Language Model
    Serving with PagedAttention", SOSP 2023.
  - Leviathan et al., "Fast Inference from Transformers via Speculative
    Decoding", ICML 2023.
  - Zheng et al., "SGLang: Efficient Execution of Structured Language
    Model Programs", NeurIPS 2024.
"""

from __future__ import annotations

from cs336.inference.engine import (
    GenerationConfig,
    GenerationResult,
    InferenceConfig,
    InferenceEngine,
    SamplingConfig,
    SamplingStrategy,
)
from cs336.inference.kv_cache_manager import (
    BlockAllocator,
    BlockTable,
    KVBlock,
    KVCacheManager,
    KVCacheQuantization,
)
from cs336.inference.metrics import (
    MetricsCollector,
    RequestMetrics,
    SystemMetrics,
    visualize_request_lifecycle,
)
from cs336.inference.paged_attention import (
    PagedAttention,
    paged_attention_forward,
)
from cs336.inference.prefix_cache import (
    PrefixCache,
    RadixNode,
    RadixTree,
)
from cs336.inference.scheduler import (
    BatchSlot,
    Request,
    RequestState,
    Scheduler,
    SchedulingPolicy,
)
from cs336.inference.speculative_decoding import (
    SpeculativeDecoder,
    VerificationResult,
)

__all__ = [
    # engine
    "GenerationConfig",
    "GenerationResult",
    "InferenceConfig",
    "InferenceEngine",
    "SamplingConfig",
    "SamplingStrategy",
    # scheduler
    "BatchSlot",
    "Request",
    "RequestState",
    "Scheduler",
    "SchedulingPolicy",
    # kv_cache_manager
    "BlockAllocator",
    "BlockTable",
    "KVBlock",
    "KVCacheManager",
    "KVCacheQuantization",
    # paged_attention
    "PagedAttention",
    "paged_attention_forward",
    # speculative_decoding
    "SpeculativeDecoder",
    "VerificationResult",
    # prefix_cache
    "PrefixCache",
    "RadixNode",
    "RadixTree",
    # metrics
    "MetricsCollector",
    "RequestMetrics",
    "SystemMetrics",
    "visualize_request_lifecycle",
]
