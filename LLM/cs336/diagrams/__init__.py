"""
CS336 Architecture Diagrams -- Visualization Module.

Provides Mermaid.js diagrams covering the full LLM system stack:
  - Architecture overview (complete system + data flow)
  - Training pipeline (FSDP + TP + PP 3D hybrid PTD-P)
  - Inference pipeline (continuous batching, PagedAttention, speculative decoding)
  - Data flow (DCLM pipeline, contamination checking)
  - Distributed topology (NVLink, InfiniBand, ring all-reduce, pipeline bubble)
  - GPU architecture (SM organization, memory hierarchy, warp execution, Tensor Core)
  - CUDA kernel optimization (coalescing, tiling, FlashAttention, bank conflicts)
  - Model architecture comparison (Llama 3, DeepSeek-V3, Mistral)

Usage:
    These .mermaid files can be rendered with:
    - mermaid-cli:    mmdc -i diagram.mermaid -o diagram.png
    - GitHub markdown (native Mermaid support)
    - VS Code extension: Markdown Preview Mermaid Support
    - Any online Mermaid editor (mermaid.live)

All diagrams use real numbers from the course GPU spec database
(V100/A100/H100/B200) and model configurations (Llama-style 7B scale).
"""

__all__ = [
    "architecture_overview",
    "training_pipeline",
    "inference_pipeline",
    "data_flow",
    "distributed_topology",
    "gpu_architecture",
    "cuda_kernel_optimization",
    "model_architecture",
]

_MERMAID_FILES: dict[str, str] = {
    "architecture_overview": "architecture_overview.mermaid",
    "training_pipeline": "training_pipeline.mermaid",
    "inference_pipeline": "inference_pipeline.mermaid",
    "data_flow": "data_flow.mermaid",
    "distributed_topology": "distributed_topology.mermaid",
    "gpu_architecture": "gpu_architecture.mermaid",
    "cuda_kernel_optimization": "cuda_kernel_optimization.mermaid",
    "model_architecture": "model_architecture.mermaid",
}


def _mermaid_path(name: str) -> str:
    """Return the full mermaid file path for a named diagram."""
    from pathlib import Path

    filename = _MERMAID_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown diagram: {name}. Available: {list(_MERMAID_FILES)}")
    return str(Path(__file__).parent / filename)


def list_diagrams() -> list[str]:
    """List all available diagram names."""
    return sorted(_MERMAID_FILES.keys())
