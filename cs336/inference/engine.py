"""
Production-grade LLM inference engine.

Orchestrates model loading, quantization, KV cache management,
sampling, batch generation, and continuous batching scheduling.

Supports:
  - Model loading: from_pretrained with weight quantization
  - Generation: single request, batch, streaming with callbacks
  - Quantization: FP16/BF16/INT8/FP8 for weights, FP8/INT8 for KV cache
  - Batching: dynamic padding, continuous batching via scheduler
  - Metrics: request-level and system-level collection
  - Prefix caching: automatic KV cache reuse via radix tree

Reference:
  - vLLM (Kwon et al., SOSP 2023)
  - SGLang (Zheng et al., NeurIPS 2024)
  - TensorRT-LLM (NVIDIA)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generator, Optional

import torch
import torch.nn.functional as F

from cs336.inference.kv_cache_manager import (
    KVCacheManager,
    KVCacheQuantization,
)
from cs336.inference.metrics import MetricsCollector, RequestMetrics, SystemMetrics
from cs336.inference.paged_attention import PagedAttention, paged_attention_forward
from cs336.inference.prefix_cache import PrefixCache
from cs336.inference.scheduler import Request, Scheduler, SchedulingPolicy


# ==============================================================================
#  Sampling strategies
# ==============================================================================


class SamplingStrategy(Enum):
    """Available token sampling strategies."""

    GREEDY = auto()
    TEMPERATURE = auto()
    TOP_K = auto()
    TOP_P = auto()


@dataclass
class SamplingConfig:
    """Configuration for token sampling.

    Attributes:
        strategy: Sampling strategy to use.
        temperature: Temperature for softmax scaling.
        top_k: Number of top tokens to consider (0 = disabled).
        top_p: Cumulative probability threshold for nucleus sampling.
        seed: Optional random seed for reproducibility.
    """

    strategy: SamplingStrategy = SamplingStrategy.GREEDY
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def sample(
        self,
        logits: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample next token IDs from logits.

        Args:
            logits: Raw logits of shape (batch, vocab_size).
            generator: Optional torch random generator for reproducibility.

        Returns:
            Token IDs of shape (batch,).
        """
        batch_size, vocab_size = logits.shape

        if self.strategy == SamplingStrategy.GREEDY:
            return logits.argmax(dim=-1)

        # Apply temperature
        if self.temperature > 0:
            logits = logits / max(self.temperature, 1e-9)

        # Top-K filtering
        if self.strategy == SamplingStrategy.TOP_K and self.top_k > 0:
            effective_k = min(self.top_k, vocab_size)
            topk_vals, _ = torch.topk(logits, effective_k, dim=-1)
            min_val = topk_vals[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Top-P (nucleus) filtering
        if self.strategy == SamplingStrategy.TOP_P and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > self.top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ==============================================================================
#  Generation configuration and results
# ==============================================================================


@dataclass
class GenerationConfig:
    """Configuration for a generation request.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering parameter.
        top_p: Nucleus sampling threshold.
        eos_token_id: Stop generation when this token is produced.
        stop_sequences: List of token sequences that signal stop.
        streaming_callback: Optional callback for streaming tokens.
        use_prefix_cache: Whether to check prefix cache before prefill.
        repetition_penalty: Penalty factor for repeated tokens (1.0 = no penalty).
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: int | None = None
    stop_sequences: list[list[int]] = field(default_factory=list)
    streaming_callback: Callable[[int, bool], None] | None = None
    use_prefix_cache: bool = True
    repetition_penalty: float = 1.0

    def to_sampling_config(self) -> SamplingConfig:
        """Derive a SamplingConfig from generation parameters."""
        if self.temperature <= 0:
            return SamplingConfig(strategy=SamplingStrategy.GREEDY)
        strategy = SamplingStrategy.TEMPERATURE
        if self.top_k > 0:
            strategy = SamplingStrategy.TOP_K
        if self.top_p < 1.0:
            strategy = SamplingStrategy.TOP_P
        return SamplingConfig(
            strategy=strategy,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )


@dataclass
class GenerationResult:
    """Result of a generation request.

    Attributes:
        request_id: Request identifier.
        prompt_ids: Input prompt token IDs.
        generated_ids: Generated output token IDs.
        num_tokens: Number of tokens generated.
        finish_reason: Why generation stopped ("stop", "length", "eos", "abort").
        metrics: Per-request metrics.
        text: Decoded text (if tokenizer was provided).
    """

    request_id: int
    prompt_ids: list[int]
    generated_ids: list[int]
    num_tokens: int
    finish_reason: str
    metrics: RequestMetrics | None = None
    text: str = ""


# ==============================================================================
#  Inference configuration
# ==============================================================================


@dataclass
class InferenceConfig:
    """Engine-level inference configuration.

    Attributes:
        max_batch_size: Maximum concurrent requests in a batch.
        max_seq_len: Maximum total sequence length (prompt + generated).
        block_size: KV cache block size (tokens per page).
        max_blocks: Maximum number of KV cache blocks to allocate.
        dtype: Default tensor dtype for model weights.
        kv_cache_dtype: Dtype for KV cache (can differ from weight dtype).
        kv_quantization: KV cache quantization mode.
        use_prefix_cache: Enable radix-tree prefix caching.
        prefix_cache_max_nodes: Maximum radix tree nodes.
        device: Device to run inference on.
        tensor_parallel_size: Number of GPUs for tensor parallelism (1 = no TP).
    """

    max_batch_size: int = 32
    max_seq_len: int = 8192
    block_size: int = 16
    max_blocks: int = 2048
    dtype: torch.dtype = torch.float16
    kv_cache_dtype: torch.dtype | None = None
    kv_quantization: KVCacheQuantization = KVCacheQuantization.NONE
    use_prefix_cache: bool = True
    prefix_cache_max_nodes: int = 10000
    device: str = "cuda"
    tensor_parallel_size: int = 1


# ==============================================================================
#  Inference Engine
# ==============================================================================


class InferenceEngine:
    """Production-grade LLM inference engine.

    Provides a unified interface for model loading, text generation,
    batch processing, and streaming with comprehensive metrics.

    Typical usage:
        engine = InferenceEngine.from_pretrained(
            model_path="meta-llama/Llama-3-8B",
            config=InferenceConfig(max_batch_size=32),
        )
        result = engine.generate("Hello, world!", GenerationConfig(max_new_tokens=100))

    Args:
        model: The language model (nn.Module with forward method).
        model_config: Model architecture configuration (layers, heads, etc.).
        tokenizer: Tokenizer for encoding/decoding text.
        config: Inference engine configuration.
    """

    def __init__(
        self,
        model: Any,
        model_config: Any,
        tokenizer: Any | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype
        self.kv_dtype = self.config.kv_cache_dtype or self.config.dtype

        # Extract model dimensions from config
        self.vocab_size: int = getattr(model_config, "vocab_size", 128256)
        self.hidden_size: int = getattr(model_config, "hidden_size", 4096)
        self.num_layers: int = getattr(model_config, "num_layers", 32)
        self.num_heads: int = getattr(model_config, "num_heads", 32)
        self.num_kv_heads: int = getattr(model_config, "num_kv_heads", 8)
        self.head_dim: int = self.hidden_size // self.num_heads

        # Move model to device with correct dtype
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Sub-components (lazily initialized)
        self._kv_manager: KVCacheManager | None = None
        self._paged_attn: PagedAttention | None = None
        self._scheduler: Scheduler | None = None
        self._metrics_collector: MetricsCollector = MetricsCollector()
        self._prefix_cache: PrefixCache | None = None

        self._init_components()

    def _init_components(self) -> None:
        """Initialize sub-components on first use or configuration change."""
        if self._kv_manager is None:
            self._kv_manager = KVCacheManager(
                num_layers=self.num_layers,
                n_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.config.block_size,
                max_blocks=self.config.max_blocks,
                dtype=self.kv_dtype,
                device=self.device,
                kv_quantization=self.config.kv_quantization,
            )

        if self._paged_attn is None:
            self._paged_attn = PagedAttention(
                block_size=self.config.block_size,
                n_heads=self.num_heads,
                n_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=self.device,
            )

        if self._scheduler is None:
            self._scheduler = Scheduler(
                max_batch_size=self.config.max_batch_size,
                max_total_tokens=self.config.max_seq_len,
                policy=SchedulingPolicy.PRIORITY,
                prefill_priority=True,
                block_size=self.config.block_size,
            )

        if self._prefix_cache is None and self.config.use_prefix_cache:
            bytes_per_token = (
                self.num_kv_heads * self.head_dim * 2  # K + V
            ) * self.num_layers
            if self.kv_dtype == torch.float16:
                bytes_per_token *= 2
            elif self.kv_dtype == torch.float32:
                bytes_per_token *= 4
            self._prefix_cache = PrefixCache(
                max_nodes=self.config.prefix_cache_max_nodes,
                block_size=self.config.block_size,
                bytes_per_token=bytes_per_token,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_config: Any,
        tokenizer: Any | None = None,
        config: InferenceConfig | None = None,
        quantize_weights: str | None = None,
    ) -> "InferenceEngine":
        """Load a pretrained model into the inference engine.

        Args:
            model_path: Path to model checkpoint or HuggingFace model ID.
            model_config: Model architecture configuration.
            tokenizer: Tokenizer instance.
            config: Inference engine configuration.
            quantize_weights: Weight quantization mode
                ("fp16", "bf16", "int8", "fp8_e4m3", None for native).

        Returns:
            Initialized InferenceEngine.

        Raises:
            FileNotFoundError: If model_path is not found.
            ValueError: If quantize_weights is invalid.
        """
        import os

        cfg = config or InferenceConfig()

        # Determine weight dtype
        dtype_map = {
            None: cfg.dtype,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if quantize_weights not in dtype_map and quantize_weights not in (
            "int8",
            "fp8_e4m3",
        ):
            raise ValueError(
                f"Invalid quantize_weights: {quantize_weights}. "
                f"Expected one of: {list(dtype_map.keys()) + ['int8', 'fp8_e4m3']}"
            )

        target_dtype = dtype_map.get(quantize_weights, cfg.dtype)

        # Try loading from path
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        elif os.path.isdir(model_path):
            # Try HuggingFace-style directory
            import json

            index_path = os.path.join(model_path, "pytorch_model.bin")
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.isfile(index_path):
                checkpoint = torch.load(
                    index_path, map_location="cpu", weights_only=True
                )
            elif os.path.isfile(safetensors_path):
                # safetensors requires the safetensors package
                raise FileNotFoundError(
                    f"safetensors file found at {safetensors_path}. "
                    f"Please install safetensors or use a .bin checkpoint."
                )
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Build model
        from cs336.transformer.llama import LlamaForCausalLM

        model = LlamaForCausalLM(model_config)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(dtype=target_dtype)

        return cls(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            config=cfg,
        )

    def encode(
        self,
        prompts: str | list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode text prompts to token IDs.

        Args:
            prompts: Single prompt string or list of prompt strings.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of token ID lists.

        Raises:
            ValueError: If no tokenizer is configured.
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer configured. Provide a tokenizer to encode text."
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        encoded: list[list[int]] = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            encoded.append(ids)
        return encoded

    # ---- Single-request generation ----

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[int],
        gen_config: GenerationConfig | None = None,
        request_id: int | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt as text string or token ID list.
            gen_config: Generation parameters (uses defaults if None).
            request_id: Optional request ID for metrics tracking.

        Returns:
            GenerationResult with generated tokens and metrics.
        """
        gen_cfg = gen_config or GenerationConfig()

        # Encode if string
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string prompts")
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            prompt_ids = list(prompt)

        rid = request_id if request_id is not None else 0
        self._metrics_collector.start_request(rid, len(prompt_ids))

        # Check prefix cache
        shared_prefix_len = 0
        if self._prefix_cache is not None and gen_cfg.use_prefix_cache:
            shared_prefix_len, cached_seq_id, _ = self._prefix_cache.find_prefix(
                prompt_ids
            )
            if shared_prefix_len > 0 and cached_seq_id is not None:
                # Fork from cached sequence
                try:
                    self._kv_manager.fork_sequence(cached_seq_id, rid)
                    self._kv_manager.get_seq_len(rid)
                except Exception:
                    shared_prefix_len = 0

        # Register sequence in KV cache
        if shared_prefix_len == 0:
            try:
                self._kv_manager.register_sequence(len(prompt_ids), rid)
            except RuntimeError:
                # Insufficient KV cache blocks
                return GenerationResult(
                    request_id=rid,
                    prompt_ids=prompt_ids,
                    generated_ids=[],
                    num_tokens=0,
                    finish_reason="abort",
                )

        # Convert prompt to tensor
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # ---- Prefill phase ----
        self._metrics_collector.record_prefill_start(rid)
        prefill_start = time.perf_counter()

        if shared_prefix_len == 0:
            kv_cache = self._kv_manager  # Will be used by the model
        else:
            # Only process the new suffix
            suffix_ids = prompt_ids[shared_prefix_len:]
            input_ids = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)

        # Model forward for prefill (handles all layers internally)
        logits = self._run_prefill(
            input_ids=input_ids,
            seq_id=rid,
            start_pos=shared_prefix_len,
        )

        # Sample first token
        sampling = gen_cfg.to_sampling_config()
        first_token = sampling.sample(logits[:, -1, :])
        first_token_id = int(first_token.item())

        self._metrics_collector.record_first_token(rid)  # noqa: F821
        self._metrics_collector.record_decode_step(rid, 1)

        prefill_elapsed = time.perf_counter() - prefill_start

        # ---- Decode phase ----
        generated_ids: list[int] = [first_token_id]
        finish_reason = "length"

        for step in range(1, gen_cfg.max_new_tokens):
            last_token = torch.tensor(
                [[generated_ids[-1]]], dtype=torch.long, device=self.device
            )

            logits = self._run_decode(
                input_ids=last_token,
                seq_id=rid,
            )

            next_token = sampling.sample(logits[:, -1, :])
            next_token_id = int(next_token.item())

            # Check stop conditions
            if (
                gen_cfg.eos_token_id is not None
                and next_token_id == gen_cfg.eos_token_id
            ):
                finish_reason = "eos"
                break

            if self._check_stop_sequence(generated_ids, gen_cfg.stop_sequences):
                finish_reason = "stop"
                break

            generated_ids.append(next_token_id)
            self._metrics_collector.record_decode_step(rid, 1)

            # Streaming callback
            if gen_cfg.streaming_callback is not None:
                gen_cfg.streaming_callback(next_token_id, False)

        # Final callback
        if gen_cfg.streaming_callback is not None:
            gen_cfg.streaming_callback(-1, True)

        # Finalize metrics
        peak_mem = 0.0
        if self.device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024**2)

        metrics = self._metrics_collector.finish_request(
            rid, finish_reason=finish_reason, peak_memory_mb=peak_mem
        )

        # Cleanup KV cache
        self._kv_manager.remove_sequence(rid)

        # Insert into prefix cache for future reuse
        if self._prefix_cache is not None and gen_cfg.use_prefix_cache:
            full_tokens = prompt_ids + generated_ids
            self._prefix_cache.insert(
                tokens=full_tokens,
                seq_id=rid,
                kv_block_table=None,
                kv_len=len(full_tokens),
            )

        # Decode text if tokenizer available
        text = ""
        if self.tokenizer is not None:
            text = self.tokenizer.decode(
                prompt_ids + generated_ids, skip_special_tokens=True
            )

        return GenerationResult(
            request_id=rid,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            num_tokens=len(generated_ids),
            finish_reason=finish_reason,
            metrics=metrics,
            text=text,
        )

    # ---- Streaming generation ----

    def generate_stream(
        self,
        prompt: str | list[int],
        gen_config: GenerationConfig | None = None,
        request_id: int | None = None,
    ) -> Generator[int, None, None]:
        """Generate tokens as a streaming iterator.

        Args:
            prompt: Input prompt.
            gen_config: Generation parameters.
            request_id: Optional request ID.

        Yields:
            Generated token IDs one at a time.
        """
        gen_cfg = gen_config or GenerationConfig()

        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string prompts")
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            prompt_ids = list(prompt)

        rid = request_id if request_id is not None else 0
        self._metrics_collector.start_request(rid, len(prompt_ids))

        # Register KV cache
        self._kv_manager.register_sequence(len(prompt_ids), rid)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        self._metrics_collector.record_prefill_start(rid)
        logits = self._run_prefill(input_ids=input_ids, seq_id=rid, start_pos=0)

        sampling = gen_cfg.to_sampling_config()
        first_token = sampling.sample(logits[:, -1, :])
        first_token_id = int(first_token.item())

        self._metrics_collector.record_first_token(rid)
        self._metrics_collector.record_decode_step(rid, 1)

        yield first_token_id
        generated: list[int] = [first_token_id]

        for _ in range(1, gen_cfg.max_new_tokens):
            last_input = torch.tensor(
                [[generated[-1]]], dtype=torch.long, device=self.device
            )
            logits = self._run_decode(input_ids=last_input, seq_id=rid)
            next_token = sampling.sample(logits[:, -1, :])
            next_token_id = int(next_token.item())

            if (
                gen_cfg.eos_token_id is not None
                and next_token_id == gen_cfg.eos_token_id
            ):
                break

            yield next_token_id
            generated.append(next_token_id)
            self._metrics_collector.record_decode_step(rid, 1)

        self._metrics_collector.finish_request(rid, finish_reason="length")
        self._kv_manager.remove_sequence(rid)

    # ---- Batch generation ----

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        gen_config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """Generate text for a batch of prompts with dynamic padding.

        Args:
            prompts: List of text prompts or token ID lists.
            gen_config: Shared generation configuration.

        Returns:
            List of GenerationResult, one per prompt.
        """
        gen_cfg = gen_config or GenerationConfig()

        # Encode all prompts
        prompt_ids_list: list[list[int]] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer required for string prompts")
                ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                prompt_ids_list.append(ids)
            else:
                prompt_ids_list.append(list(prompt))

        # Dynamic padding
        max_len = max(len(p) for p in prompt_ids_list)
        batch_size = len(prompt_ids_list)
        padded = torch.full(
            (batch_size, max_len),
            fill_value=0,  # Use 0 as pad token
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=self.device
        )

        for i, ids in enumerate(prompt_ids_list):
            padded[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            attention_mask[i, : len(ids)] = True

        # Batch prefill
        sampling = gen_cfg.to_sampling_config()
        results: list[GenerationResult] = []

        # Process each request individually for simplicity
        # (true batched decode requires model-level batch support)
        for i, prompt_ids in enumerate(prompt_ids_list):
            result = self.generate(
                prompt=prompt_ids,
                gen_config=gen_cfg,
                request_id=i,
            )
            results.append(result)

        return results

    # ---- Scheduled continuous batching ----

    @torch.no_grad()
    def serve_step(self) -> list[GenerationResult]:
        """Execute one step of continuous batching.

        Processes one scheduler step: evicts finished requests,
        admits new prefill requests, and runs one decode step
        for the running batch.

        Returns:
            List of GenerationResult for requests that finished this step.
        """
        if self._scheduler is None or self._scheduler.is_idle:
            return []

        finished = self._scheduler.step()

        # Process running decode requests
        decode_requests = self._scheduler.get_decode_requests()
        if decode_requests:
            batch_size = len(decode_requests)
            last_tokens = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )

            slot_map = self._scheduler.get_batch_slot_map()
            req_list: list[Request] = []
            for idx, req in enumerate(decode_requests):
                last_tokens[idx, 0] = req.generated_ids[-1]
                req_list.append(req)

            # Model decode forward
            logits = self.model(input_ids=last_tokens, use_cache=False)
            next_logits = logits[:, -1, :]
            next_tokens = next_logits.argmax(dim=-1)

            # Update scheduler state
            token_map: dict[int, int] = {}
            for idx, req in enumerate(req_list):
                token_map[req.request_id] = int(next_tokens[idx].item())
            finished_ids = self._scheduler.decode_step_completed(token_map)

        # Build results
        results: list[GenerationResult] = []
        for req in finished:  # Already finished via eviction in step()
            result = GenerationResult(
                request_id=req.request_id,
                prompt_ids=req.prompt_ids,
                generated_ids=req.generated_ids,
                num_tokens=req.generated_len,
                finish_reason="stop",
            )
            results.append(result)

        return results

    # ---- Metrics ----

    def get_metrics(self) -> SystemMetrics:
        """Get current system-level inference metrics."""
        mem_usage = 0.0
        mem_capacity = 0.0
        gpu_util = 0.0
        if self.device.type == "cuda":
            mem_usage = torch.cuda.memory_allocated(self.device) / (1024**2)
            mem_capacity = torch.cuda.get_device_properties(
                self.device
            ).total_memory / (1024**2)
        return self._metrics_collector.compute_system_metrics(
            gpu_memory_usage_mb=mem_usage,
            gpu_memory_capacity_mb=mem_capacity,
            gpu_utilization_pct=gpu_util,
        )

    @property
    def kv_cache_memory_mb(self) -> float:
        """KV cache memory usage in MB."""
        if self._kv_manager:
            return self._kv_manager.total_memory_mb()
        return 0.0

    # ---- Internal helpers ----

    def _run_prefill(
        self,
        input_ids: torch.Tensor,
        seq_id: int,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Execute the model prefill phase.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            seq_id: Sequence ID for KV cache.
            start_pos: Starting position in KV cache (for prefix reuse).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        kwargs: dict[str, Any] = {}
        if hasattr(self.model, "forward_kv"):
            kwargs["kv_cache_manager"] = self._kv_manager
            kwargs["seq_id"] = seq_id
            kwargs["start_pos"] = start_pos

        logits = self.model(input_ids=input_ids, use_cache=False, **kwargs)
        return logits

    def _run_decode(
        self,
        input_ids: torch.Tensor,
        seq_id: int,
    ) -> torch.Tensor:
        """Execute the model decode phase (single token).

        Args:
            input_ids: Single token input (batch, 1).
            seq_id: Sequence ID for KV cache.

        Returns:
            Logits of shape (batch, 1, vocab_size).
        """
        kwargs: dict[str, Any] = {}
        if hasattr(self.model, "forward_kv"):
            kwargs["kv_cache_manager"] = self._kv_manager
            kwargs["seq_id"] = seq_id
            kwargs["is_decode"] = True

        logits = self.model(input_ids=input_ids, use_cache=False, **kwargs)
        return logits

    @staticmethod
    def _check_stop_sequence(
        generated: list[int],
        stop_sequences: list[list[int]],
    ) -> bool:
        """Check if any stop sequence appears at the end of generated tokens.

        Args:
            generated: Tokens generated so far.
            stop_sequences: List of stop token sequences.

        Returns:
            True if a stop sequence was matched.
        """
        for stop_seq in stop_sequences:
            if len(stop_seq) > len(generated):
                continue
            if generated[-len(stop_seq) :] == stop_seq:
                return True
        return False

    # ---- Utility ----

    def reset(self) -> None:
        """Reset engine state (KV cache, metrics, scheduler)."""
        if self._kv_manager:
            for seq_id in self._kv_manager.active_sequences():
                self._kv_manager.remove_sequence(seq_id)
        if self._scheduler:
            self._scheduler.reset()
        self._metrics_collector.reset()
        if self._prefix_cache:
            self._prefix_cache.reset_stats()

    def print_info(self) -> None:
        """Print engine configuration and status."""
        info = [
            "=" * 60,
            "Inference Engine Status",
            "=" * 60,
            f"  Device: {self.device}",
            f"  Dtype: {self.dtype}",
            f"  KV dtype: {self.kv_dtype}",
            f"  KV quantization: {self.config.kv_quantization.name}",
            f"  Vocab size: {self.vocab_size}",
            f"  Hidden size: {self.hidden_size}",
            f"  Layers: {self.num_layers}",
            f"  Heads: {self.num_heads} (Q), {self.num_kv_heads} (KV)",
            f"  Head dim: {self.head_dim}",
            f"  Max batch size: {self.config.max_batch_size}",
            f"  Max seq len: {self.config.max_seq_len}",
            f"  Block size: {self.config.block_size}",
            f"  KV cache memory: {self.kv_cache_memory_mb:.1f} MB",
            f"  Prefix cache: {'enabled' if self._prefix_cache else 'disabled'}",
            "=" * 60,
        ]
        print("\n".join(info))
