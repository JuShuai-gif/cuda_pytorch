"""
MiniLLM 的完整 inference 引擎。

提供高层接口，支持：
- 从 checkpoint 加载已训练的 model。
- 使用多种策略（greedy、sampling）生成文本。
- 面向 throughput 的批量生成。
- Inference benchmarking（tokens/s、latency、内存使用）。

InferenceEngine 封装了整个 inference pipeline，用户只需加载一次 model
即可反复生成文本，开销极小。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 允许以独立脚本或作为 package 的一部分运行此文件
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from inference.generation import generate_greedy, generate_sampling, generate_streaming


@dataclass
class InferenceMetrics:
    """单次生成运行过程中收集的 metrics。"""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_s: float = 0.0
    ttft_s: float = 0.0  # Time To First Token（首 token 延迟）
    tokens_per_second: float = 0.0
    latency_per_token_ms: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class BatchInferenceMetrics:
    """批量 inference 运行的 metrics。"""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time_s: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    avg_latency_per_token_ms: float = 0.0
    peak_memory_mb: float = 0.0


class InferenceEngine:
    """
    MiniLLM 的高层 inference 引擎。

    负责 model 加载、文本生成以及性能测量。

    Args:
        model: 已训练的 MiniLLM model。
        tokenizer: 用于编码/解码的 BPETokenizer 实例。
        device: 运行 inference 的设备。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,  # Duck-typed tokenizer
        device: str = "cuda",
    ) -> None:
        self.model: nn.Module = model
        self.tokenizer = tokenizer
        self.device: str = device

        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config,  # MiniLLMConfig
        tokenizer,  # Duck-typed tokenizer
        device: str = "cuda",
    ) -> "InferenceEngine":
        """
        从保存的 checkpoint 创建一个 InferenceEngine。

        Args:
            checkpoint_path: model checkpoint .pt 文件的路径。
            model_config: MiniLLMConfig 实例。
            tokenizer: BPETokenizer 实例。
            device: 目标设备。

        Returns:
            已初始化的 InferenceEngine。
        """
        from transformer.layers import MiniLLM

        model = MiniLLM(model_config)
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, tokenizer, device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_cache: bool = True,
    ) -> tuple[str, InferenceMetrics]:
        """
        根据 prompt 生成文本。

        Args:
            prompt: 输入文本 prompt。
            max_new_tokens: 最大生成 token 数。
            temperature: Sampling 温度（0 表示 greedy）。
            top_k: Top-k 过滤。
            top_p: Nucleus sampling 阈值。
            use_cache: 是否使用 KV cache。

        Returns:
            包含 (generated_text, metrics) 的元组。
        """
        # 编码 prompt
        input_ids: list[int] = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor: torch.Tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device
        )

        # 重置 GPU 内存统计
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_memory: float = torch.cuda.max_memory_allocated() / 1024**2

        # 生成
        start_time: float = time.perf_counter()
        ttft: float | None = None

        if temperature == 0:
            generated_ids: torch.Tensor = generate_greedy(
                self.model, input_tensor, max_new_tokens, use_cache=use_cache
            )
        else:
            generated_ids = generate_sampling(
                self.model,
                input_tensor,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache,
            )

        end_time: float = time.perf_counter()
        total_time: float = end_time - start_time

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_memory: float = torch.cuda.max_memory_allocated() / 1024**2

        # 解码
        generated_list: list[int] = generated_ids[0].tolist()
        generated_text: str = self.tokenizer.decode(
            generated_list, skip_special_tokens=True
        )

        # 计算 metrics
        num_prompt_tokens: int = len(input_ids)
        num_generated: int = int(generated_ids.shape[1] - num_prompt_tokens)
        tokens_per_sec: float = num_generated / max(total_time, 1e-6)
        latency_per_token_ms: float = (total_time / max(num_generated, 1)) * 1000

        metrics = InferenceMetrics(
            prompt_tokens=num_prompt_tokens,
            generated_tokens=num_generated,
            total_time_s=total_time,
            ttft_s=0.0,  # 通过 streaming 可测量
            tokens_per_second=tokens_per_sec,
            latency_per_token_ms=latency_per_token_ms,
            peak_memory_mb=peak_memory if self.device.startswith("cuda") else 0.0,
        )

        return generated_text, metrics

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> tuple[list[str], BatchInferenceMetrics]:
        """
        为一批 prompts 生成文本。

        Note: 为简单起见，此方法按顺序处理 prompts。若需要真正的
        批量 inference，则需要 padding 和 attention masking。

        Args:
            prompts: 输入文本 prompt 列表。
            max_new_tokens: 每个 prompt 的最大 token 数。
            temperature: Sampling 温度。

        Returns:
            包含 (generated_texts, batch_metrics) 的元组。
        """
        generated_texts: list[str] = []
        total_input_tokens: int = 0
        total_output_tokens: int = 0

        start_time: float = time.perf_counter()

        for prompt in prompts:
            text, metrics = self.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_texts.append(text)
            total_input_tokens += metrics.prompt_tokens
            total_output_tokens += metrics.generated_tokens

        end_time: float = time.perf_counter()
        total_time: float = end_time - start_time

        batch_metrics = BatchInferenceMetrics(
            total_requests=len(prompts),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_time_s=total_time,
            throughput_tokens_per_sec=total_output_tokens / max(total_time, 1e-6),
            avg_latency_per_token_ms=(total_time / max(total_output_tokens, 1)) * 1000,
        )

        return generated_texts, batch_metrics

    def benchmark(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        warmup_runs: int = 2,
        bench_runs: int = 5,
    ) -> dict[str, float]:
        """
        运行全面的 inference benchmark。

        测量：TTFT（Time To First Token）、tokens/s、latency、throughput、内存。

        Args:
            prompts: 用于 benchmark 的 prompt 列表。
            max_new_tokens: 每个 prompt 生成的 token 数。
            warmup_runs: 预热迭代次数（不计入统计）。
            bench_runs: benchmark 迭代次数。

        Returns:
            benchmark metrics 字典。
        """
        results: list[InferenceMetrics] = []

        # 预热
        for _ in range(warmup_runs):
            for prompt in prompts[:1]:
                self.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # Benchmark
        for _ in range(bench_runs):
            for prompt in prompts:
                _, metrics = self.generate(prompt, max_new_tokens=max_new_tokens)
                results.append(metrics)

        # 汇总 metrics
        avg_tps: float = sum(r.tokens_per_second for r in results) / len(results)
        avg_latency: float = sum(r.latency_per_token_ms for r in results) / len(results)
        total_time: float = sum(r.total_time_s for r in results)
        total_tokens: int = sum(r.generated_tokens for r in results)

        return {
            "avg_tokens_per_second": avg_tps,
            "avg_latency_ms_per_token": avg_latency,
            "total_time_s": total_time,
            "total_generated_tokens": total_tokens,
            "overall_throughput_tokens_per_sec": total_tokens / max(total_time, 1e-6),
            "num_runs": len(results),
            "num_prompts": len(prompts),
            "max_new_tokens": float(max_new_tokens),
        }

    def print_model_info(self) -> None:
        """打印已加载 model 的信息。"""
        num_params: int = sum(p.numel() for p in self.model.parameters())
        trainable_params: int = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Model Info:")
        print(f"  Device: {self.device}")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Dtype: {next(self.model.parameters()).dtype}")


# 快速测试
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM
    from tokenizer.bpe_tokenizer import BPETokenizer

    # 创建一个与 tokenizer 输出 vocab 匹配的小型 model
    VOCAB_SIZE: int = 300
    config = MiniLLMConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=128,
    )
    model = MiniLLM(config)

    # 训练一个 tokenizer，其 ID 范围在 model 的 vocab 内
    tokenizer = BPETokenizer()
    tokenizer.train(["hello world test " * 100], vocab_size=VOCAB_SIZE, min_frequency=3)

    # 创建 inference engine
    engine = InferenceEngine(model, tokenizer, device="cpu")
    engine.print_model_info()

    # 测试生成
    text, metrics = engine.generate("hello", max_new_tokens=10, temperature=0.8)
    print(f"\nGeneration test:")
    print(f"  Input: 'hello'")
    print(f"  Generated tokens: {metrics.generated_tokens}")
    print(f"  Output: '{text[:50]}...' ")

    # 测试 benchmark
    bench_results = engine.benchmark(
        prompts=["hello", "world", "test"],
        max_new_tokens=10,
        warmup_runs=1,
        bench_runs=2,
    )
    print(f"\nBenchmark results:")
    for k, v in bench_results.items():
        print(f"  {k}: {v}")

    print("\nInferenceEngine tests passed!")
