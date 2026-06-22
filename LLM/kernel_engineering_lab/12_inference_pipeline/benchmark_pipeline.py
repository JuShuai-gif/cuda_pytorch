"""
推理管线端到端 benchmark。

对比维度：
  1. eager（全部 PyTorch）vs fused（全部优化 kernel）TransformerBlock
  2. torch.compile vs 自定义融合实现
  3. 不同模型配置：hidden=512/1024/4096, heads=8/16/32, layers=4/8/16
  4. prefill 延迟：处理 32/128/512/1024 个 token
  5. decode 延迟：逐 token 生成 100 步
  6. 吞吐量：不同 batch size (1/4/8/16) 的 tokens/sec
  7. 峰值显存用量
  8. 生成综合性能报告

用法：
    python 12_inference_pipeline/benchmark_pipeline.py
    python 12_inference_pipeline/benchmark_pipeline.py --output report
    python 12_inference_pipeline/benchmark_pipeline.py --skip-compile
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_operator_fusion"))
sys.path.insert(0, str(_PROJECT_ROOT / "06_attention_flash_like"))
sys.path.insert(0, str(_PROJECT_ROOT / "02_triton_basics"))

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)

from kv_cache import KVCache
from pipeline import (
    InferencePipeline,
    OptimizedTransformer,
    TransformerBlock,
    _SimpleKVCachePipeline,
)

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

# 不同规模的模型配置
MODEL_CONFIGS = [
    {"hidden_dim": 512, "num_heads": 8, "head_dim": 64, "ffn_dim": 2048, "num_layers": 4},
    {"hidden_dim": 1024, "num_heads": 16, "head_dim": 64, "ffn_dim": 4096, "num_layers": 8},
    {"hidden_dim": 4096, "num_heads": 32, "head_dim": 128, "ffn_dim": 16384, "num_layers": 16},
]

# prefill 测试的 token 数量
PREFILL_LENS = [32, 128, 512, 1024]

# decode 测试的步数
DECODE_STEPS = 100

# 吞吐量测试的 batch size
THROUGHPUT_BATCHES = [1, 4, 8, 16]

# benchmark 时间配置
BENCH_CONFIG = BenchmarkConfig(warmup_steps=3, measure_steps=10, repeat=3)


# ---------------------------------------------------------------------------
# 辅助结构
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetric:
    """单次管线运行的多维度指标。"""

    name: str = ""
    prefill_ms: float = 0.0
    decode_per_step_ms: float = 0.0
    total_ms: float = 0.0
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    tokens_per_sec: float = 0.0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """获取 CUDA 设备名称，不可用时返回 'cpu'。"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _measure_memory_mb() -> float:
    """获取当前 CUDA 已分配显存（MB）。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _peak_memory_mb() -> float:
    """获取 CUDA 峰值显存（MB）。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_memory_stats() -> None:
    """重置 CUDA 显存统计。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 1. TransformerBlock eager vs fused 对比
# ---------------------------------------------------------------------------


def bench_block_eager_vs_fused() -> list[BenchmarkResult]:
    """对比 eager（全部 PyTorch）和 fused（全部优化 kernel）TransformerBlock。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 1: TransformerBlock eager vs fused")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    B = 4  # 固定 batch size

    for cfg in MODEL_CONFIGS:
        hidden = cfg["hidden_dim"]
        heads = cfg["num_heads"]
        h_dim = cfg["head_dim"]
        ffn = cfg["ffn_dim"]

        for seq_len in [64, 256]:
            cfg_name = f"h{hidden}_h{heads}_l{cfg['num_layers']}_s{seq_len}"

            # 复制权重以确保公平对比
            torch.manual_seed(42)
            block_eager = TransformerBlock(
                hidden_dim=hidden,
                num_heads=heads,
                head_dim=h_dim,
                ffn_dim=ffn,
                use_fusions=False,
            ).cuda()
            torch.manual_seed(42)
            block_fused = TransformerBlock(
                hidden_dim=hidden,
                num_heads=heads,
                head_dim=h_dim,
                ffn_dim=ffn,
                use_fusions=True,
            ).cuda()

            x = torch.randn(B, seq_len, hidden, device="cuda", dtype=torch.float32)
            # 在 GPU 上 warmup 一次以排除初始化开销
            _ = block_fused(x)
            _ = block_eager(x)
            torch.cuda.synchronize()

            results.append(
                benchmark_kernel(
                    block_eager,
                    args=(x,),
                    name=f"block_eager_{cfg_name}",
                    config=BENCH_CONFIG,
                )
            )
            results.append(
                benchmark_kernel(
                    block_fused,
                    args=(x,),
                    name=f"block_fused_{cfg_name}",
                    config=BENCH_CONFIG,
                )
            )

            # 计算加速比
            eager_t = results[-2].p50_ms
            fused_t = results[-1].p50_ms
            speedup = eager_t / fused_t if fused_t > 0 else 0
            print(
                f"  {cfg_name}: eager={eager_t:.3f}ms, fused={fused_t:.3f}ms, "
                f"speedup={speedup:.2f}x"
            )

    return results


# ---------------------------------------------------------------------------
# 2. torch.compile vs 自定义融合
# ---------------------------------------------------------------------------


def bench_torch_compile_vs_fused() -> list[BenchmarkResult]:
    """对比 torch.compile 优化和自定义 Triton 融合。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 2: torch.compile vs 自定义融合")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    B = 2

    # 使用中等规模配置进行对比
    cfg = MODEL_CONFIGS[1]  # hidden=1024, heads=16
    hidden = cfg["hidden_dim"]
    heads = cfg["num_heads"]
    h_dim = cfg["head_dim"]
    ffn = cfg["ffn_dim"]

    torch.manual_seed(42)
    block = TransformerBlock(
        hidden_dim=hidden,
        num_heads=heads,
        head_dim=h_dim,
        ffn_dim=ffn,
        use_fusions=False,
    ).cuda()

    # 创建 torch.compile 版本
    block_compiled = torch.compile(block, mode="reduce-overhead")

    for seq_len in [64, 256, 512]:
        x = torch.randn(B, seq_len, hidden, device="cuda", dtype=torch.float32)
        cfg_name = f"h{hidden}_h{heads}_s{seq_len}"

        # warmup torch.compile（第一次调用触发编译）
        _ = block_compiled(x)
        torch.cuda.synchronize()

        results.append(
            benchmark_kernel(
                block,
                args=(x,),
                name=f"torch_compile_eager_{cfg_name}",
                config=BENCH_CONFIG,
            )
        )
        results.append(
            benchmark_kernel(
                block_compiled,
                args=(x,),
                name=f"torch_compile_opt_{cfg_name}",
                config=BENCH_CONFIG,
            )
        )

        # 对比自定义融合版本
        torch.manual_seed(42)
        block_fused = TransformerBlock(
            hidden_dim=hidden,
            num_heads=heads,
            head_dim=h_dim,
            ffn_dim=ffn,
            use_fusions=True,
        ).cuda()
        _ = block_fused(x)
        torch.cuda.synchronize()

        results.append(
            benchmark_kernel(
                block_fused,
                args=(x,),
                name=f"custom_fused_{cfg_name}",
                config=BENCH_CONFIG,
            )
        )

        eager_t = results[-3].p50_ms
        compile_t = results[-2].p50_ms
        fused_t = results[-1].p50_ms
        print(f"  {cfg_name}: eager={eager_t:.3f}, compile={compile_t:.3f}, fused={fused_t:.3f} ms")

    return results


# ---------------------------------------------------------------------------
# 3. 不同模型配置对比
# ---------------------------------------------------------------------------


def bench_model_configs() -> list[BenchmarkResult]:
    """对比不同模型配置下的推理性能。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 3: 不同模型配置对比")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    B = 2
    seq_len = 64  # 固定序列长度以公平对比

    for cfg in MODEL_CONFIGS:
        hidden = cfg["hidden_dim"]
        heads = cfg["num_heads"]
        h_dim = cfg["head_dim"]
        ffn = cfg["ffn_dim"]
        nlayers = cfg["num_layers"]
        cfg_name = f"h{hidden}_h{heads}_l{nlayers}"

        model = OptimizedTransformer(
            num_layers=nlayers,
            hidden_dim=hidden,
            num_heads=heads,
            head_dim=h_dim,
            ffn_dim=ffn,
            use_fusions=True,
        ).cuda()

        x = torch.randn(B, seq_len, hidden, device="cuda", dtype=torch.float32)
        _ = model(x)
        torch.cuda.synchronize()

        result = benchmark_kernel(
            model,
            args=(x,),
            name=f"model_{cfg_name}",
            config=BENCH_CONFIG,
        )
        results.append(result)

        # 估算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {cfg_name}: {result.p50_ms:.3f}ms, params={total_params / 1e6:.1f}M")

    return results


# ---------------------------------------------------------------------------
# 4. prefill 延迟测试
# ---------------------------------------------------------------------------


def bench_prefill_latency() -> list[PipelineMetric]:
    """测试不同 prefill token 数量下的延迟。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 4: prefill 延迟（不同 prompt 长度）")
    print(f"{'=' * 60}")

    metrics: list[PipelineMetric] = []
    B = 1  # 单 batch 测量延迟

    cfg = MODEL_CONFIGS[1]  # medium: hidden=1024
    hidden = cfg["hidden_dim"]
    heads = cfg["num_heads"]
    h_dim = cfg["head_dim"]
    ffn = cfg["ffn_dim"]
    nlayers = cfg["num_layers"]

    model = OptimizedTransformer(
        num_layers=nlayers,
        hidden_dim=hidden,
        num_heads=heads,
        head_dim=h_dim,
        ffn_dim=ffn,
        use_fusions=True,
    ).cuda()

    pipeline = InferencePipeline(model)

    for prompt_len in PREFILL_LENS:
        cache = _SimpleKVCachePipeline(
            num_layers=nlayers,
            batch_size=B,
            num_heads=heads,
            max_seq_len=prompt_len + DECODE_STEPS,
            head_dim=h_dim,
        )

        x = torch.randn(B, prompt_len, hidden, device="cuda", dtype=torch.float32)

        # warmup
        _ = pipeline.prefill(x, cache)
        torch.cuda.synchronize()
        _reset_memory_stats()

        # 正式测量
        t_start = time.perf_counter()
        _ = pipeline.prefill(x, cache)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        mem_mb = _measure_memory_mb()
        peak_mb = _peak_memory_mb()

        m = PipelineMetric(
            name=f"prefill_len{prompt_len}",
            prefill_ms=elapsed * 1000,
            memory_mb=mem_mb,
            peak_memory_mb=peak_mb,
            tokens_per_sec=prompt_len / elapsed if elapsed > 0 else 0,
        )
        metrics.append(m)
        print(
            f"  prompt_len={prompt_len}: {m.prefill_ms:.2f}ms, "
            f"mem={mem_mb:.1f}MB, peak={peak_mb:.1f}MB, "
            f"{m.tokens_per_sec:.0f} tokens/s"
        )

    return metrics


# ---------------------------------------------------------------------------
# 5. decode 延迟测试
# ---------------------------------------------------------------------------


def bench_decode_latency() -> list[PipelineMetric]:
    """测试逐 token decode 的延迟。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 5: decode 延迟（逐 token 生成）")
    print(f"{'=' * 60}")

    metrics: list[PipelineMetric] = []
    B = 1
    prompt_len = 32

    cfg = MODEL_CONFIGS[1]
    hidden = cfg["hidden_dim"]
    heads = cfg["num_heads"]
    h_dim = cfg["head_dim"]
    ffn = cfg["ffn_dim"]
    nlayers = cfg["num_layers"]

    model = OptimizedTransformer(
        num_layers=nlayers,
        hidden_dim=hidden,
        num_heads=heads,
        head_dim=h_dim,
        ffn_dim=ffn,
        use_fusions=True,
    ).cuda()

    pipeline = InferencePipeline(model)
    cache = _SimpleKVCachePipeline(
        num_layers=nlayers,
        batch_size=B,
        num_heads=heads,
        max_seq_len=prompt_len + DECODE_STEPS + 100,
        head_dim=h_dim,
    )

    # prefill 先处理 prompt
    x_prompt = torch.randn(B, prompt_len, hidden, device="cuda", dtype=torch.float32)
    _ = pipeline.prefill(x_prompt, cache)
    torch.cuda.synchronize()

    # 测量 decode 延迟
    decode_times: list[float] = []
    for step in range(DECODE_STEPS):
        x_next = torch.randn(B, 1, hidden, device="cuda", dtype=torch.float32)
        t_start = time.perf_counter()
        _ = pipeline.decode_step(x_next, cache, step=prompt_len + step)
        torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - t_start) * 1000)

    avg_ms = sum(decode_times) / len(decode_times)
    p50_ms = sorted(decode_times)[len(decode_times) // 2]
    p90_ms = sorted(decode_times)[int(len(decode_times) * 0.9)]
    p99_ms = sorted(decode_times)[int(len(decode_times) * 0.99)]

    m = PipelineMetric(
        name="decode_100steps",
        decode_per_step_ms=avg_ms,
        total_ms=sum(decode_times),
        tokens_per_sec=1000 / avg_ms if avg_ms > 0 else 0,
    )
    metrics.append(m)
    print(
        f"  decode 100 steps: avg={avg_ms:.3f}ms/step, p50={p50_ms:.3f}ms, "
        f"p90={p90_ms:.3f}ms, p99={p99_ms:.3f}ms, "
        f"{m.tokens_per_sec:.1f} tokens/s"
    )

    return metrics


# ---------------------------------------------------------------------------
# 6. 吞吐量测试
# ---------------------------------------------------------------------------


def bench_throughput() -> list[PipelineMetric]:
    """测试不同 batch size 下的吞吐量。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 6: 吞吐量（不同 batch size）")
    print(f"{'=' * 60}")

    metrics: list[PipelineMetric] = []
    prompt_len = 64
    decode_len = 20  # 每次跑少量的 decode step 以缩短 benchmark

    cfg = MODEL_CONFIGS[0]  # 小模型以支持大 batch
    hidden = cfg["hidden_dim"]
    heads = cfg["num_heads"]
    h_dim = cfg["head_dim"]
    ffn = cfg["ffn_dim"]
    nlayers = cfg["num_layers"]

    for B in THROUGHPUT_BATCHES:
        try:
            model = OptimizedTransformer(
                num_layers=nlayers,
                hidden_dim=hidden,
                num_heads=heads,
                head_dim=h_dim,
                ffn_dim=ffn,
                use_fusions=True,
            ).cuda()

            pipeline = InferencePipeline(model)
            cache = _SimpleKVCachePipeline(
                num_layers=nlayers,
                batch_size=B,
                num_heads=heads,
                max_seq_len=prompt_len + decode_len + 10,
                head_dim=h_dim,
            )

            x = torch.randn(B, prompt_len, hidden, device="cuda", dtype=torch.float32)

            # warmup
            _ = pipeline.prefill(x, cache)
            torch.cuda.synchronize()
            _reset_memory_stats()

            # 测量 prefill + decode throughput
            t_start = time.perf_counter()
            for step in range(decode_len):
                x_next = torch.randn(B, 1, hidden, device="cuda", dtype=torch.float32)
                _ = pipeline.decode_step(x_next, cache, step=prompt_len + step)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start

            total_tokens = B * decode_len
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            mem_mb = _measure_memory_mb()

            m = PipelineMetric(
                name=f"throughput_batch{B}",
                total_ms=elapsed * 1000,
                tokens_per_sec=tokens_per_sec,
                memory_mb=mem_mb,
            )
            metrics.append(m)
            print(f"  batch={B}: {tokens_per_sec:.0f} tokens/s, mem={mem_mb:.1f}MB")

        except torch.cuda.OutOfMemoryError:
            print(f"  batch={B}: OOM（显存不足，跳过）")
            m = PipelineMetric(
                name=f"throughput_batch{B}_OOM",
                tokens_per_sec=0.0,
            )
            metrics.append(m)

    return metrics


# ---------------------------------------------------------------------------
# 7. 峰值显存测试
# ---------------------------------------------------------------------------


def bench_peak_memory() -> list[PipelineMetric]:
    """测量不同配置下的峰值显存用量。"""
    print(f"\n{'=' * 60}")
    print("  BENCH 7: 峰值显存用量")
    print(f"{'=' * 60}")

    metrics: list[PipelineMetric] = []
    B = 4
    seq_len = 256

    for cfg in MODEL_CONFIGS:
        hidden = cfg["hidden_dim"]
        heads = cfg["num_heads"]
        h_dim = cfg["head_dim"]
        ffn = cfg["ffn_dim"]
        nlayers = cfg["num_layers"]
        cfg_name = f"h{hidden}_h{heads}_l{nlayers}"

        try:
            _reset_memory_stats()

            model = OptimizedTransformer(
                num_layers=nlayers,
                hidden_dim=hidden,
                num_heads=heads,
                head_dim=h_dim,
                ffn_dim=ffn,
                use_fusions=True,
            ).cuda()

            model_mem = _measure_memory_mb()

            x = torch.randn(B, seq_len, hidden, device="cuda", dtype=torch.float32)
            input_mem = _measure_memory_mb() - model_mem

            _ = model(x)
            torch.cuda.synchronize()
            peak_mb = _peak_memory_mb()

            # KV cache 理论显存（假设分配所有层、所有 batch 的缓存）
            kv_cache_mem = nlayers * 2 * B * heads * seq_len * h_dim * 4 / (1024 * 1024)

            m = PipelineMetric(
                name=f"memory_{cfg_name}",
                memory_mb=model_mem,
                peak_memory_mb=peak_mb,
            )
            metrics.append(m)
            print(
                f"  {cfg_name}: model={model_mem:.1f}MB, peak={peak_mb:.1f}MB, "
                f"kv_cache_est={kv_cache_mem:.1f}MB"
            )

            del model, x
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  {cfg_name}: OOM（显存不足，跳过）")

    return metrics


# ---------------------------------------------------------------------------
# 8. 综合性能报告
# ---------------------------------------------------------------------------


def generate_pipeline_report(
    block_results: list[BenchmarkResult],
    compile_results: list[BenchmarkResult],
    model_results: list[BenchmarkResult],
    prefill_metrics: list[PipelineMetric],
    decode_metrics: list[PipelineMetric],
    throughput_metrics: list[PipelineMetric],
    memory_metrics: list[PipelineMetric],
    output_path: Optional[str],
) -> None:
    """生成综合性能报告（终端输出 + 可选文件输出）。"""
    print("\n")
    print("=" * 70)
    print("  综合推理管线性能报告")
    print("=" * 70)
    print(f"  设备: {_get_device()}")
    print(f"  PyTorch: {torch.__version__}")
    try:
        import triton

        print(f"  Triton: {triton.__version__}")
    except ImportError:
        print("  Triton: 不可用")
    print("=" * 70)

    # 汇总 compare_kernels
    all_bench_results = block_results + compile_results + model_results
    if all_bench_results:
        compare_kernels(all_bench_results)
        if output_path:
            generate_report(all_bench_results, output_path)

    # prefill 延迟表
    print(f"\n{'─' * 60}")
    print("  Prefill 延迟")
    print(f"{'─' * 60}")
    for m in prefill_metrics:
        print(
            f"  {m.name:30s} | {m.prefill_ms:8.2f} ms | "
            f"{m.tokens_per_sec:10.0f} tokens/s | "
            f"peak={m.peak_memory_mb:8.1f} MB"
        )

    # decode 延迟表
    print(f"\n{'─' * 60}")
    print("  Decode 延迟")
    print(f"{'─' * 60}")
    for m in decode_metrics:
        print(
            f"  {m.name:30s} | {m.decode_per_step_ms:8.3f} ms/step | "
            f"{m.tokens_per_sec:10.1f} tokens/s"
        )

    # 吞吐量表
    print(f"\n{'─' * 60}")
    print("  吞吐量")
    print(f"{'─' * 60}")
    for m in throughput_metrics:
        print(f"  {m.name:30s} | {m.tokens_per_sec:10.0f} tokens/s | mem={m.memory_mb:8.1f} MB")

    # 显存用量表
    print(f"\n{'─' * 60}")
    print("  峰值显存用量")
    print(f"{'─' * 60}")
    for m in memory_metrics:
        print(f"  {m.name:30s} | model={m.memory_mb:8.1f} MB | peak={m.peak_memory_mb:8.1f} MB")

    print(f"\n{'=' * 70}")
    print("  报告生成完毕")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_all_benchmarks(
    output_path: Optional[str] = None,
    skip_compile: bool = False,
) -> None:
    """运行所有推理管线 benchmark 并生成综合报告。"""
    print("\n" + "=" * 70)
    print("  推理管线端到端 BENCHMARK")
    print("=" * 70)
    print(f"  设备: {_get_device()}")
    print("=" * 70)

    t_total = time.perf_counter()

    # 1. eager vs fused
    block_results = bench_block_eager_vs_fused()

    # 2. torch.compile vs 自定义融合
    compile_results: list[BenchmarkResult] = []
    if not skip_compile:
        compile_results = bench_torch_compile_vs_fused()
    else:
        print(f"\n{'=' * 60}")
        print("  BENCH 2: torch.compile vs 自定义融合 - 已跳过")
        print(f"{'=' * 60}")

    # 3. 不同模型配置
    model_results = bench_model_configs()

    # 4. prefill 延迟
    prefill_metrics = bench_prefill_latency()

    # 5. decode 延迟
    decode_metrics = bench_decode_latency()

    # 6. 吞吐量
    throughput_metrics = bench_throughput()

    # 7. 峰值显存
    memory_metrics = bench_peak_memory()

    # 8. 综合报告
    generate_pipeline_report(
        block_results=block_results,
        compile_results=compile_results,
        model_results=model_results,
        prefill_metrics=prefill_metrics,
        decode_metrics=decode_metrics,
        throughput_metrics=throughput_metrics,
        memory_metrics=memory_metrics,
        output_path=output_path,
    )

    total_elapsed = time.perf_counter() - t_total
    print(f"\n  总耗时: {total_elapsed:.1f}s")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法运行 benchmark。")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="推理管线端到端 benchmark")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出报告基础路径（不含扩展名）",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="跳过 torch.compile 对比测试",
    )
    args = parser.parse_args()

    run_all_benchmarks(output_path=args.output, skip_compile=args.skip_compile)
