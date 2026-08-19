"""Theoretical LLM inference roofline: prefill vs decode.

Compute FLOPs, memory traffic, and arithmetic intensity for the two phases of
autoregressive generation, and classify each phase against a roofline model.
This is the quantitative basis for the whole LLM-serving design: prefill is
compute-bound, decode is memory-bandwidth-bound (KV cache reads).

Model: GPT-style, ``L`` layers, hidden ``d``, seq ``S``, batch ``B``, vocab
``V``.  Per token the parameter FLOPs are ~2*(12 d^2) (QKV 3d^2 + out d^2 +
MLP 8d^2) and the attention FLOPs are 2*d*S (scores + values).  fp16 weights
and activations throughout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhaseMetrics:
    phase: str
    flops: float          # total FLOPs for this phase
    memory_bytes: float   # bytes read+written for this phase
    arithmetic_intensity: float  # FLOPs per byte
    weight_bytes: float   # model weights (resident)
    kv_cache_bytes: float  # KV cache (accumulated)


def model_weight_bytes(L: int, d: int) -> float:
    # ~12 d^2 parameters per layer (QKV + out + MLP) + embedding/output.
    return L * 12 * d * d * 2  # fp16


def prefill_metrics(L: int, d: int, S: int, B: int, V: int) -> PhaseMetrics:
    # Per-token FLOPs ~ 2*(12 d^2 + 2 d S); over B*S tokens.
    flops = 2 * L * (12 * d * d + 2 * d * S) * B * S
    # Memory: read weights once per layer (amortized over B*S tokens) + read
    # activations.  The weights dominate for small batch.
    weight = model_weight_bytes(L, d)
    # KV cache written once (this phase).
    kv = 2 * L * S * d * 2 * B
    mem = weight + kv + B * S * d * 2  # weights + kv + input activation
    ai = flops / mem
    return PhaseMetrics("prefill", flops, mem, ai, weight, kv)


def decode_metrics(L: int, d: int, S: int, B: int, V: int) -> PhaseMetrics:
    # One new token per request: FLOPs ~ 2*L*(12 d^2 + 2 d S) per request.
    flops = 2 * L * (12 * d * d + 2 * d * S) * B
    weight = model_weight_bytes(L, d)
    kv = 2 * L * S * d * 2 * B
    # Memory: weights (amortized over B) + the entire KV cache (S tokens).
    mem = weight + kv + B * d * 2
    ai = flops / mem
    return PhaseMetrics("decode", flops, mem, ai, weight, kv)


def classify(ai: float, peak_flops: float, peak_bw: float) -> str:
    """Roofline classification: compute-bound vs memory-bound."""
    ridge = peak_flops / peak_bw  # arithmetic intensity at the ridge point
    if ai >= ridge:
        return "compute-bound"
    return "memory-bound"


def sweep(device_peak_flops: float, device_peak_bw_gbs: float,
          L=32, d=4096, B=8, V=32000, seqs=(128, 512, 2048, 8192)) -> list[dict]:
    """Sweep sequence length and classify prefill vs decode."""
    peak_bw = device_peak_bw_gbs * 1e9
    out = []
    for S in seqs:
        p = prefill_metrics(L, d, S, B, V)
        q = decode_metrics(L, d, S, B, V)
        out.append({
            "seq_len": S,
            "prefill_flops_g": p.flops / 1e9,
            "prefill_ai": p.arithmetic_intensity,
            "prefill_bound": classify(p.arithmetic_intensity, device_peak_flops, peak_bw),
            "decode_flops_m": q.flops / 1e6,
            "decode_ai": q.arithmetic_intensity,
            "decode_bound": classify(q.arithmetic_intensity, device_peak_flops, peak_bw),
            "kv_cache_mb": q.kv_cache_bytes / 1e6,
            "weight_mb": q.weight_bytes / 1e6,
        })
    return out
