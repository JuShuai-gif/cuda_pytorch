import torch
import time
import numpy as np
from typing import List, Tuple

def production_latency_benchmark(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    warmup: int = 50,
    repeat: int = 500,
) -> dict:
    """Production-grade latency measurement that avoids common pitfalls.
    
    Critical rules for accurate GPU timing:
    1. ALWAYS use torch.cuda.Event for GPU timing (CPU-side `time.perf_counter`
       does NOT wait for GPU to finish — it measures host time, not device time)
    2. ALWAYS synchronize after each iteration (not just at end) to prevent
       overlapping executions that inflate throughput numbers
    3. Skip the first several warmup iterations (CUDA JIT compilation,
       cudnn auto-tune, memory allocation warmup)
    4. Report p50, p99, p99.9, not just mean — tail latency kills user experience
    5. Use separate CUDA streams if you need to measure concurrent execution
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    
    # Use CUDA events for device-accurate timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    # Warmup: more iterations on GPU to trigger all auto-tuning
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()
    
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(repeat):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
    
    timings = np.array(timings)
    return {
        'mean_ms': round(np.mean(timings), 3),
        'std_ms': round(np.std(timings), 3),
        'p50_ms': round(np.percentile(timings, 50), 3),
        'p99_ms': round(np.percentile(timings, 99), 3),
        'p99.9_ms': round(np.percentile(timings, 99.9), 3),
        'min_ms': round(np.min(timings), 3),
        'max_ms': round(np.max(timings), 3),
    }

# Common error: measuring latency with CPU clock on GPU operations
# WRONG:
# start = time.perf_counter()
# _ = model(dummy)  # GPU may still be executing!
# elapsed = time.perf_counter() - start  # This measures HOST time, not DEVICE time