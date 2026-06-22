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
    """生产级 GPU 延迟测量函数，避免常见 benchmark 误区。

    准确测量 GPU 延迟的关键规则：
    1. GPU 计时必须使用 torch.cuda.Event。CPU 侧的 `time.perf_counter()`
       不会等待 GPU kernel 执行完成，测到的是 host 侧发射开销，不是 device 真实耗时。
    2. 每次迭代后都要 synchronize，而不是只在最后同步。否则多次 GPU 执行可能重叠，
       得到虚假的高吞吐或低延迟。
    3. 正式计时前要跳过 warmup。CUDA JIT 编译、cuDNN auto-tune、显存分配等
       首次开销会显著污染延迟统计。
    4. 不要只报告 mean，要报告 p50、p99、p99.9。线上体验通常被尾延迟决定。
    5. 如果要测并发执行，应显式使用独立 CUDA stream，并单独设计 benchmark。
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    
    # 使用 CUDA Event 做 device 侧精确计时；比 CPU 计时更接近真实 kernel 耗时
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    # 预热阶段：触发 CUDA/cuDNN 的自动调优、JIT 编译和显存分配，避免污染正式计时
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

# 常见错误：用 CPU 时钟直接测 GPU 操作延迟
# 错误示例：
# start = time.perf_counter()
# _ = model(dummy)  # GPU kernel 可能还在异步执行，CPU 已经继续往下走
# elapsed = time.perf_counter() - start  # 这测到的是 HOST 发射时间，不是 DEVICE 执行时间