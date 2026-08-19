# Inference latency/throughput benchmark

This module measures inference latency and throughput with explicit, honest
timing.  The core lesson of Stage 1 is that "runtime = X ms" is meaningless
without stating what was measured.

## Two notions of latency

`benchmark_latency.py` reports both, and the gap between them is the lesson:

- `wall` - wall-clock time bracketed by host `cudaDeviceSynchronize` calls.
  This is what an end-to-end client observes for one isolated request, and it
  includes Python, dispatcher, allocator and kernel-launch overhead.
- `event` - CUDA-event device time.  This isolates GPU execution and excludes
  host overhead, so on a launch-bound batch=1 model it reports a smaller
  number.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m inference.benchmark_latency \
  --device cuda --dtype float32 --hidden 1024 --layers 4 --batch 1 \
  --warmup 20 --iterations 200 --output /tmp/latency.json

$PY -m inference.benchmark_throughput \
  --device cuda --dtype float32 --hidden 1024 --layers 4 \
  --batch-sweep 1 8 64 256 --iterations 200 --output /tmp/throughput.json
```

Every report records environment/device metadata, the raw samples, and
mean/p50/p90/p95/p99.  Reports refuse to overwrite an existing file.

## Test

```bash
$PY -m unittest discover -s Work/src/inference/tests -v
```

## What the numbers should teach

1. For batch=1, `wall > event`: the delta is host launch overhead.
2. Sweeping `--batch-sweep` raises samples/s (better tensor-core utilization)
   but also raises per-request latency (each request waits longer in batch).
3. A small model can be *launch-bound*: GPU work is tiny, so the wall time is
   dominated by the CPU feeding kernels, not by the GPU executing them.

GPU utilization, SM utilization, DRAM throughput and occupancy are trace
metrics, not wall-clock guesses.  They are measured in `src/profiling/`, never
estimated from `perf_counter`.
