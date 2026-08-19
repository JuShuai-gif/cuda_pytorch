# CUDA Graph vs normal launch lab

A launch-bound workload is a chain of tiny elementwise kernels.  Each kernel
does almost no GPU work, so wall time is dominated by CPU launch overhead.
CUDA Graphs capture the chain once and replay it with a single launch.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m kernel.cuda_graph.benchmark \
  --device cuda --n-ops 64 --n 1024 --iterations 100 --output /tmp/graph.json
```

## Test

```bash
$PY -m unittest discover -s Work/src/kernel/cuda_graph/tests -v
```

## What to expect

- `normal_wall_mean` is much larger than `graph_wall_mean` (host launch cost).
- `event` device time changes little: the GPU work is identical, so the win is
  host-side.  That "event barely moves while wall collapses" signature is how
  you recognize a launch-bound workload in a timeline.

## Nsight Systems

```bash
PROFILING_PYTHON=$PY \
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output /tmp/graph_nsys -- \
  $PY -m kernel.cuda_graph.profile_target --n-ops 64 --steps 5
```

See `note/kernel/02_cuda_async_and_stream.md` for the CUDA Graph model and
when it helps (batch=1 real-time inference) versus when it does not (dynamic
shapes, changing memory allocations).
