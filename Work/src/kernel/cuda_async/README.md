# CUDA async / stream / pinned-memory lab

Makes the host/device execution model observable: whether a copy blocks the
CPU, whether host memory is page-locked, and whether independent kernels
overlap across streams.

## Experiments

1. pageable vs pinned H2D copy bandwidth
2. blocking vs non-blocking H2D (does the CPU return before the copy finishes?)
3. single stream vs N streams for independent GEMM work

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m kernel.cuda_async.benchmark \
  --device cuda --bytes 67108864 --iterations 30 --output /tmp/async.json
```

## Test

```bash
$PY -m unittest discover -s Work/src/kernel/cuda_async/tests -v
```

## Nsight Systems

```bash
PROFILING_PYTHON=$PY \
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output /tmp/async_nsys -- \
  $PY -m kernel.cuda_async.profile_target --mat-size 512 --steps 5
```

## Platform note (Jetson/Thor unified memory)

Host and device share physical DRAM here, so H2D does not cross PCIe.  The
pinned-vs-pageable difference is about DMA efficiency and page migration, not
the large PCIe penalty of discrete GPUs.  The benchmark reports measured
numbers so this platform difference shows up instead of being assumed away.
