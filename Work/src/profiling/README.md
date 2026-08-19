# Nsight Systems / Nsight Compute drivers

Standard profiling workflow: `nsys` to find the slow kernel / CPU gap, `ncu`
to analyze why a specific kernel is slow.

## Run

```bash
export PROFILING_PYTHON=/home/guhaoran/miniconda3/envs/flashrt/bin/python

bash Work/src/profiling/scripts/run_nsys.sh \
  --hidden 1024 --layers 4 --batch 1 --steps 3 --output-root /tmp/nsys

bash Work/src/profiling/scripts/run_ncu.sh \
  --hidden 1024 --layers 4 --batch 1 --steps 3 --set basic --output-root /tmp/ncu
```

Both scripts create a timestamped run directory under `--output-root`,
write the exact `command.txt`, and never overwrite an existing report.

## Perf-counter permission (this machine)

`ncu` requires access to GPU performance counters.  On this Jetson/Thor the
driver sets `RmProfilingAdminOnly=1`, so non-root `ncu` fails with
`ERR_NVGPUCTRPERM`.  Until an admin sets it to 0 (or runs `ncu` under sudo),
NCU *kernel counter* results are **Not Validated**.  `nsys` timeline profiling
works without root.

## Files

- `profile_target.py` - NVTX-marked inference target (``h2d``, ``block_N``,
  ``postprocess`` ranges; underscore names because ``/`` is NCU range-stack
  syntax).
- `analyze_nsys.py` - parse ``nsys stats`` text into a small JSON summary.
- `scripts/run_nsys.sh` / `scripts/run_ncu.sh` / `scripts/common.sh`.

See `note/profiling/01_nsys_inference_profiling.md` and
`note/profiling/02_ncu_kernel_profiling.md`.
