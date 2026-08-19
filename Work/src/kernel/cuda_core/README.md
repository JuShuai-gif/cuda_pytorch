# CUDA execution model lab (native CUDA C++)

Stage 1 established the theory with PyTorch probes; this module verifies it at
the **native CUDA C++ level**, which is where edge/runtime inference code
(TensorRT plugins, custom kernels, robot runtime) actually lives.  Each binary
is a self-contained experiment that prints a JSON result.

## Build

```bash
bash Work/src/kernel/cuda_core/scripts/build.sh
```

Uses CMake + nvcc for `sm_110` (this machine's NVIDIA Thor).  The build passes
`-Xptxas -v`, so the per-kernel register/shared-memory usage is printed at
compile time -- that is the first-hand evidence for occupancy limits.

## Run

```bash
bash Work/src/kernel/cuda_core/scripts/run_all.sh
# or individual binaries:
Work/src/kernel/cuda_core/build/bin/coalescing
Work/src/kernel/cuda_core/build/bin/bank_conflict
Work/src/kernel/cuda_core/build/bin/occupancy
Work/src/kernel/cuda_core/build/bin/async_copy
Work/src/kernel/cuda_core/build/bin/stream_overlap
Work/src/kernel/cuda_core/build/bin/graph_launch
```

## Experiments

| binary | verifies |
|---|---|
| `coalescing` | contiguous vs strided global access -> effective bandwidth |
| `bank_conflict` | 0 / 2-way / 32-way shared-memory bank conflicts |
| `occupancy` | `__launch_bounds__` -> registers/thread -> achieved occupancy |
| `async_copy` | pinned vs pageable + blocking vs async H2D |
| `stream_overlap` | single vs multiple streams for under-utilizing kernels |
| `graph_launch` | normal launch vs `cudaGraphLaunch` |

## Inspect the compile chain

```bash
BIN=Work/src/kernel/cuda_core/build/bin/occupancy
cuobjdump --dump-resource-usage $BIN    # REG/STACK/SHARED/LOCAL per kernel
cuobjdump --dump-sass $BIN              # sm_110 machine code (SASS)
cuobjdump --dump-ptx  $BIN              # portable virtual ISA (PTX)
```

See `note/kernel/03_coalescing_and_bank_conflict.md` and
`note/kernel/04_occupancy_register_pressure.md`.
