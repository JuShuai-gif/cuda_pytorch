# TensorRT C++ lab

Full PyTorch -> ONNX -> TensorRT pipeline in C++ (the edge/runtime way).

- `python/export_onnx.py` - the only Python step (torch lives in flashrt):
  exports ONNX with dynamic batch/seq, saves reference I/O, benchmarks
  torch eager vs torch.compile.
- `src/build_engine.cpp` - ONNX -> serialized engine (FP32/FP16, optimization
  profile for dynamic shape).
- `src/run_engine.cpp` - deserialize -> set dynamic shape -> enqueueV3 ->
  correctness vs reference -> latency/throughput benchmark.

## Run

```bash
bash scripts/export_onnx.sh /tmp/trt_model          # torch -> ONNX
bash scripts/build.sh                               # cmake build
bash scripts/run_all.sh /tmp/trt_model 1 16         # build engines + benchmark
```

## Result (Thor/sm_110, batch=1 seq=16, residual MLP 4x1024)

| runtime      | latency (mean) | max diff | engine size |
|--------------|----------------|----------|-------------|
| torch eager  | 0.44 ms        | -        | -           |
| torch.compile| 0.51 ms        | -        | -           |
| TRT FP32     | 0.11 ms        | 6.7e-4   | 33.9 MB     |
| TRT FP16     | 0.09 ms        | 4.1e-3   | 17.1 MB     |

Dynamic shape works across batch 1/8/32; throughput scales from ~5.4k to ~66k
samples/s as batch grows.  See `note/inference/07_tensorrt.md`.
