# Benchmark Results

- Model: `examples/end_to_end/mlp.mlir`
- Iterations: 15

| metric | value |
|--------|-------|
| Compilation time (Edge→LLVM, wall) | 10.84 ms (min 9.64) |
| Optimization time (shape+fusion, wall) | 5.01 ms (min 4.15) |
| Runtime latency (pure compute) | 0.0068 ms |
| Throughput | 146771.0 inferences/s |
| Memory peak (naive→planned) | 6144 → 4608 bytes |

> Wall-clock numbers include process startup; runtime latency is the in-process Profiler measurement (pure kernel time).
