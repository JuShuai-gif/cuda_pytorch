# Final Project: Mini TensorRT-LLM Attention Engine

A minimal inference engine demonstrating all optimization techniques learned.

## Architecture

```
final_project/
├── include/            # Header files
│   ├── attention_engine.h
│   ├── flash_attn.h
│   ├── kv_cache.h
│   ├── paged_attn.h
│   └── config.h
├── src/
│   ├── attention_engine.cu     # Main attention engine
│   ├── flash_attn.cu           # FlashAttention V2 kernel
│   ├── kv_cache.cu             # KV Cache with block table
│   ├── paged_attn.cu           # PagedAttention kernel
│   └── batcher.cu              # Continuous batching scheduler
├── python/
│   ├── engine.py               # Python wrapper
│   └── benchmark.py            # End-to-end benchmark
├── tests/
│   └── test_attention.cpp      # Correctness tests
├── CMakeLists.txt
└── README.md
```

## Features

- [ ] FlashAttention V2 integration
- [ ] KV Cache management with block table
- [ ] PagedAttention for memory-efficient serving
- [ ] MQA / GQA support (configurable KV head ratio)
- [ ] Continuous batching scheduler
- [ ] Llama-style inference loop
- [ ] Performance comparison vs PyTorch

## Build & Run

```bash
cd final_project
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j$(nproc)

# Run tests
./tests/test_attention

# Run benchmark
python ../python/benchmark.py
```

## Performance Targets (A100)

| Model size | Seq len | Latency (ms/tok) | Throughput (tok/s) |
|-----------|---------|------------------|-------------------|
| 7B        | 2048   | < 15ms          | > 500             |
| 7B        | 4096   | < 20ms          | > 300             |
