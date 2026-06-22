# Chapter 09: MQA / GQA

## Scope

This chapter demonstrates how Multi-Query Attention and Grouped-Query Attention reduce KV cache memory by reducing the number of KV heads while preserving many query heads.

## Files

- `mqa.cpp` - MQA KV cache memory reduction demo.
- `gqa.cpp` - GQA KV cache memory reduction demo.
- `CMakeLists.txt` - builds both examples.

## Build

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target mqa gqa -j
```

## Run

```bash
./chapters/mqa
./chapters/gqa
```

## Metrics To Report

| Metric | Why it matters |
|---|---|
| KV cache MB | Direct memory capacity cost. |
| Reduction ratio | Shows MQA/GQA benefit vs MHA. |
| KV heads | Determines bandwidth per decode step. |

## Engineering Notes

MQA/GQA changes model architecture and checkpoint shape. Runtime kernels need a `q_head -> kv_head` mapping and must avoid reloading the same KV head redundantly for adjacent query heads.
