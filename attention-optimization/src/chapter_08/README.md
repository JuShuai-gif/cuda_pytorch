# Chapter 08: PagedAttention

Implementation of vLLM's PagedAttention.

## Files

- `mini_paged_attention.cpp` - C++ implementation with block table
- `paged_attention.py` - Python simulation
- `block_allocator.h` - Block management utilities

## Key Concepts

- Block Table (logical → physical mapping)
- Physical KV Cache in blocks
- Memory fragmentation elimination
- Block allocation / deallocation
