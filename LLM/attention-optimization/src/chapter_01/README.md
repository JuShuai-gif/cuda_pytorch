# Chapter 01: Attention Basics

## Implementation

- `naive_attention.py` - Pure Python implementation with detailed step-by-step comments
- `naive_attention.cpp` - C++ implementation for performance baseline

## Key Concepts

- Q, K, V projections
- Scaled Dot-Product Attention
- O(N^2) complexity analysis
- Memory analysis
- Cache behavior analysis

## Run

```bash
# Python
python naive_attention.py

# C++
g++ -O2 naive_attention.cpp -o naive_attention -std=c++17
./naive_attention
```

## Expected Output

The implementations verify correctness against PyTorch's `F.scaled_dot_product_attention`.
Wall-clock timing and memory usage are reported for different sequence lengths.
