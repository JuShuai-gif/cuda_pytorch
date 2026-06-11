# Chapter 15: xFormers Source Analysis

## Reading Guide

Refer to `source_reading_notes.md` for annotated source code walkthrough.

Key files to read:
1. `xformers/components/attention/_sdp_backend.py` - Dispatch logic
2. `xformers/components/attention/csrc/attention/attention_forward_generic.cu` - Generic CUDA
3. `xformers/ops/fmha/flash.py` - FlashAttention wrapper

## Call Example

```python
import xformers.ops as xops

# Standard usage
out = xops.memory_efficient_attention(Q, K, V)
```
