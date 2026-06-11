# Chapter 16: vLLM Source Analysis

## Reading Guide

Refer to `source_reading_notes.md` for annotated source code walkthrough.

Key files to read:
1. `vllm/worker/worker.py` - Inference loop
2. `vllm/core/scheduler.py` - Continuous batching
3. `vllm/attention/ops/paged_attn.py` - PagedAttention
4. `vllm/core/block_manager.py` - Block management

## Call Example

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello, my name is"])
```
