# LLM serving scheduler lab

Discrete-event simulation of continuous vs static batching and paged vs
contiguous KV cache (the two core vLLM mechanisms), since vLLM itself cannot
be installed here without upgrading torch (2.11 -> 2.13).

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/llm_scheduler/tests -v
$PY -m serving.llm_scheduler.benchmark --output /tmp/serving.json
```

## Headline

- continuous batching: TTFT p50 3.1s -> 0.09s (34x lower; requests don't wait
  for a full batch), throughput slightly higher.
- paged KV: serves 159 vs 128 concurrent requests, waste 2% vs 20%.

See `note/serving/01_llm_serving_vllm.md`.
