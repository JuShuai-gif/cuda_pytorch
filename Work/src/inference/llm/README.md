# LLM inference lab

Prefill vs decode roofline analysis for a single transformer layer.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/inference/llm/tests -v
$PY -m inference.llm.benchmark --device cuda --output /tmp/llm.json
```

## The split

prefill arithmetic intensity grows with seq (compute-bound, ~8-11 TFLOPS);
decode AI stays ~1.0 regardless of seq (memory-bound, KV cache reads).  See
`note/inference/09_llm_inference.md`.
