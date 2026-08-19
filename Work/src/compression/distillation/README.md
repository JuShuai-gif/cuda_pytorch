# Distillation lab

Teacher -> student logit distillation on MNIST, in the few-shot regime where
the soft-label gain is most visible.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/compression/distillation/tests -v
$PY -m compression.distillation.benchmark --device cuda --output /tmp/distill.json
```

## Headline (Thor/sm_110, MNIST, 2000 training samples)

- teacher 97.7% (932k params, full data)
- student direct 90.5% (few-shot)
- student distilled 93.9% (+3.4%, same architecture)
- latency 1.4x smaller/faster

Temperature T=1..20: accuracy rises 0.914 -> 0.946 as the soft labels get
smoother.  See `note/compression/04_distillation.md`.
