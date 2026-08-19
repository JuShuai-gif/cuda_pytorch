# Production inference service lab

Reliability primitives: token-bucket rate limit, circuit breaker, load
shedding, with an overload simulation.

## Run

```bash
export PYTHONPATH="$PWD/Work/src"
PY=/home/guhaoran/miniconda3/envs/flashrt/bin/python

$PY -m unittest discover -s Work/src/serving/production_service/tests -v
$PY -m serving.production_service.benchmark --output /tmp/prod.json
```

## Headline

10000 requests in 2s vs a 1000 req/s GPU: unprotected -> p99 9.9s (latency
explosion); load shedding -> p99 0.1s with 99% dropped.  See
`note/serving/03_production_service.md`.
