#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Start the production service with 12-factor env config.
# Secrets are passed via environment, never on the command line.
export PYTHONPATH="${PYTHONPATH:-$(pwd)/Work/src}"
export SERVICE_HOST="${SERVICE_HOST:-0.0.0.0}"
export SERVICE_PORT="${SERVICE_PORT:-8000}"
export MODEL_VERSION="${MODEL_VERSION:-v1}"
export MAX_BATCH="${MAX_BATCH:-8}"

exec python -m serving.production.benchmark
