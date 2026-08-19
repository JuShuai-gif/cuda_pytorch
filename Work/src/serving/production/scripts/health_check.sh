#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Probe the service's health endpoint.  Used by the orchestrator / liveness
# probe to decide whether to route traffic or restart the pod.
HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-8000}"

if curl -fsS "http://${HOST}:${PORT}/health"; then
  echo "healthy"
  exit 0
else
  echo "unhealthy" >&2
  exit 1
fi
