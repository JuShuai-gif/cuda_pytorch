"""Production service skeleton: config + logging + health check + graceful shutdown.

This is what a research script grows into.  Four things a production service
must have that a script does not:

1. configuration from the environment (see config.py)
2. structured logging (JSON lines, so a log collector can parse them)
3. a health check (so the orchestrator can probe liveness/readiness)
4. graceful shutdown (drain in-flight work, then exit cleanly)

The service wraps the Stage-16 inference server; the point here is the
*operational* surface, not the model math.
"""

from __future__ import annotations

import json
import signal
import threading
import time

from serving.inference_server.server import InferenceServer, make_model
from serving.production.config import ServiceConfig, load_config


class StructuredLogger:
    def __init__(self):
        self._lock = threading.Lock()

    def log(self, level: str, msg: str, **fields):
        entry = {"ts": time.time(), "level": level, "msg": msg, **fields}
        with self._lock:
            print(json.dumps(entry, ensure_ascii=False))


class ProductionService:
    def __init__(self, config: ServiceConfig | None = None):
        self.config = config or load_config()
        self.logger = StructuredLogger()
        self.healthy = True
        self.shutdown_requested = threading.Event()
        self.server = InferenceServer(
            make_model(), "cuda" if _cuda() else "cpu",
            strategy="dynamic", max_batch=self.config.max_batch,
            max_wait=self.config.max_wait_ms / 1000.0,
            max_queue=self.config.max_queue,
        )

    def health(self) -> dict:
        return {
            "status": "ok" if self.healthy else "unhealthy",
            "version": self.config.model_version,
            "model": self.config.model_path,
        }

    def start(self):
        self.logger.log("INFO", "service_started", config=self.config.to_dict())
        signal.signal(signal.SIGTERM, self._handle_term)
        signal.signal(signal.SIGINT, self._handle_term)

    def _handle_term(self, signum, frame):
        self.logger.log("INFO", "shutdown_signal", signal=signum)
        self.shutdown_requested.set()

    def run_until_shutdown(self):
        """Block until SIGTERM/SIGINT, then shut down gracefully."""
        while not self.shutdown_requested.is_set():
            time.sleep(0.1)
        self.shutdown()

    def shutdown(self):
        self.logger.log("INFO", "graceful_shutdown_begin")
        self.healthy = False
        self.server.stop()  # drain in-flight and stop the worker
        self.logger.log("INFO", "graceful_shutdown_complete")


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
