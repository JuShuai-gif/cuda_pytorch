"""12-factor configuration.

A research script hardcodes numbers; a production service reads them from the
environment so the same image can run anywhere (dev/staging/prod) with
different settings, and secrets never live in the code.  Each setting has a
type and a default, and an explicit docstring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class ServiceConfig:
    # Server
    host: str = field(default_factory=lambda: _env("SERVICE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("SERVICE_PORT", "8000")))
    # Model
    model_path: str = field(default_factory=lambda: _env("MODEL_PATH", "/models/model.engine"))
    model_version: str = field(default_factory=lambda: _env("MODEL_VERSION", "v1"))
    # Batching
    max_batch: int = field(default_factory=lambda: int(_env("MAX_BATCH", "8")))
    max_wait_ms: float = field(default_factory=lambda: float(_env("MAX_WAIT_MS", "5")))
    # Limits
    max_queue: int = field(default_factory=lambda: int(_env("MAX_QUEUE", "128")))
    request_timeout_s: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT_S", "10")))
    # Secrets are passed via env only (never defaults that leak).
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["api_key"] = "***" if d["api_key"] else ""  # never log the secret
        return d


def load_config() -> ServiceConfig:
    return ServiceConfig()
