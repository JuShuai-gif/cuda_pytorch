"""Correctness tests for config and service.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/serving/production/tests -v
"""

from __future__ import annotations

import os
import unittest

from serving.production.config import ServiceConfig
from serving.production.service import ProductionService


class TestConfig(unittest.TestCase):
    def test_env_overrides_default(self):
        os.environ["MODEL_VERSION"] = "v9"
        cfg = ServiceConfig()
        self.assertEqual(cfg.model_version, "v9")
        del os.environ["MODEL_VERSION"]

    def test_secret_redacted(self):
        os.environ["API_KEY"] = "supersecret"
        cfg = ServiceConfig()
        d = cfg.to_dict()
        self.assertEqual(d["api_key"], "***")
        del os.environ["API_KEY"]


class TestService(unittest.TestCase):
    def test_health_check(self):
        svc = ProductionService(ServiceConfig())
        h = svc.health()
        self.assertEqual(h["status"], "ok")
        svc.shutdown()


if __name__ == "__main__":
    unittest.main()
