"""Watchdog: monitor an inference process, restart on crash, fallback on timeout.

The robot-side safety net.  Two failure modes are handled:

  crash   - the inference process dies; the watchdog restarts it
  timeout - an inference takes too long; the watchdog aborts and returns a
            fallback (a safe default action) instead of blocking the control
            loop forever

The watchdog keeps counts so operators can see how often it fired.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class InferenceProcess:
    crash_on_call: int = -1   # crash on this call index (-1 = never)
    hang_seconds: float = 0.0  # simulate a hang if > 0
    call_count: int = 0
    alive: bool = True

    def infer(self) -> str:
        self.call_count += 1
        if self.crash_on_call >= 0 and self.call_count == self.crash_on_call:
            self.alive = False
            raise RuntimeError("process crashed")
        if self.hang_seconds > 0:
            time.sleep(self.hang_seconds)
        return "action"


class Watchdog:
    def __init__(self, make_process, fallback_action: str = "safe_stop",
                 timeout_s: float = 0.5):
        self.make_process = make_process
        self.fallback_action = fallback_action
        self.timeout_s = timeout_s
        self.process = make_process()
        self.restarts = 0
        self.fallbacks = 0

    def guarded_infer(self) -> str:
        if not self.process.alive:
            self._restart()
        import threading

        result = {}

        def run():
            try:
                result["out"] = self.process.infer()
            except RuntimeError:
                result["crash"] = True

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=self.timeout_s)

        if "crash" in result:
            # Process crashed mid-call -> restart.
            self._restart()
            return self.fallback_action
        if "out" not in result:
            # Timed out -> abort and fallback.
            self.fallbacks += 1
            return self.fallback_action
        return result["out"]

    def _restart(self):
        self.restarts += 1
        self.process = self.make_process()
