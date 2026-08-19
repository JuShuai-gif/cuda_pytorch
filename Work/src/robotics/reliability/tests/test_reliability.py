"""Correctness tests for watchdog and fault profiles.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/robotics/reliability/tests -v
"""

from __future__ import annotations

import unittest

from robotics.reliability.faults import FAULTS
from robotics.reliability.watchdog import InferenceProcess, Watchdog


class TestWatchdog(unittest.TestCase):
    def test_crash_triggers_restart(self):
        wd = Watchdog(lambda: InferenceProcess(crash_on_call=2))
        out1 = wd.guarded_infer()  # ok
        out2 = wd.guarded_infer()  # crashes -> restart + fallback
        self.assertEqual(out1, "action")
        self.assertEqual(out2, "safe_stop")
        self.assertEqual(wd.restarts, 1)

    def test_hang_triggers_fallback(self):
        wd = Watchdog(lambda: InferenceProcess(hang_seconds=10.0), timeout_s=0.05)
        self.assertEqual(wd.guarded_infer(), "safe_stop")
        self.assertEqual(wd.fallbacks, 1)

    def test_playbook_covers_all_faults(self):
        # master prompt lists these 9; the playbook must have all of them.
        expected = {"process_crash", "gpu_oom", "cuda_error", "model_load_failure",
                    "network_failure", "cloud_disconnect", "disk_full",
                    "memory_leak", "thermal_throttling"}
        self.assertEqual(set(FAULTS.keys()), expected)


if __name__ == "__main__":
    unittest.main()
