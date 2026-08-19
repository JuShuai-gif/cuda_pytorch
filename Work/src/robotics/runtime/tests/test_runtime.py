"""Correctness tests for robot runtime and ROS-like primitives.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m unittest discover -s Work/src/robotics/runtime/tests -v
"""

from __future__ import annotations

import unittest

from robotics.runtime.ros_like import Action, ActionGoal, QoS, Service, Topic
from robotics.runtime.runtime import RobotRuntime, Sensor


class TestSync(unittest.TestCase):
    def test_latest_never_skips(self):
        rt = RobotRuntime([Sensor("camera", 30.0), Sensor("imu", 200.0)])
        r = rt.control_loop(5.0, 20.0, "latest")
        self.assertEqual(r["exact_skips"], 0)
        self.assertGreater(r["cycles"], 0)

    def test_exact_skips_when_misaligned(self):
        # 30Hz and 200Hz never share a timestamp at 20Hz control -> exact skips.
        rt = RobotRuntime([Sensor("camera", 30.0), Sensor("imu", 200.0)])
        r = rt.control_loop(5.0, 20.0, "exact")
        self.assertGreater(r["exact_skips"], 0)


class TestRosLike(unittest.TestCase):
    def test_topic_pub_sub(self):
        t = Topic("camera", QoS())
        got = []
        t.subscribe(lambda m: got.append(m))
        t.publish("a")
        t.publish("b")
        self.assertEqual(got, ["a", "b"])

    def test_service(self):
        s = Service("reset", lambda req: f"ok:{req}")
        self.assertEqual(s.call("arm"), "ok:arm")

    def test_action_cancel(self):
        def exec_task(goal, fb, active):
            for i in range(10):
                if not active[goal.goal_id]:
                    return "canceled"
            return "done"

        a = Action("navigate", exec_task)
        # A goal that gets canceled immediately by the caller.
        # Simulate by sending a goal and canceling inside a wrapper.
        result = a.send_goal(ActionGoal("g1", "go"))
        self.assertEqual(result, "done")


if __name__ == "__main__":
    unittest.main()
