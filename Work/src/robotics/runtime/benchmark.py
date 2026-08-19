"""Benchmark sensor sync strategies and demonstrate ROS-like primitives.

Run from the repo root:

    export PYTHONPATH="$PWD/Work/src"
    python -m robotics.runtime.benchmark --output /tmp/runtime.json
"""

from __future__ import annotations

import argparse
import json

from common.report import write_report
from robotics.runtime.ros_like import Action, ActionGoal, QoS, Service, Topic
from robotics.runtime.runtime import RobotRuntime, Sensor


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    # Sensors at different rates.
    sensors = [Sensor("camera", 30.0), Sensor("imu", 200.0), Sensor("joints", 100.0)]
    runtime = RobotRuntime(sensors)

    latest = runtime.control_loop(10.0, 20.0, strategy="latest")
    exact = runtime.control_loop(10.0, 20.0, strategy="exact")

    # ROS-like demo.
    topic = Topic("camera", QoS(reliable=True, depth=10))
    received = []
    topic.subscribe(lambda m: received.append(m))
    for i in range(5):
        topic.publish(f"frame_{i}")

    service = Service("reset", lambda req: f"reset {req} done")

    def exec_task(goal, feedback, active):
        for step in range(5):
            if not active[goal.goal_id]:
                return f"{goal.data}: canceled"
            feedback(f"step {step}")
        return f"{goal.data}: done"

    action = Action("navigate", exec_task)
    goal = ActionGoal("g1", "goto_point_A")
    result = action.send_goal(goal)

    report = {
        "kind": "robot_runtime",
        "sync_latest": latest,
        "sync_exact": exact,
        "topic_received": len(received),
        "service_result": service.call("arm"),
        "action_result": result,
        "action_feedback": action.feedback,
    }
    write_report(args.output, report)

    print("== sensor sync (10s, 20Hz control) ==")
    for r in [latest, exact]:
        print(f"  {r['strategy']:8s} cycles={r['cycles']:3d} skips={r['exact_skips']:3d} "
              f"staleness mean={r['mean_staleness_s']:.4f}s max={r['max_staleness_s']:.4f}s")
    print("== ROS-like ==")
    print(f"  topic: {len(received)} msgs received")
    print(f"  service: {service.call('arm')}")
    print(f"  action: {result}  feedback={action.feedback}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
