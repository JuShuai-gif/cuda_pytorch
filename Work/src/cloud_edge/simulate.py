"""Cloud-edge cooperation flow simulation.

Walks four flows end to end and reports what each tier did:

1. task dispatch   cloud schedules -> edge forwards -> robot executes
2. model update    cloud publishes v2 -> edge distributes -> robot applies
3. data upload     robot reports -> edge collects -> cloud stores
4. fault recovery  robot goes offline -> edge detects -> cloud reschedules
"""

from __future__ import annotations

from cloud_edge.architecture import Cloud, EdgeGateway, Robot, Task


def simulate() -> dict:
    cloud = Cloud()
    edge = EdgeGateway()
    robots = [Robot(f"robot_{i}") for i in range(3)]
    for r in robots:
        edge.register(r)

    events: list[str] = []

    # 1. Task dispatch.
    t = Task("t1", "robot_0", "pick_object")
    cloud.schedule(t)
    result = edge.forward(t)
    events.append(f"dispatch t1 -> robot_0: {result}")

    # 2. Model update (v2 rollout to all robots).
    cloud.publish_model("v2")
    for r in robots:
        r.apply_model("v2")
    events.append("model v2 rolled out to 3 robots")

    # 3. Data upload (robot reports a sensor reading).
    edge.collect("robot_0", "temp=65C")
    for entry in edge.telemetry:
        cloud.store_data(entry)
    events.append(f"data uploaded: {len(cloud.data_store)} entries")

    # 4. Fault recovery: robot_1 goes offline, its task is rescheduled.
    robots[1].online = False
    t2 = Task("t2", "robot_1", "navigate")
    cloud.schedule(t2)
    result = edge.forward(t2)
    if result == "offline":
        cloud.record_failure("robot_1", "offline")
        # reschedule to a healthy robot.
        t2.robot_id = "robot_2"
        result2 = edge.forward(t2)
        events.append(f"robot_1 offline -> t2 rescheduled to robot_2: {result2}")

    return {
        "events": events,
        "robot_versions": {r.robot_id: r.model_version for r in robots},
        "task_status": {tid: t.status for tid, t in cloud.tasks.items()},
        "failures": cloud.failures,
    }
