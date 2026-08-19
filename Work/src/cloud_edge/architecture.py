"""Cloud-edge architecture components.

Three tiers and their responsibilities:

    Cloud          model registry, task scheduler, data service, monitor
    Edge Gateway   fan-out to robots, fan-in telemetry, local cache
    Robot          execute tasks, report state, apply model updates

Each component is a thin object with an event log, so the cooperation flows
(task dispatch, model update, data upload, fault recovery) are visible end to
end.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    robot_id: str
    payload: str
    status: str = "pending"   # pending | running | done | failed


@dataclass
class Robot:
    robot_id: str
    model_version: str = "v1"
    online: bool = True
    task_log: list[str] = field(default_factory=list)
    data_log: list[str] = field(default_factory=list)

    def execute(self, task: Task) -> str:
        if not self.online:
            task.status = "failed"
            return "offline"
        task.status = "done"
        self.task_log.append(f"{task.task_id}:{task.payload}")
        return "ok"

    def apply_model(self, version: str) -> str:
        self.model_version = version
        return "ok"


class EdgeGateway:
    def __init__(self):
        self.robots: dict[str, Robot] = {}
        self.telemetry: list[str] = []

    def register(self, robot: Robot):
        self.robots[robot.robot_id] = robot

    def forward(self, task: Task) -> str:
        robot = self.robots.get(task.robot_id)
        if robot is None or not robot.online:
            task.status = "failed"
            return "offline"
        return robot.execute(task)

    def collect(self, robot_id: str, data: str):
        self.telemetry.append(f"{robot_id}:{data}")


class Cloud:
    def __init__(self):
        self.model_registry: dict[str, str] = {"v1": "active"}
        self.tasks: dict[str, Task] = {}
        self.data_store: list[str] = []
        self.failures: list[str] = []

    def publish_model(self, version: str):
        self.model_registry[version] = "active"

    def schedule(self, task: Task):
        self.tasks[task.task_id] = task

    def store_data(self, entry: str):
        self.data_store.append(entry)

    def record_failure(self, robot_id: str, reason: str):
        self.failures.append(f"{robot_id}:{reason}")
