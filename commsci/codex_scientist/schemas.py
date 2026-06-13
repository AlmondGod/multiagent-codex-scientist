from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CodexNode:
    node_id: str
    parent_id: str | None
    agent_id: str
    branch_id: str
    condition: str
    depth: int
    hypothesis: str
    action: dict[str, Any]
    metrics: dict[str, Any]
    command: list[str]
    workspace_path: str
    artifact_paths: list[str]
    critique_received: str | None = None
    memory: list[str] = field(default_factory=list)
    logs_summary: str = ""
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_AGENT_ARTIFACTS = [
    "branch_summary.json",
    "critique.md",
    "decision_change.json",
    "metrics_experiment_1.json",
    "metrics_experiment_2.json",
    "logs_experiment_1.txt",
    "logs_experiment_2.txt",
    "research_note.md",
    "review.json",
    "git_diff.patch",
]


REQUIRED_NODE_ARTIFACTS = [
    "node.json",
    "action.json",
    "worker_task.md",
    "memory.md",
    "metrics.json",
    "logs.txt",
]
