from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


V2_STAGES = [
    "literature_review",
    "initial_implementation",
    "baseline_tuning",
    "creative_research",
    "ablation_studies",
    "plot_aggregation",
    "paper_writeup",
    "review",
]


@dataclass
class RichCodexNode:
    node_id: str
    parent_id: str | None
    agent_id: str
    generation: int
    stage: str
    recipe_id: str | None
    inheritance_mode: str | None
    source_node_ids: list[str]
    patch_recipe_id: str | None
    knobs: dict[str, Any]
    metrics: dict[str, Any]
    action_path: str
    metrics_path: str
    logs_path: str
    code_diff_path: str | None
    rationale: str | None = None
    analysis: str | None = None
    is_buggy: bool = False
    plots: list[str] = field(default_factory=list)
    ablation_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageReport:
    name: str
    goals: list[str]
    total_nodes: int
    best_node_id: str | None
    best_score: float | None
    findings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
